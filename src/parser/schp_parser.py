"""
SCHP Parser - Production-grade wrapper for Self-Correction Human Parsing.
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import torchvision.transforms as transforms

from .model_loader import ModelLoader, DATASET_SETTINGS
from ..processing.utils import get_affine_transform

logger = logging.getLogger(__name__)


class SCHPParser:
    """Production-grade SCHP parser with error handling and optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model_name: str = 'atr', 
                 checkpoint_path: Optional[str] = None, device: str = 'auto', 
                 use_gpu: Optional[bool] = None, gpu_id: int = 0):
        """
        Initialize SCHP parser.
        
        Args:
            config: Configuration dictionary (optional)
            model_name: Model name ('atr', 'lip', 'pascal')
            checkpoint_path: Path to model checkpoint
            device: Device selection ('auto', 'cuda', 'cpu')
            use_gpu: Use GPU if available (deprecated, use 'device' instead)
            gpu_id: GPU device ID (only used if device='cuda')
        """
        if config is None:
            from .model_loader import load_config
            config = load_config()
        
        # Override config with parameters
        if model_name:
            config['model']['name'] = model_name
        if checkpoint_path:
            config['model']['checkpoint_path'] = checkpoint_path
        if device and device != 'auto':
            config['model']['device'] = device
        elif use_gpu is not None:
            # Legacy support: convert use_gpu to device
            config['model']['device'] = 'cuda' if use_gpu else 'cpu'
            config['model']['use_gpu'] = use_gpu
        if gpu_id is not None:
            config['model']['gpu_id'] = gpu_id
        
        self.config = config
        self.model_loader = ModelLoader(config)
        self.model_name = self.model_loader.model_name
        self.num_classes = self.model_loader.num_classes
        self.input_size = self.model_loader.input_size
        self.labels = self.model_loader.labels
        self.device = self.model_loader.device
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        
        logger.info(f"SCHP Parser initialized: {self.model_name}, device: {self.device}")
    
    def parse_image(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Parse a single image.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Tuple of (parsed_image, metadata)
            - parsed_image: PIL Image with label values
            - metadata: Dict with parsing info
        """
        try:
            # Convert PIL to numpy
            img_array = np.array(image.convert('RGB'))
            h, w = img_array.shape[:2]
            
            # Get person center and scale (matching SCHP simple_extractor_dataset.py)
            # Use full image as bounding box [0, 0, w-1, h-1]
            # Then adjust aspect ratio to match input_size
            aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]  # width/height ratio
            
            # Calculate center (from box [0, 0, w-1, h-1])
            person_center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float32)
            
            # Adjust scale to match aspect ratio (like _xywh2cs does)
            scale_w = float(w - 1)
            scale_h = float(h - 1)
            if scale_w > aspect_ratio * scale_h:
                scale_h = scale_w / aspect_ratio
            elif scale_w < aspect_ratio * scale_h:
                scale_w = scale_h * aspect_ratio
            scale = np.array([scale_w, scale_h], dtype=np.float32)
            
            # Prepare input
            trans = get_affine_transform(person_center, scale, 0, self.input_size)
            input_img = cv2.warpAffine(
                img_array,
                trans,
                (int(self.input_size[1]), int(self.input_size[0])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            
            # Transform to tensor
            input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
            
            # Run inference
            model = self.model_loader.model
            with torch.no_grad():
                output = model(input_tensor)
                
                # Get the last output (self-correction output)
                upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
                
                # Transform back to original size using transform_logits
                SCHP_ROOT = Path(__file__).absolute().parent.parent.parent / "Self-Correction-Human-Parsing"
                if not SCHP_ROOT.exists():
                    raise RuntimeError("SCHP repository not found")
                
                import sys
                schp_path = str(SCHP_ROOT)
                if schp_path not in sys.path:
                    sys.path.insert(0, schp_path)
                
                # Import transform_logits - it should be available
                try:
                    # Import directly from SCHP utils using importlib
                    import importlib.util
                    transforms_path = SCHP_ROOT / "utils" / "transforms.py"
                    if not transforms_path.exists():
                        raise ImportError(f"transforms.py not found at {transforms_path}")
                    
                    # Load the module
                    spec = importlib.util.spec_from_file_location("utils.transforms", transforms_path)
                    transforms_module = importlib.util.module_from_spec(spec)
                    
                    # Ensure SCHP_ROOT is in path for any imports within transforms.py
                    if schp_path not in sys.path:
                        sys.path.insert(0, schp_path)
                    
                    # Execute the module
                    spec.loader.exec_module(transforms_module)
                    transform_logits = transforms_module.transform_logits
                    
                    # Use transform_logits for proper affine transformation
                    logits_np = upsample_output.data.cpu().numpy()
                    logits_result = transform_logits(
                        logits_np,
                        person_center,
                        scale,
                        w,
                        h,
                        input_size=self.input_size
                    )
                    # Get parsing result
                    parsing_result = np.argmax(logits_result, axis=2).astype(np.uint8)
                    logger.info("✓ Used transform_logits for proper affine transformation")
                except Exception as e:
                    logger.warning(f"transform_logits failed: {e}, using direct resize (less accurate)")
                    # Fallback: simple resize (less accurate)
                    logits_np = upsample_output.data.cpu().numpy()
                    logits_resized = cv2.resize(logits_np, (w, h), interpolation=cv2.INTER_LINEAR)
                    parsing_result = np.argmax(logits_resized, axis=2).astype(np.uint8)
            
            # Convert to PIL Image
            parsed_image = Image.fromarray(parsing_result, mode='L')
            
            # Metadata
            metadata = {
                'model': self.model_name,
                'num_classes': self.num_classes,
                'input_size': self.input_size,
                'original_size': (w, h),
                'labels': self.labels,
                'unique_labels': np.unique(parsing_result).tolist()
            }
            
            return parsed_image, metadata
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to parse image: {e}")
    
    def parse_image_path(self, image_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Parse image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (parsed_image, metadata)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return self.parse_image(image)
    
    def get_label_name(self, label_id: int) -> str:
        """Get label name by ID."""
        return self.model_loader.get_label_name(label_id)


# Helper function for affine transform (from SCHP utils)
def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """Get affine transformation matrix."""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir
    
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    
    return trans


def get_dir(src_point, rot_rad):
    """Get direction vector."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_3rd_point(a, b):
    """Get third point for affine transform."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

