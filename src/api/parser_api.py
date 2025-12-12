"""
Production-grade API for SCHP parsing, cropping, and masking.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from ..parser.schp_parser import SCHPParser
from ..processing.cropper import crop_clothing_region
from ..processing.masker import create_green_bounding_box_mask
from ..parser.model_loader import load_config
import os

logger = logging.getLogger(__name__)


class SCHPParserAPI:
    """Production-grade API for SCHP human parsing."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = 'auto', **kwargs):
        """
        Initialize SCHP Parser API.
        
        Args:
            config_path: Path to config YAML file
            device: Device selection ('auto', 'cuda', 'cpu')
                   Can also be set via SCHP_DEVICE environment variable
            **kwargs: Override config values (model_name, checkpoint_path, device, use_gpu, etc.)
        """
        # Check environment variable for device override
        env_device = os.environ.get('SCHP_DEVICE', '').lower()
        if env_device in ['auto', 'cuda', 'cpu']:
            device = env_device
            logger.info(f"Device set from environment variable: {device}")
        
        # Load config
        if config_path:
            config = load_config(config_path)
        else:
            config = load_config()
        
        # Override with kwargs
        if kwargs:
            if 'model_name' in kwargs:
                config['model']['name'] = kwargs.pop('model_name')
            if 'checkpoint_path' in kwargs:
                config['model']['checkpoint_path'] = kwargs.pop('checkpoint_path')
            if 'device' in kwargs:
                config['model']['device'] = kwargs.pop('device')
            elif 'use_gpu' in kwargs:
                # Legacy support
                use_gpu = kwargs.pop('use_gpu')
                config['model']['device'] = 'cuda' if use_gpu else 'cpu'
                config['model']['use_gpu'] = use_gpu
            if 'gpu_id' in kwargs:
                config['model']['gpu_id'] = kwargs.pop('gpu_id')
        
        # Override device if provided as parameter (highest priority)
        if device and device != 'auto':
            config['model']['device'] = device
        
        self.config = config
        self.parser = SCHPParser(config)
        self.processing_config = config.get('processing', {})
        
        logger.info("SCHP Parser API initialized")
    
    def process_image(self, image_path: str, mask_type: str = 'upper_body',
                     mode: str = 'crop', output_path: Optional[str] = None,
                     debug: bool = False) -> str:
        """
        Process a single image: parse, crop, or mask.
        
        Args:
            image_path: Path to input image
            mask_type: 'upper_body', 'lower_body', or 'other' (all clothing labels)
            mode: 'crop' (crop on white background) or 'green_mask' (green bounding box)
            output_path: Path to save output (optional)
            debug: Enable debug mode
        
        Returns:
            Path to output image
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Parse image
            parse_image, metadata = self.parser.parse_image(image)
            
            if debug:
                logger.info(f"Parsing metadata: {metadata}")
            
            # Determine parse debug output path (for cropped region)
            # Disabled by default - user only wants 3 images per test case
            parse_debug_path = None
            if self.processing_config.get('save_parse_debug', False):  # Changed to False
                try:
                    if output_path:
                        parse_debug_path = str(Path(output_path).parent / f"{Path(output_path).stem}_parse_debug.png")
                    else:
                        parse_debug_path = str(image_path.parent / f"{image_path.stem}_parse_debug.png")
                except Exception as e:
                    logger.warning(f"Failed to determine parse debug path: {e}")
            
            # Process based on mode
            if mode == 'crop':
                # Crop clothing on white background
                crop_config = self.processing_config.get('crop', {})
                output_size = tuple(crop_config.get('output_size', [1024, 1024]))
                bg_color = crop_config.get('background_color', 'white')
                
                result = crop_clothing_region(
                    image=image,
                    parse_image=parse_image,
                    mask_type=mask_type,
                    output_size=output_size,
                    background_color=bg_color,
                    debug=debug,
                    save_parse_debug_path=parse_debug_path
                )
                
            elif mode == 'green_mask':
                # For 'other' mask_type, skip processing and return original
                if mask_type == 'other':
                    result = image
                else:
                    # Green bounding box mask
                    result = create_green_bounding_box_mask(
                        image=image,
                        parse_image=parse_image,
                        mask_type=mask_type,
                        debug=debug
                    )
                
            elif mode == 'parse_only':
                # Just return parsing result
                result = parse_image
                
            else:
                raise ValueError(f"Unknown mode: {mode}. Choose from: 'crop', 'green_mask', 'parse_only'")
            
            # Save output
            if output_path is None:
                suffix = '_cropped' if mode == 'crop' else '_masked' if mode == 'green_mask' else '_parsed'
                output_path = image_path.parent / f"{image_path.stem}{suffix}{image_path.suffix}"
            else:
                output_path = Path(output_path)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For 'other' mask_type in green_mask mode, copy original file to avoid compression
            if mode == 'green_mask' and mask_type == 'other':
                import shutil
                shutil.copy2(str(image_path), str(output_path))
                logger.info(f"Original image copied to: {output_path} (mask_type='other', no processing)")
            else:
                # Save image (may apply compression)
                result.save(str(output_path))
                logger.info(f"Output saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Processing failed for {image_path}: {e}", exc_info=True)
            
            # Don't fallback to original - raise the error so caller knows processing failed
            # This ensures we don't upload unmasked images as "success"
            raise RuntimeError(f"Failed to process image: {e}")
    
    def parse_only(self, image_path: str, output_path: Optional[str] = None) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Parse image only, return parsing result and metadata.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save parsing result
        
        Returns:
            Tuple of (parsed_image, metadata)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        parse_image, metadata = self.parser.parse_image(image)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parse_image.save(str(output_path))
        
        return parse_image, metadata


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration."""
    import logging
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

