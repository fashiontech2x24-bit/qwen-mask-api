"""
Mask Builder - Creates green bounding box masked images for upper and lower body.

Output: Original image with solid green bounding box overlay on clothing regions.
Filename format: {job_id}_upper_body_mask.png, {job_id}_lower_body_mask.png
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.masker import create_green_bounding_box_mask

logger = logging.getLogger(__name__)


def generate_masks(
    parse_image: Image.Image,
    output_dir: Path,
    job_id: str,
    original_image: Optional[Image.Image] = None,
    original_image_path: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Generate green bounding box masked images for upper and lower body.
    
    The output images are the original image with a solid green rectangle
    covering the clothing region's bounding box.
    
    Args:
        parse_image: Parsed image with label values (PIL Image, mode='L')
        output_dir: Directory to save masks
        job_id: Job ID for filename
        original_image: Original image (PIL Image) - required for green box overlay
        original_image_path: Path to original image (alternative to original_image)
    
    Returns:
        Tuple of (upper_body_mask_path, lower_body_mask_path)
    """
    # Load original image if not provided
    if original_image is None:
        if original_image_path is None:
            raise ValueError("Either original_image or original_image_path must be provided")
        original_image = Image.open(original_image_path).convert('RGB')
    else:
        original_image = original_image.convert('RGB')
    
    # Generate green bounding box masked images
    upper_mask_img = create_green_bounding_box_mask(
        image=original_image,
        parse_image=parse_image,
        mask_type="upper_body",
        debug=True
    )
    
    lower_mask_img = create_green_bounding_box_mask(
        image=original_image,
        parse_image=parse_image,
        mask_type="lower_body",
        debug=True
    )
    
    # Save masks
    output_dir.mkdir(parents=True, exist_ok=True)
    
    upper_path = output_dir / f"{job_id}_upper_body_mask.png"
    lower_path = output_dir / f"{job_id}_lower_body_mask.png"
    
    upper_mask_img.save(upper_path)
    lower_mask_img.save(lower_path)
    
    logger.info(f"Generated green box masks: {upper_path.name}, {lower_path.name}")
    
    return upper_path, lower_path


# Legacy function for backward compatibility (creates binary masks)
def create_mask_from_parse(parse_array: np.ndarray, mask_type: str) -> np.ndarray:
    """
    Create binary mask from parsing result (legacy, kept for compatibility).
    
    Args:
        parse_array: Parsing result array (H, W) with label values
        mask_type: 'upper_body' or 'lower_body'
    
    Returns:
        Binary mask array (H, W) with 255 for target regions, 0 otherwise
    """
    height, width = parse_array.shape
    
    # ATR labels
    if mask_type == "upper_body":
        target_labels = [4, 7]  # Upper-clothes and Dress
    elif mask_type == "lower_body":
        target_labels = [5, 6, 7]  # Skirt, Pants, and Dress
    else:
        raise ValueError(f"Invalid mask_type: {mask_type}")
    
    mask = np.zeros((height, width), dtype=np.uint8)
    for label in target_labels:
        mask[parse_array == label] = 255
    
    return mask

