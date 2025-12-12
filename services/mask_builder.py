"""
Mask Builder - Creates upper_body_mask.png and lower_body_mask.png from parsing results.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_mask_from_parse(parse_array: np.ndarray, mask_type: str) -> np.ndarray:
    """
    Create binary mask from parsing result.
    
    Args:
        parse_array: Parsing result array (H, W) with label values
        mask_type: 'upper_body' or 'lower_body'
    
    Returns:
        Binary mask array (H, W) with 255 for target regions, 0 otherwise
    """
    height, width = parse_array.shape
    
    # Define target labels (ATR dataset)
    # ATR labels: 0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 
    #             5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
    #             11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
    #             16=Bag, 17=Scarf
    
    if mask_type == "upper_body":
        # Upper body: Upper-clothes (4) and Dress (7) - dress covers upper body
        target_labels = [4, 7]
    elif mask_type == "lower_body":
        # Lower body: Skirt (5), Pants (6), and Dress (7) - dress covers lower body
        target_labels = [5, 6, 7]
    else:
        raise ValueError(f"Invalid mask_type: {mask_type}. Must be 'upper_body' or 'lower_body'")
    
    # Create binary mask: 255 for target labels, 0 for everything else
    mask = np.zeros((height, width), dtype=np.uint8)
    for label in target_labels:
        mask[parse_array == label] = 255
    
    return mask


def generate_masks(
    parse_image: Image.Image,
    output_dir: Path,
    job_id: str
) -> Tuple[Path, Path]:
    """
    Generate upper_body_mask.png and lower_body_mask.png from parsing result.
    
    Args:
        parse_image: Parsed image with label values (PIL Image, mode='L')
        output_dir: Directory to save masks
        job_id: Job ID for filename
    
    Returns:
        Tuple of (upper_body_mask_path, lower_body_mask_path)
    """
    # Convert to numpy array
    parse_array = np.array(parse_image)
    
    # Create masks
    upper_mask = create_mask_from_parse(parse_array, "upper_body")
    lower_mask = create_mask_from_parse(parse_array, "lower_body")
    
    # Convert to PIL Images
    upper_mask_img = Image.fromarray(upper_mask, mode='L')
    lower_mask_img = Image.fromarray(lower_mask, mode='L')
    
    # Save masks
    output_dir.mkdir(parents=True, exist_ok=True)
    
    upper_path = output_dir / f"{job_id}_upper_body_mask.png"
    lower_path = output_dir / f"{job_id}_lower_body_mask.png"
    
    upper_mask_img.save(upper_path)
    lower_mask_img.save(lower_path)
    
    logger.info(f"Generated masks: {upper_path.name}, {lower_path.name}")
    
    return upper_path, lower_path

