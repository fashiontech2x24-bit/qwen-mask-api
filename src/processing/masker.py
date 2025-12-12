"""
Green bounding box masking for clothing regions.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def get_simple_bounding_box(parse_array: np.ndarray, mask_type: str, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box that includes ALL target label pixels.
    No strict exclusion - just find the bounding box of all target labels.
    
    Args:
        parse_array: Parsing result array (H, W) with label values
        mask_type: 'upper_body', 'lower_body', or 'other'
        debug: Enable debug output
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None
    """
    height, width = parse_array.shape
    
    # Define target labels (no strict exclusion - just target labels)
    if mask_type == "upper_body":
        target_labels = [4, 7]  # Upper-clothes and dress
    elif mask_type == "lower_body":
        target_labels = [5, 6, 7]  # Skirt, pants, and dress
    elif mask_type == "other":
        target_labels = [4, 5, 6, 7]  # All clothing labels
    else:
        target_labels = [4, 5, 6, 7]  # All clothing (fallback)
    
    # Create mask: 1 for target labels, 0 for everything else
    target_mask = np.zeros((height, width), dtype=np.uint8)
    for label in target_labels:
        target_mask[parse_array == label] = 1
    
    # Check if any target pixels
    target_pixel_count = target_mask.sum()
    if target_pixel_count == 0:
        if debug:
            logger.debug("No target pixels found")
        return None
    
    # Find bounding box of ALL target pixels (no strict exclusion)
    rows, cols = np.where(target_mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    # Get bounding box that includes all target pixels
    y_min, y_max = int(rows.min()), int(rows.max()) + 1
    x_min, x_max = int(cols.min()), int(cols.max()) + 1
    
    # Ensure valid
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Ensure within bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    if debug:
        logger.debug(f"Simple bbox (all target labels): ({x_min}, {y_min}) to ({x_max}, {y_max})")
        logger.debug(f"Bbox size: {x_max-x_min}x{y_max-y_min}")
        logger.debug(f"Target pixels: {target_pixel_count}")
    
    return (x_min, y_min, x_max, y_max)


def create_green_bounding_box_mask(image: Image.Image, parse_image: Image.Image, 
                                   mask_type: str, debug: bool = False) -> Image.Image:
    """
    Create green bounding box mask on original image.
    Bounding box includes ALL target label pixels (no strict exclusion).
    
    Args:
        image: Original image (PIL Image)
        parse_image: Parsed image with labels (PIL Image)
        mask_type: 'upper_body', 'lower_body', or 'other' (returns original if 'other')
        debug: Enable debug output
    
    Returns:
        Image with green bounding box mask (or original image if mask_type='other')
    """
    # If mask_type is 'other', return original image unchanged (make a copy to avoid issues)
    if mask_type == "other":
        if debug:
            logger.debug("mask_type='other' selected, returning original image unchanged")
        # Return a copy to ensure no modifications
        return image.copy()
    
    # Convert to arrays
    img_array = np.array(image.convert('RGB'))
    parse_array = np.array(parse_image)
    
    # Get simple bounding box (includes all target labels, no strict exclusion)
    bbox = get_simple_bounding_box(parse_array, mask_type, debug=debug)
    if bbox is None:
        logger.warning(f"No clothing detected for {mask_type}, returning original image")
        return image
    
    x_min, y_min, x_max, y_max = bbox
    
    # Create mask with green bounding box FILLED (entire rectangle)
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    
    # Fill the entire bounding box area
    mask[y_min:y_max, x_min:x_max] = 255
    
    # Apply solid green color to the entire bounding box area
    green_color = np.array([0, 255, 0], dtype=np.uint8)  # RGB green
    masked_image = img_array.copy()
    masked_image[y_min:y_max, x_min:x_max] = green_color
    
    result = Image.fromarray(masked_image)
    
    if debug:
        logger.debug(f"Green bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
    
    return result

