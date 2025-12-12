"""
Clothing region cropping with tight bounding box and white background.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def get_clothing_bounding_box(parse_array: np.ndarray, mask_type: str, 
                              debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Get tight bounding box for clothing region.
    
    Args:
        parse_array: Parsing result array (H, W) with label values
        mask_type: 'upper_body' or 'lower_body'
        debug: Enable debug output
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None
    """
    height, width = parse_array.shape
    
    # Define target and opposite labels (ATR dataset)
    # ATR labels: 0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 
    #             5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
    #             11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
    #             16=Bag, 17=Scarf
    
    if mask_type == "upper_body":
        # Include both upper-clothes (4) and dress (7) for women
        # Dress covers upper body, so it should be included for upper_body
        target_labels = [4, 7]  # Upper-clothes and dress
        opposite_labels = [5, 6]  # skirt, pants
        # Exclude all non-clothing parts: face, neck, arms, legs, background, etc.
        # Note: dress (7) is NOT excluded - it's a target label
        exclude_labels = {0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    elif mask_type == "lower_body":
        # Include skirt, pants, and dress (dress covers lower body too)
        target_labels = [5, 6, 7]  # Skirt, pants, and dress
        opposite_labels = [4]  # upper_clothes (but not dress)
        # Exclude all non-clothing parts
        # Note: dress (7) is NOT excluded - it's a target label
        exclude_labels = {0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    elif mask_type == "other":
        # Crop all clothing labels: upper-clothes, skirt, pants, and dress
        target_labels = [4, 5, 6, 7]  # All clothing labels
        opposite_labels = []  # No opposite labels - include all clothing
        # Exclude all non-clothing parts only
        exclude_labels = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    else:
        target_labels = [4, 5, 6, 7]  # all clothing
        opposite_labels = []
        exclude_labels = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    
    # Create target mask - ONLY target labels
    target_mask = np.zeros((height, width), dtype=np.uint8)
    for label in target_labels:
        target_mask[parse_array == label] = 1
    
    # STRICTLY exclude opposite labels
    if opposite_labels:
        opposite_mask = np.zeros((height, width), dtype=bool)
        for label in opposite_labels:
            opposite_mask |= (parse_array == label)
        target_mask = np.where(opposite_mask, 0, target_mask).astype(np.uint8)
    
    # STRICTLY exclude all non-target labels (face, arms, legs, etc.)
    exclude_mask = np.zeros((height, width), dtype=bool)
    for label in exclude_labels:
        exclude_mask |= (parse_array == label)
    target_mask = np.where(exclude_mask, 0, target_mask).astype(np.uint8)
    
    # Check if any target pixels
    target_pixel_count = target_mask.sum()
    if target_pixel_count == 0:
        if debug:
            logger.debug("No target pixels found")
        return None
    
    # Find tight bounding box - ONLY from pure target pixels
    rows, cols = np.where(target_mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return None
    
    # Get tight bounding box (no padding)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    
    # Ensure valid
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # CRITICAL: Shrink bounding box to exclude regions with too many non-target labels
    # This prevents including face/neck/arms that are adjacent to clothing
    bbox_region = parse_array[y_min:y_max, x_min:x_max]
    
    # Check each row - if a row has < 50% target pixels, exclude it (STRICT)
    target_pixels_per_row = np.zeros(y_max - y_min, dtype=np.int32)
    for label in target_labels:
        target_pixels_per_row += (bbox_region == label).sum(axis=1)
    
    row_width = x_max - x_min
    valid_rows = target_pixels_per_row >= (row_width * 0.50)  # At least 50% target pixels (STRICT)
    
    if valid_rows.sum() > 0:
        # Find first and last valid rows
        valid_row_indices = np.where(valid_rows)[0]
        y_min_new = y_min + valid_row_indices[0]
        y_max_new = y_min + valid_row_indices[-1] + 1
        y_min, y_max = y_min_new, y_max_new
    
    # Check each column - if a column has < 50% target pixels, exclude it (STRICT)
    target_pixels_per_col = np.zeros(x_max - x_min, dtype=np.int32)
    for label in target_labels:
        target_pixels_per_col += (bbox_region == label).sum(axis=0)
    
    col_height = y_max - y_min
    valid_cols = target_pixels_per_col >= (col_height * 0.50)  # At least 50% target pixels (STRICT)
    
    if valid_cols.sum() > 0:
        # Find first and last valid columns
        valid_col_indices = np.where(valid_cols)[0]
        x_min_new = x_min + valid_col_indices[0]
        x_max_new = x_min + valid_col_indices[-1] + 1
        x_min, x_max = x_min_new, x_max_new
    
    # Re-validate after shrinking
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # Final validation: check that bbox region contains mostly target labels
    bbox_region_final = parse_array[y_min:y_max, x_min:x_max]
    target_pixels_in_bbox = 0
    for label in target_labels:
        target_pixels_in_bbox += (bbox_region_final == label).sum()
    total_bbox_pixels = bbox_region_final.size
    target_ratio = target_pixels_in_bbox / total_bbox_pixels if total_bbox_pixels > 0 else 0
    
    if debug:
        logger.debug(f"Bbox target ratio: {target_ratio:.2%} ({target_pixels_in_bbox}/{total_bbox_pixels})")
        logger.debug(f"Bounding box: ({x_min}, {y_min}, {x_max}, {y_max}), size: {x_max-x_min}x{y_max-y_min}")
    
    # If bbox contains less than 50% target pixels after shrinking, it's still problematic
    if target_ratio < 0.5:
        if debug:
            logger.warning(f"Bbox contains only {target_ratio:.2%} target pixels, may be inaccurate")
    
    return (x_min, y_min, x_max, y_max)


def crop_clothing_region(image: Image.Image, parse_image: Image.Image, mask_type: str,
                        output_size: Tuple[int, int] = (1024, 1024),
                        background_color: str = 'white', debug: bool = False,
                        save_parse_debug_path: Optional[str] = None) -> Image.Image:
    """
    Extract clothing region (only parsed label pixels) and paste on white background.
    No bounding box needed - we extract clothing pixels directly from the mask.
    
    Args:
        image: Original image (PIL Image)
        parse_image: Parsed image with labels (PIL Image)
        mask_type: 'upper_body' or 'lower_body'
        output_size: Output image size (width, height)
        background_color: Background color ('white', 'black', 'transparent')
        debug: Enable debug output
    
    Returns:
        Image with only clothing pixels on white background
    """
    # Convert to arrays
    img_array = np.array(image.convert('RGB'))
    parse_array = np.array(parse_image)
    
    # Create mask: 1 for target labels, 0 for everything else
    # We work directly on the full image - no bounding box needed!
    mask = np.zeros((parse_array.shape[0], parse_array.shape[1]), dtype=np.uint8)
    
    # Define target and opposite labels
    # Body labels that must be STRICTLY excluded: Face, Arms, Legs, etc.
    # ATR labels: 0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 
    #             5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
    #             11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
    #             16=Bag, 17=Scarf
    body_labels = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}  # All non-clothing labels
    
    if mask_type == "upper_body":
        # Include both upper-clothes (4) and dress (7) for women
        # Dress covers upper body, so it should be included for upper_body cropping
        target_labels = [4, 7]  # Upper-clothes and dress
        opposite_labels = [5, 6]  # Exclude lower body (skirt, pants)
    elif mask_type == "lower_body":
        # Include skirt, pants, and dress (dress covers lower body too)
        target_labels = [5, 6, 7]  # Skirt, pants, and dress
        opposite_labels = [4]  # Exclude upper body (upper-clothes only, not dress)
    elif mask_type == "other":
        # Crop all clothing labels: upper-clothes, skirt, pants, and dress
        target_labels = [4, 5, 6, 7]  # All clothing labels
        opposite_labels = []  # No opposite labels - include all clothing
    else:
        target_labels = [4, 5, 6, 7]  # All clothing (fallback)
        opposite_labels = []
    
    # Set mask to 1 only for target labels (the parsed label pixels)
    for label in target_labels:
        mask[parse_array == label] = 1
    
    # STRICTLY set mask to 0 for opposite labels (this ensures they are excluded)
    if opposite_labels:
        for label in opposite_labels:
            mask[parse_array == label] = 0
    
    # STRICTLY set mask to 0 for ALL body labels (face, arms, legs, background, etc.)
    # This ensures body parts are NEVER included in the crop
    for label in body_labels:
        mask[parse_array == label] = 0
    
    # Check if any clothing detected
    if mask.sum() == 0:
        logger.warning(f"No clothing detected for {mask_type}, returning original image")
        # Return original image on background with SAME scaling rules
        white_bg_size = output_size[0]  # Should be 1024
        img_width, img_height = img_array.shape[1], img_array.shape[0]
        
        # Apply same scaling rules as clothing extraction
        scale_x = white_bg_size / img_width if img_width > white_bg_size else 1.0
        scale_y = white_bg_size / img_height if img_height > white_bg_size else 1.0
        scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed
        
        # Scale image if needed
        scaled_image = image
        if scale_factor < 1.0:
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create background and center scaled image
        bg = Image.new('RGB', (white_bg_size, white_bg_size), color=background_color)
        paste_x = (white_bg_size - scaled_image.width) // 2
        paste_y = (white_bg_size - scaled_image.height) // 2
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)
        bg.paste(scaled_image, (paste_x, paste_y))
        return bg
    
    # Find bounding box of clothing pixels to determine if scaling is needed
    # This ensures the clothing fits within 1024x1024 without clipping
    clothing_mask_binary = (mask == 1)
    if clothing_mask_binary.sum() > 0:
        # Find rows and columns with clothing pixels
        rows_with_clothing = np.any(clothing_mask_binary, axis=1)
        cols_with_clothing = np.any(clothing_mask_binary, axis=0)
        
        if rows_with_clothing.sum() > 0 and cols_with_clothing.sum() > 0:
            # Get bounding box of clothing pixels
            clothing_y_min = int(np.where(rows_with_clothing)[0][0])
            clothing_y_max = int(np.where(rows_with_clothing)[0][-1]) + 1
            clothing_x_min = int(np.where(cols_with_clothing)[0][0])
            clothing_x_max = int(np.where(cols_with_clothing)[0][-1]) + 1
            
            clothing_width = clothing_x_max - clothing_x_min
            clothing_height = clothing_y_max - clothing_y_min
        else:
            # Fallback: use full image dimensions
            clothing_width, clothing_height = img_array.shape[1], img_array.shape[0]
            clothing_x_min, clothing_y_min = 0, 0
    else:
        # No clothing detected (shouldn't happen, but handle it)
        clothing_width, clothing_height = img_array.shape[1], img_array.shape[0]
        clothing_x_min, clothing_y_min = 0, 0
    
    # Create RGBA image with alpha channel - only clothing pixels are visible
    # Alpha = 255 for clothing pixels, Alpha = 0 for everything else
    clothing_rgba = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
    clothing_rgba[:, :, :3] = img_array  # RGB channels from original image
    clothing_rgba[:, :, 3] = (mask * 255).astype(np.uint8)  # Alpha channel: 255 for clothing, 0 for everything else
    
    # Convert to PIL Image with alpha channel
    clothing_with_alpha = Image.fromarray(clothing_rgba, mode='RGBA')
    
    # Calculate scale factor to ensure the FULL image fits within output_size
    # This prevents any part of the image (including clothing) from being clipped
    white_bg_size = output_size[0]  # Should be 1024
    img_width, img_height = img_array.shape[1], img_array.shape[0]
    
    # Calculate scale factor to fit full image within 1024x1024
    scale_x = white_bg_size / img_width if img_width > white_bg_size else 1.0
    scale_y = white_bg_size / img_height if img_height > white_bg_size else 1.0
    scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed
    
    # Scale the image if needed to fit within 1024x1024
    if scale_factor < 1.0:
        new_width = int(clothing_with_alpha.width * scale_factor)
        new_height = int(clothing_with_alpha.height * scale_factor)
        clothing_with_alpha = clothing_with_alpha.resize((new_width, new_height), Image.LANCZOS)
        if debug:
            logger.debug(f"Scaled image from {clothing_rgba.shape[1]}x{clothing_rgba.shape[0]} to {new_width}x{new_height} (scale: {scale_factor:.3f})")
    
    if debug:
        clothing_pixels = mask.sum()
        total_pixels = mask.size
        logger.debug(f"Clothing pixels: {clothing_pixels} ({clothing_pixels/total_pixels*100:.1f}%)")
        logger.debug(f"Clothing bbox: ({clothing_x_min}, {clothing_y_min}) to ({clothing_x_max}, {clothing_y_max})")
        logger.debug(f"Clothing size: {clothing_width}x{clothing_height}")
        logger.debug(f"Scale factor: {scale_factor:.3f}")
    
    # Paste clothing onto 1024x1024 white background
    # The alpha channel ensures only clothing pixels are visible
    white_background = Image.new('RGB', (white_bg_size, white_bg_size), color='white')
    
    # Calculate position to center the clothing on white background
    final_width, final_height = clothing_with_alpha.size
    paste_x = (white_bg_size - final_width) // 2
    paste_y = (white_bg_size - final_height) // 2
    
    # Ensure paste coordinates are non-negative
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)
    
    # Verify clothing fits within bounds
    if paste_x + final_width > white_bg_size or paste_y + final_height > white_bg_size:
        # This shouldn't happen if scaling worked correctly, but handle it
        logger.warning(f"Clothing may be clipped: paste at ({paste_x}, {paste_y}), size {final_width}x{final_height}, canvas {white_bg_size}x{white_bg_size}")
    
    # Paste clothing onto white background (alpha channel handles transparency)
    white_background.paste(clothing_with_alpha, (paste_x, paste_y), clothing_with_alpha)
    bg = white_background
    
    if debug:
        logger.debug(f"Clothing size: {clothing_width}x{clothing_height}, pasted at ({paste_x}, {paste_y})")
        logger.debug(f"Final output size: {bg.size[0]}x{bg.size[1]}")
    
    return bg

