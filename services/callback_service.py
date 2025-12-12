"""
Callback Service - Uploads masks to Mask Asset Service with retry logic.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import aiohttp
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class CallbackService:
    """Service for calling back to Mask Asset Service."""
    
    def __init__(self):
        """Initialize callback service."""
        self.mask_asset_url = os.environ.get(
            'MASK_ASSET_URL',
            'https://apiprod.xapien.in:9002/v1/mask/upload'
        )
        self.secret = os.environ.get(
            'MASK_COMPONENT_SECRET',
            'supersecret-internal-token'
        )
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            aiohttp.ClientResponseError,
            Exception
        )),
        reraise=True
    )
    async def upload_masks(
        self,
        user_id: str,
        job_id: str,
        upper_body_mask_path: Path,
        lower_body_mask_path: Path,
        seg_map_path: Optional[Path] = None
    ) -> dict:
        """
        Upload masks to Mask Asset Service with retry logic.
        
        Args:
            user_id: User ID
            job_id: Job ID
            upper_body_mask_path: Path to upper body mask
            lower_body_mask_path: Path to lower body mask
            seg_map_path: Optional path to segmentation map
        
        Returns:
            Response from Mask Asset Service
        """
        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('user_id', user_id)
            data.add_field('job_id', job_id)
            data.add_field('provider', 'qwen')
            
            # Asset service expects all files under field name "files" (plural) as a list
            # Read all mask files
            async with aiofiles.open(upper_body_mask_path, 'rb') as f:
                upper_content = await f.read()
            
            async with aiofiles.open(lower_body_mask_path, 'rb') as f:
                lower_content = await f.read()
            
            # Add all files with field name "files" (plural)
            # Each file is added with the same field name to create a list
            data.add_field(
                'files',
                upper_content,
                filename=upper_body_mask_path.name,
                content_type='image/png'
            )
            
            data.add_field(
                'files',
                lower_content,
                filename=lower_body_mask_path.name,
                content_type='image/png'
            )
            
            # Add optional seg_map (also as "files")
            if seg_map_path and seg_map_path.exists():
                async with aiofiles.open(seg_map_path, 'rb') as f:
                    seg_content = await f.read()
                    data.add_field(
                        'files',
                        seg_content,
                        filename=seg_map_path.name,
                        content_type='image/png'
                    )
            
            # Make request with retry
            headers = {
                'X-Internal-Auth': self.secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.mask_asset_url,
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    ssl=False  # Disable SSL verification if needed
                ) as response:
                    # Log response status for debugging
                    logger.debug(f"Callback response status: {response.status}")
                    
                    # Handle error responses with detailed logging
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(
                            f"Asset service returned {response.status} for job {job_id}. "
                            f"Response: {error_text[:500]}"
                        )
                        logger.error(
                            f"Request URL: {self.mask_asset_url}, "
                            f"Headers: {dict(headers)}, "
                            f"Form fields: user_id={user_id}, job_id={job_id}, provider=qwen"
                        )
                    
                    response.raise_for_status()
                    result = await response.json()
                    
                    logger.info(
                        f"Successfully uploaded masks for job {job_id}: {result}"
                    )
                    return result
                    
        except Exception as e:
            logger.error(
                f"Failed to upload masks for job {job_id}: {e}",
                exc_info=True
            )
            raise

