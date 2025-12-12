"""
Qwen Segmentation Service - Wraps SCHP parser for mask generation.
"""

import logging
from pathlib import Path
from typing import Optional

from src.api.parser_api import SCHPParserAPI

logger = logging.getLogger(__name__)


class QwenSegmentationService:
    """Service for running Qwen-based segmentation."""
    
    def __init__(self, config_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize segmentation service.
        
        Args:
            config_path: Path to config YAML file
            device: Device selection ('cpu', 'cuda', 'auto')
        """
        self.config_path = config_path
        self.device = device
        self.parser_api: Optional[SCHPParserAPI] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the parser."""
        try:
            logger.info(f"Initializing SCHP Parser (device={self.device})...")
            self.parser_api = SCHPParserAPI(
                config_path=self.config_path,
                device=self.device
            )
            logger.info("SCHP Parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize parser: {e}", exc_info=True)
            raise
    
    def parse_image(self, image_path: str):
        """
        Parse an image and return parsing result.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (parsed_image, metadata)
        """
        if self.parser_api is None:
            raise RuntimeError("Parser not initialized")
        
        return self.parser_api.parse_only(image_path)

