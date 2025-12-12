"""
Device utility functions for CPU/GPU management.
"""

import torch
import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal['auto', 'cuda', 'cpu']


def get_device(device_config: DeviceType = 'auto', gpu_id: int = 0) -> torch.device:
    """
    Get the appropriate torch device based on configuration.
    
    Args:
        device_config: 'auto' (auto-detect), 'cuda' (force GPU), 'cpu' (force CPU)
        gpu_id: GPU device ID (only used if device_config='cuda')
    
    Returns:
        torch.device object
    """
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Auto-detected GPU, using device: {device}")
        else:
            device = torch.device('cpu')
            logger.info("Auto-detected CPU (GPU not available)")
    
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU device: {device}")
    
    elif device_config == 'cpu':
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    else:
        raise ValueError(f"Invalid device config: {device_config}. Use 'auto', 'cuda', or 'cpu'")
    
    return device


def print_device_info(device: torch.device):
    """Print information about the device being used."""
    if device.type == 'cuda':
        print(f"Device: {device}")
        print(f"GPU Name: {torch.cuda.get_device_name(device.index)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024**3:.2f} GB")
    else:
        print(f"Device: {device} (CPU)")
        print(f"CPU Cores: {torch.get_num_threads()}")


def check_cuda_availability() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()

