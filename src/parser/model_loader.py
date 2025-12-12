"""
Model loader for SCHP models with caching and GPU/CPU support.
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add src to path for device utils
import sys
src_root = Path(__file__).absolute().parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
try:
    from utils.device_utils import get_device, print_device_info
except ImportError:
    # Fallback for direct imports
    from src.utils.device_utils import get_device, print_device_info

# Add SCHP repository to path
SCHP_ROOT = Path(__file__).absolute().parent.parent.parent / "Self-Correction-Human-Parsing"
if SCHP_ROOT.exists():
    import sys
    if str(SCHP_ROOT) not in sys.path:
        sys.path.insert(0, str(SCHP_ROOT))
    try:
        import networks
        # Try importing transform_logits, but don't warn if it fails
        # It will be imported dynamically in schp_parser.py when needed
        try:
            from utils.transforms import transform_logits
        except ImportError:
            # Will be imported dynamically later, no warning needed
            transform_logits = None
        try:
            from datasets.simple_extractor_dataset import SimpleFolderDataset
        except ImportError:
            SimpleFolderDataset = None
    except ImportError:
        # SCHP modules not available, will be imported when needed
        networks = None
        transform_logits = None
        SimpleFolderDataset = None
else:
    networks = None
    transform_logits = None
    SimpleFolderDataset = None

logger = logging.getLogger(__name__)

# Dataset settings from SCHP
DATASET_SETTINGS = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


class ModelLoader:
    """Loads and manages SCHP models with caching."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.model_name = self.model_config.get('name', 'atr')
        self.checkpoint_path = self.model_config.get('checkpoint_path', '')
        
        # Device selection: support new 'device' config or legacy 'use_gpu'
        device_config = self.model_config.get('device', 'auto')
        use_gpu_legacy = self.model_config.get('use_gpu', True)
        self.gpu_id = self.model_config.get('gpu_id', 0)
        
        # Get dataset settings
        if self.model_name not in DATASET_SETTINGS:
            raise ValueError(f"Unknown model: {self.model_name}. Choose from: {list(DATASET_SETTINGS.keys())}")
        
        self.dataset_settings = DATASET_SETTINGS[self.model_name]
        self.num_classes = self.dataset_settings['num_classes']
        self.input_size = self.dataset_settings['input_size']
        self.labels = self.dataset_settings['label']
        
        # Model cache
        self._model = None
        self._device = None
        
        # Determine device using utility function
        # Convert legacy use_gpu to device config if needed
        if device_config == 'auto' and not use_gpu_legacy:
            device_config = 'cpu'
        
        self._device = get_device(device_config, self.gpu_id)
        self.device_type = 'cuda' if self._device.type == 'cuda' else 'cpu'
        self.use_gpu = (self.device_type == 'cuda')
        
        # Set CUDA_VISIBLE_DEVICES if using GPU
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        logger.info(f"Model loader initialized: {self.model_name}, device: {self._device}")
        if logger.isEnabledFor(logging.DEBUG):
            print_device_info(self._device)
    
    def load_model(self) -> torch.nn.Module:
        """
        Load the SCHP model.
        
        Returns:
            Loaded model
        """
        if self._model is not None:
            return self._model
        
        # Re-import networks if needed (in case import failed at module level)
        if networks is None:
            if not SCHP_ROOT.exists():
                raise RuntimeError(f"SCHP repository not found at {SCHP_ROOT}. Please ensure Self-Correction-Human-Parsing is cloned.")
            try:
                import sys
                if str(SCHP_ROOT) not in sys.path:
                    sys.path.insert(0, str(SCHP_ROOT))
                import networks as schp_networks
                # Try importing transform_logits (may not be needed for model loading)
                # Note: transform_logits is imported dynamically in schp_parser.py when needed
                # This is just for reference, don't warn if it fails here
                try:
                    from utils.transforms import transform_logits
                    transform_logits_available = True
                except ImportError:
                    transform_logits = None
                    transform_logits_available = False
                    # Don't warn - it will be imported dynamically in schp_parser.py
                    logger.debug("transform_logits not available at module load, will be imported dynamically")
                # Update global reference
                import sys as sys_mod
                sys_mod.modules[__name__].networks = schp_networks
                if transform_logits_available:
                    sys_mod.modules[__name__].transform_logits = transform_logits
            except ImportError as e:
                raise RuntimeError(f"Failed to import SCHP modules: {e}. Please ensure Self-Correction-Human-Parsing is properly cloned and dependencies are installed.")
        
        # Use networks (either from module level or re-imported)
        schp_networks = networks
        
        # Check checkpoint path
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.is_absolute():
            # Relative to project root
            checkpoint_path = Path(__file__).parent.parent.parent / checkpoint_path
        
        # Check if model needs to be downloaded from S3
        if not checkpoint_path.exists():
            # Try downloading from S3 if configured
            s3_config = self.config.get('model', {}).get('s3', {})
            if s3_config.get('enabled', False):
                checkpoint_path = self._download_model_from_s3(checkpoint_path, s3_config)
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found: {checkpoint_path}\n"
                    f"Please download the model from: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing\n"
                    f"Or configure S3 model storage in config.yaml"
                )
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Initialize model
        model = schp_networks.init_model('resnet101', num_classes=self.num_classes, pretrained=None)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Remove 'module.' prefix if present
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            logger.info("Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model checkpoint: {e}")
        
        # Move to device
        model = model.to(self._device)
        model.eval()
        
        self._model = model
        logger.info(f"Model loaded on device: {self._device}")
        
        return model
    
    @property
    def model(self) -> torch.nn.Module:
        """Get the loaded model (lazy loading)."""
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def device(self) -> torch.device:
        """Get the device."""
        return self._device
    
    def get_label_name(self, label_id: int) -> str:
        """Get label name by ID."""
        if 0 <= label_id < len(self.labels):
            return self.labels[label_id]
        return f"Unknown_{label_id}"
    
    def _download_model_from_s3(self, local_path: Path, s3_config: Dict[str, Any]) -> Path:
        """
        Download model from S3/MinIO if not present locally.
        Supports both AWS S3 and MinIO.
        
        Args:
            local_path: Local path where model should be stored
            s3_config: S3 configuration dictionary
        
        Returns:
            Path to downloaded model file
        """
        try:
            from minio import Minio
            from minio.error import S3Error
        except ImportError:
            raise RuntimeError("minio library is required for S3 model download. Install with: pip install minio")
        
        # Ensure local directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If model already exists locally, return it
        if local_path.exists():
            logger.info(f"Model already exists locally: {local_path}")
            return local_path
        
        # S3 configuration
        endpoint = s3_config.get('endpoint', '').strip()
        access_key = s3_config.get('access_key', '').strip()
        secret_key = s3_config.get('secret_key', '').strip()
        bucket = s3_config.get('bucket', '').strip()
        prefix = s3_config.get('prefix', 'models/').strip()
        secure = s3_config.get('secure', True)  # Default to secure for AWS S3
        
        if not all([endpoint, access_key, secret_key, bucket]):
            raise ValueError("S3 model storage enabled but missing required configuration (endpoint, access_key, secret_key, bucket)")
        
        # Handle AWS S3 endpoint format
        # If endpoint contains 'amazonaws.com', it's AWS S3
        is_aws_s3 = 'amazonaws.com' in endpoint.lower()
        region = None
        
        if is_aws_s3:
            # For AWS S3, normalize endpoint
            # Endpoint format should be: s3.amazonaws.com or s3.region.amazonaws.com
            endpoint_lower = endpoint.lower()
            if 's3.' in endpoint_lower and 'amazonaws.com' in endpoint_lower:
                # Extract the s3.region.amazonaws.com part
                # Handle formats like: s3.amazonaws.com, s3.us-east-1.amazonaws.com
                if endpoint_lower.startswith('s3.'):
                    # Already in correct format
                    pass
                else:
                    # Extract from URL like: xapienappassets.s3.amazonaws.com
                    parts = endpoint.split('.')
                    s3_index = next((i for i, part in enumerate(parts) if part.lower() == 's3'), None)
                    if s3_index is not None:
                        endpoint = '.'.join(parts[s3_index:])
                    else:
                        endpoint = 's3.amazonaws.com'
            else:
                endpoint = 's3.amazonaws.com'
            secure = True  # AWS S3 always uses HTTPS
            # Get region from config or extract from endpoint
            region = s3_config.get('region', '').strip()
            if not region and 's3.' in endpoint.lower():
                # Try to extract region from endpoint: s3.us-east-1.amazonaws.com
                parts = endpoint.split('.')
                if len(parts) >= 2 and parts[1] != 'amazonaws':
                    region = parts[1]
        
        # Construct S3 object name
        model_filename = local_path.name
        s3_object_name = f"{prefix.rstrip('/')}/{model_filename}" if prefix else model_filename
        
        logger.info(f"Downloading model from S3: {bucket}/{s3_object_name} (endpoint: {endpoint})")
        
        try:
            # Initialize MinIO client (works with both MinIO and AWS S3)
            client_kwargs = {
                'endpoint': endpoint,
                'access_key': access_key,
                'secret_key': secret_key,
                'secure': secure
            }
            
            # Add region for AWS S3 (if available)
            if is_aws_s3 and region:
                client_kwargs['region'] = region
            
            client = Minio(**client_kwargs)
            
            # Download model
            client.fget_object(
                bucket_name=bucket,
                object_name=s3_object_name,
                file_path=str(local_path)
            )
            
            logger.info(f"Successfully downloaded model from S3 to {local_path}")
            return local_path
            
        except S3Error as e:
            raise FileNotFoundError(
                f"Failed to download model from S3: {e}\n"
                f"S3 path: {bucket}/{s3_object_name}\n"
                f"Endpoint: {endpoint}\n"
                f"Please ensure the model file exists in S3 or download it manually."
            )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and override with environment variables.
    
    Args:
        config_path: Path to config file. If None, uses default.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override S3 model configuration with environment variables if present
    if 'model' not in config:
        config['model'] = {}
    if 's3' not in config['model']:
        config['model']['s3'] = {}
    
    s3_config = config['model']['s3']
    
    # Override with environment variables (if set)
    if os.environ.get('MODEL_S3_ENABLED'):
        s3_config['enabled'] = os.environ.get('MODEL_S3_ENABLED', '').lower() in ('true', '1', 'yes')
    if os.environ.get('MODEL_S3_ENDPOINT'):
        s3_config['endpoint'] = os.environ.get('MODEL_S3_ENDPOINT')
    if os.environ.get('MODEL_S3_ACCESS_KEY'):
        s3_config['access_key'] = os.environ.get('MODEL_S3_ACCESS_KEY')
    if os.environ.get('MODEL_S3_SECRET_KEY'):
        s3_config['secret_key'] = os.environ.get('MODEL_S3_SECRET_KEY')
    if os.environ.get('MODEL_S3_BUCKET'):
        s3_config['bucket'] = os.environ.get('MODEL_S3_BUCKET')
    if os.environ.get('MODEL_S3_PREFIX'):
        s3_config['prefix'] = os.environ.get('MODEL_S3_PREFIX')
    if os.environ.get('MODEL_S3_REGION'):
        s3_config['region'] = os.environ.get('MODEL_S3_REGION')
    if os.environ.get('MODEL_S3_SECURE'):
        s3_config['secure'] = os.environ.get('MODEL_S3_SECURE', '').lower() in ('true', '1', 'yes')
    
    return config

