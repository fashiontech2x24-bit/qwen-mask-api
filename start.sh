#!/bin/bash
# Start Qwen Mask Component Service
# This script handles full environment setup, dependency installation, and service startup

set -e

echo "=========================================="
echo "Qwen Mask Component - Start Script"
echo "=========================================="

# Determine working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# STEP 1: AWS CREDENTIALS SETUP
# ==========================================
echo ""
echo "[1/7] Checking AWS/S3 Credentials..."

# Set default S3 configuration
export MODEL_S3_ENABLED=${MODEL_S3_ENABLED:-"true"}
export MODEL_S3_ENDPOINT=${MODEL_S3_ENDPOINT:-"s3.amazonaws.com"}
export MODEL_S3_BUCKET=${MODEL_S3_BUCKET:-"xapienappassets"}
export MODEL_S3_PREFIX=${MODEL_S3_PREFIX:-"models/"}
export MODEL_S3_REGION=${MODEL_S3_REGION:-"ap-south-1"}

# Check for S3 credentials (support both naming conventions)
if [ -z "$MODEL_S3_ACCESS_KEY" ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
    export MODEL_S3_ACCESS_KEY="$AWS_ACCESS_KEY_ID"
fi
if [ -z "$MODEL_S3_SECRET_KEY" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    export MODEL_S3_SECRET_KEY="$AWS_SECRET_ACCESS_KEY"
fi

if [ -n "$MODEL_S3_ACCESS_KEY" ] && [ -n "$MODEL_S3_SECRET_KEY" ]; then
    echo "✓ S3 credentials found"
else
    echo "⚠️  S3 credentials not fully configured"
    echo "   Model download may fail if model is not present locally"
    echo "   Set: MODEL_S3_ACCESS_KEY and MODEL_S3_SECRET_KEY"
fi

# ==========================================
# STEP 2: PYTHON ENVIRONMENT SETUP
# ==========================================
echo ""
echo "[2/7] Setting up Python environment..."

VENV_DIR="venv"

# Find compatible Python version (3.7-3.10 required for PyTorch 1.12.0)
find_python() {
    for version in python3.10 python3.9 python3.8 python3.7 python3; do
        if command -v $version &> /dev/null; then
            PYTHON_VER=$($version --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
            MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
            MINOR=$(echo $PYTHON_VER | cut -d. -f2)
            if [ "$MAJOR" = "3" ] && [ "$MINOR" -ge 7 ] && [ "$MINOR" -le 10 ]; then
                echo $version
                return 0
            fi
        fi
    done
    return 1
}

if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual environment not found. Creating..."
    
    PYTHON_CMD=$(find_python)
    if [ -z "$PYTHON_CMD" ]; then
        echo "❌ Error: Python 3.7-3.10 is required for PyTorch 1.12.0"
        echo "   Please install Python 3.10: sudo apt-get install python3.10 python3.10-venv"
        exit 1
    fi
    
    echo "Using $PYTHON_CMD"
    $PYTHON_CMD -m venv ${VENV_DIR}
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate

# ==========================================
# STEP 3: DEPENDENCY INSTALLATION
# ==========================================
echo ""
echo "[3/7] Checking and installing dependencies..."

# Upgrade pip first
pip install --upgrade pip -q

# Check if PyTorch is installed and correct version
PYTORCH_OK=false
if python -c "import torch; exit(0 if '1.12' in torch.__version__ else 1)" 2>/dev/null; then
    echo "✓ PyTorch 1.12.x already installed"
    PYTORCH_OK=true
fi

if [ "$PYTORCH_OK" = false ]; then
    echo "Installing PyTorch 1.12.0+cpu (required for SCHP compatibility)..."
    pip uninstall -y torch torchvision 2>/dev/null || true
    pip install torch==1.12.0+cpu torchvision==0.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        echo "Trying alternative installation method..."
        pip install torch==1.12.0 torchvision==0.13.0 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
    fi
    echo "✓ PyTorch installed"
fi

# Install remaining dependencies (excluding torch/torchvision which are already handled)
echo "Installing other dependencies..."
TEMP_REQ=$(mktemp)
grep -v "^torch" requirements.txt | grep -v "^#" | grep -v "^$" > "$TEMP_REQ" 2>/dev/null || true
if [ -s "$TEMP_REQ" ]; then
    pip install -r "$TEMP_REQ" -q 2>&1 | tail -5
fi
rm -f "$TEMP_REQ"
echo "✓ Dependencies installed"

# Install ninja for C++ extensions if not present
if ! command -v ninja &> /dev/null; then
    pip install ninja -q
fi

# ==========================================
# STEP 4: CLONE SCHP REPOSITORY
# ==========================================
echo ""
echo "[4/7] Checking SCHP repository..."

if [ ! -d "Self-Correction-Human-Parsing" ]; then
    echo "Cloning SCHP repository..."
    git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git
    echo "✓ SCHP repository cloned"
else
    echo "✓ SCHP repository exists"
fi

# ==========================================
# STEP 5: CREATE DIRECTORIES
# ==========================================
echo ""
echo "[5/7] Creating required directories..."

mkdir -p models logs outputs checkpoints /tmp/qwen
echo "✓ Directories ready"

# ==========================================
# STEP 6: DOWNLOAD MODELS
# ==========================================
echo ""
echo "[6/7] Checking model files..."

MODEL_FILE="models/exp-schp-201908301523-atr.pth"

if [ -f "$MODEL_FILE" ]; then
    echo "✓ Model exists: $MODEL_FILE"
    echo "  Size: $(du -h "$MODEL_FILE" | cut -f1)"
elif [ "$MODEL_S3_ENABLED" = "true" ] && [ -n "$MODEL_S3_ACCESS_KEY" ] && [ -n "$MODEL_S3_SECRET_KEY" ]; then
    echo "Downloading model from S3..."
    
    # Export AWS credentials for boto3/minio
    export AWS_ACCESS_KEY_ID="$MODEL_S3_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$MODEL_S3_SECRET_KEY"
    export AWS_DEFAULT_REGION="$MODEL_S3_REGION"
    
    # Try AWS CLI first, then Python
    if command -v aws &> /dev/null; then
        aws s3 cp "s3://${MODEL_S3_BUCKET}/${MODEL_S3_PREFIX}exp-schp-201908301523-atr.pth" "$MODEL_FILE" && {
            echo "✓ Model downloaded via AWS CLI"
        } || {
            echo "AWS CLI download failed, trying Python..."
        }
    fi
    
    # Fallback to Python minio client
    if [ ! -f "$MODEL_FILE" ]; then
        python3 << EOF
import os
import sys
try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "minio", "-q"])
    from minio import Minio
    from minio.error import S3Error

bucket = os.environ.get('MODEL_S3_BUCKET', 'xapienappassets')
prefix = os.environ.get('MODEL_S3_PREFIX', 'models/')
endpoint = os.environ.get('MODEL_S3_ENDPOINT', 's3.amazonaws.com')
access_key = os.environ.get('MODEL_S3_ACCESS_KEY')
secret_key = os.environ.get('MODEL_S3_SECRET_KEY')
region = os.environ.get('MODEL_S3_REGION', 'ap-south-1')
object_name = f"{prefix}exp-schp-201908301523-atr.pth"
local_path = "models/exp-schp-201908301523-atr.pth"

print(f"Downloading from s3://{bucket}/{object_name}")

try:
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region
    )
    client.fget_object(bucket, object_name, local_path)
    file_size = os.path.getsize(local_path)
    print(f"✓ Download complete! Size: {file_size / (1024*1024):.1f} MB")
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)
EOF
    fi
else
    echo "⚠️  Model not found and S3 credentials not set"
    echo "   Please download manually or set S3 credentials"
    echo "   Manual download: https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K1j6XmTd3yisGXq7/view"
    echo "   Save as: $MODEL_FILE"
fi

# ==========================================
# STEP 7: VERIFY INSTALLATION
# ==========================================
echo ""
echo "[7/7] Verifying installation..."

python -c "
import sys
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
    
    import torchvision
    print(f'✓ torchvision')
    
    import PIL
    print(f'✓ Pillow')
    
    import cv2
    print(f'✓ OpenCV')
    
    import fastapi
    print(f'✓ FastAPI')
    
    import uvicorn
    print(f'✓ Uvicorn')
    
    import aiohttp
    print(f'✓ aiohttp')
    
    # Test SCHP
    sys.path.insert(0, 'Self-Correction-Human-Parsing')
    import networks
    print(f'✓ SCHP networks')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || {
    echo ""
    echo "⚠️  Some imports failed. Running full setup..."
    ./setup.sh
}

# ==========================================
# SET SERVICE ENVIRONMENT VARIABLES
# ==========================================
echo ""
echo "=========================================="
echo "Environment Configuration"
echo "=========================================="

export MASK_ASSET_URL=${MASK_ASSET_URL:-"http://51.20.86.26:9009/v1/mask/upload"}
export MASK_COMPONENT_SECRET=${MASK_COMPONENT_SECRET:-"supersecret-internal-token"}
export PORT=${PORT:-8000}
export HOST=${HOST:-"0.0.0.0"}
export QUEUE_MAX_SIZE=${QUEUE_MAX_SIZE:-100}
export WORKER_COUNT=${WORKER_COUNT:-2}
export TEMP_FOLDER=${TEMP_FOLDER:-"/tmp/qwen"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export SCHP_DEVICE=${SCHP_DEVICE:-"cpu"}

echo "Port: $PORT"
echo "Host: $HOST"
echo "Workers: $WORKER_COUNT"
echo "Queue Size: $QUEUE_MAX_SIZE"
echo "Device: $SCHP_DEVICE"
echo "Mask Asset URL: $MASK_ASSET_URL"
echo "S3 Enabled: $MODEL_S3_ENABLED"

# ==========================================
# START THE SERVICE
# ==========================================
echo ""
echo "=========================================="
echo "Starting Qwen Mask Component..."
echo "=========================================="
echo ""

# Ensure venv is activated
source ${VENV_DIR}/bin/activate

# Start the service
exec python -m uvicorn app:app --host "$HOST" --port "$PORT"
