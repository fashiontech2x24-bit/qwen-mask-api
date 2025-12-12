#!/bin/bash
# Setup script for Qwen Mask Component with compatible PyTorch version using venv

set -e

echo "=========================================="
echo "Qwen Mask Component Setup (using venv)"
echo "=========================================="

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Checking for Python 3.8-3.10..."
    PYTHON_CMD=""
    for version in python3.10 python3.9 python3.8 python3; do
        if command -v $version &> /dev/null; then
            PYTHON_VER=$($version --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
            if python3 -c "import sys; sys.exit(0 if (3, 7) <= sys.version_info[:2] <= (3, 10) else 1)" 2>/dev/null; then
                PYTHON_CMD=$version
                echo "Found compatible Python: $PYTHON_CMD ($PYTHON_VER)"
                break
            fi
        fi
    done
    
    if [ -z "$PYTHON_CMD" ]; then
        echo "Error: Python 3.7-3.10 is required for PyTorch 1.12.0"
        echo "Please install Python 3.10: sudo apt-get install python3.10 python3.10-venv"
        exit 1
    fi
else
    PYTHON_CMD="python3.10"
    echo "Using Python 3.10"
fi

# Environment directory
VENV_DIR="venv"
ENV_NAME="venv"

# Check if venv already exists
if [ -d "${VENV_DIR}" ]; then
    echo "Virtual environment '${VENV_DIR}' already exists."
    if [ "${RECREATE_ENV}" = "true" ]; then
        echo "RECREATE_ENV=true, removing existing environment..."
        rm -rf ${VENV_DIR}
    else
        echo "Using existing virtual environment."
        echo "To recreate, run: RECREATE_ENV=true ./setup.sh"
    fi
fi

# Create venv if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment with ${PYTHON_CMD}..."
    echo "This Python version is compatible with PyTorch 1.12.0"
    ${PYTHON_CMD} -m venv ${VENV_DIR}
fi

echo "Activating virtual environment..."
source ${VENV_DIR}/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing compatible PyTorch 1.12.0 CPU-only (required for SCHP)..."
echo "NOTE: PyTorch 2.0+ is NOT compatible - it removed deprecated APIs used by SCHP extensions"
echo "PyTorch 1.12.0 requires Python 3.7-3.10"
echo "Using CPU-only version to avoid CUDA linking issues with CPU-only extensions"
echo "For GPU support, install CUDA version manually after setup"

# Uninstall any existing PyTorch (CUDA or CPU) to ensure clean install
if python -c "import torch" 2>/dev/null; then
    CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "Current PyTorch version: ${CURRENT_TORCH}"
    if [[ "${CURRENT_TORCH}" == "1.12.0"* ]] || [[ "${CURRENT_TORCH}" == "1.12.1"* ]]; then
        # Check if it's CPU version
        if python -c "import torch; print('CPU' if '+cpu' in torch.__version__ else 'CUDA')" 2>/dev/null | grep -q "CPU"; then
            echo "✓ Compatible CPU-only PyTorch version already installed"
        else
            echo "⚠️  CUDA version detected. Reinstalling CPU-only version..."
            pip uninstall -y torch torchvision || true
        fi
    else
        echo "⚠️  Incompatible PyTorch version detected. Reinstalling..."
        pip uninstall -y torch torchvision || true
    fi
else
    echo "PyTorch not installed. Installing CPU-only version..."
fi

# Always install CPU-only version for now
# This prevents CUDA linking issues with CPU-only extensions
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch 1.12.0 CPU-only..."
    pip install torch==1.12.0+cpu torchvision==0.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html 2>&1 | tail -10
    if [ $? -ne 0 ]; then
        echo "⚠️  Installation from PyTorch index failed, trying alternative method..."
        pip install torch==1.12.0 torchvision==0.13.0 --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -10
    fi
fi

echo ""
echo "Installing other dependencies from requirements.txt..."
# Create a temporary requirements file without torch/torchvision
TEMP_REQ=$(mktemp)
grep -v "^torch" requirements.txt | grep -v "^#" | grep -v "^$" > "$TEMP_REQ"
pip install -r "$TEMP_REQ"
rm "$TEMP_REQ"

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p models logs outputs checkpoints /tmp/qwen

# Clone SCHP repository if not exists
if [ ! -d "Self-Correction-Human-Parsing" ]; then
    echo ""
    echo "Cloning SCHP repository..."
    git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git
else
    echo "SCHP repository already exists, skipping clone..."
fi

# Note: SCHP extensions compile automatically on first import
# They use torch.utils.cpp_extension.load() which compiles JIT
echo ""
echo "SCHP C++ extensions will compile automatically on first use."
echo "This may take a few minutes the first time."

# Check for required build tools
echo ""
echo "Checking build tools..."
if ! command -v g++ &> /dev/null; then
    echo "Warning: g++ not found. C++ extensions may not compile."
    echo "Install with: sudo apt-get install build-essential"
fi

# Check for Python development headers (Python.h)
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_INCLUDE_DIR="/usr/include/python${PYTHON_VERSION}"
if [ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" ]; then
    echo ""
    echo "⚠️  Python development headers not found!"
    echo "   Required file: ${PYTHON_INCLUDE_DIR}/Python.h"
    echo ""
    echo "Installing python${PYTHON_VERSION}-dev..."
    if command -v apt-get &> /dev/null; then
        apt-get update && apt-get install -y python${PYTHON_VERSION}-dev || {
            echo "Failed to install automatically. Please run:"
            echo "  sudo apt-get install python${PYTHON_VERSION}-dev"
            exit 1
        }
    else
        echo "Please install Python development headers manually:"
        echo "  Ubuntu/Debian: sudo apt-get install python${PYTHON_VERSION}-dev"
        echo "  Fedora/RHEL:   sudo dnf install python${PYTHON_VERSION}-devel"
        exit 1
    fi
fi

if ! command -v ninja &> /dev/null; then
    echo "Installing ninja (required for PyTorch extensions)..."
    pip install ninja
fi

# Download models from S3 (if credentials are set)
echo ""
echo "=========================================="
echo "Model Download"
echo "=========================================="

if [ -f "models/exp-schp-201908301523-atr.pth" ]; then
    echo "✓ Model already exists: models/exp-schp-201908301523-atr.pth"
elif [ -n "$MODEL_S3_ACCESS_KEY" ] && [ -n "$MODEL_S3_SECRET_KEY" ]; then
    echo "Downloading model from S3..."
    ./scripts/download_models_s3.sh || {
        echo "⚠️  Model download failed, but setup will continue."
        echo "   You can download manually or run: ./scripts/download_models_s3.sh"
    }
elif [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Downloading model from S3 (using AWS credentials)..."
    export MODEL_S3_ACCESS_KEY="$AWS_ACCESS_KEY_ID"
    export MODEL_S3_SECRET_KEY="$AWS_SECRET_ACCESS_KEY"
    ./scripts/download_models_s3.sh || {
        echo "⚠️  Model download failed, but setup will continue."
        echo "   You can download manually or run: ./scripts/download_models_s3.sh"
    }
else
    echo "No S3 credentials found. To download the model:"
    echo ""
    echo "Option 1 - Set credentials and run download script:"
    echo "  export MODEL_S3_ACCESS_KEY=\"your-access-key\""
    echo "  export MODEL_S3_SECRET_KEY=\"your-secret-key\""
    echo "  ./scripts/download_models_s3.sh"
    echo ""
    echo "Option 2 - Download manually:"
    echo "  https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K1j6XmTd3yisGXq7/view"
    echo "  Save as: models/exp-schp-201908301523-atr.pth"
fi

# Test imports
echo "=========================================="
echo "Testing Installation"
echo "=========================================="
python -c "
import sys
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} imported successfully')
    
    import torchvision
    print(f'✓ torchvision imported successfully')
    
    import PIL
    print(f'✓ Pillow imported successfully')
    
    import cv2
    print(f'✓ OpenCV imported successfully')
    
    import yaml
    print(f'✓ PyYAML imported successfully')
    
    import fastapi
    print(f'✓ FastAPI imported successfully')
    
    import aiohttp
    print(f'✓ aiohttp imported successfully')
    
    print('')
    print('✓ All core dependencies imported successfully!')
    
    # Test SCHP imports (may fail if extensions not compiled)
    try:
        sys.path.insert(0, 'Self-Correction-Human-Parsing')
        import networks
        print('✓ SCHP networks imported successfully')
    except Exception as e:
        print(f'⚠ SCHP networks import failed (extensions may need compilation): {e}')
        print('  This is OK - extensions will compile on first use')
    
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the service:"
echo "  uvicorn app:app --host 0.0.0.0 --port 8003"
echo ""
echo "Environment variables:"
echo "  PORT=8003"
echo "  HOST=0.0.0.0"
echo "  QUEUE_MAX_SIZE=100"
echo "  WORKER_COUNT=2"
echo "  MASK_ASSET_URL=https://apiprod.xapien.in:9002/v1/mask/upload"
echo "  MASK_COMPONENT_SECRET=supersecret-internal-token"
echo ""
echo "Note: Download the ATR model checkpoint and place it in models/ directory"
echo "      before running the service."
echo ""
