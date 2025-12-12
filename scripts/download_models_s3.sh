#!/bin/bash
# Download models from S3/AWS
# This script downloads the required SCHP model from AWS S3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Model configuration
MODEL_DIR="models"
MODEL_FILE="exp-schp-201908301523-atr.pth"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

# S3 configuration (can be overridden by environment variables)
S3_BUCKET="${MODEL_S3_BUCKET:-xapienappassets}"
S3_PREFIX="${MODEL_S3_PREFIX:-models/}"
S3_REGION="${MODEL_S3_REGION:-ap-south-1}"
S3_ENDPOINT="${MODEL_S3_ENDPOINT:-s3.amazonaws.com}"

# AWS Credentials - MUST be set as environment variables
AWS_ACCESS_KEY="${MODEL_S3_ACCESS_KEY:-$AWS_ACCESS_KEY_ID}"
AWS_SECRET_KEY="${MODEL_S3_SECRET_KEY:-$AWS_SECRET_ACCESS_KEY}"

echo "=========================================="
echo "Qwen Mask Model Downloader"
echo "=========================================="

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Model already exists: $MODEL_PATH"
    echo "  Size: $(du -h "$MODEL_PATH" | cut -f1)"
    exit 0
fi

# Check for credentials
if [ -z "$AWS_ACCESS_KEY" ] || [ -z "$AWS_SECRET_KEY" ]; then
    echo "❌ AWS credentials not set!"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export MODEL_S3_ACCESS_KEY=\"your-access-key\""
    echo "  export MODEL_S3_SECRET_KEY=\"your-secret-key\""
    echo ""
    echo "Or use AWS CLI credentials:"
    echo "  export AWS_ACCESS_KEY_ID=\"your-access-key\""
    echo "  export AWS_SECRET_ACCESS_KEY=\"your-secret-key\""
    exit 1
fi

# Create models directory
mkdir -p "$MODEL_DIR"

echo "Downloading model from S3..."
echo "  Bucket: $S3_BUCKET"
echo "  Object: ${S3_PREFIX}${MODEL_FILE}"
echo "  Region: $S3_REGION"
echo ""

# Method 1: Try using AWS CLI if available
if command -v aws &> /dev/null; then
    echo "Using AWS CLI..."
    export AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$AWS_SECRET_KEY"
    export AWS_DEFAULT_REGION="$S3_REGION"
    
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}${MODEL_FILE}" "$MODEL_PATH" && {
        echo ""
        echo "✓ Model downloaded successfully: $MODEL_PATH"
        echo "  Size: $(du -h "$MODEL_PATH" | cut -f1)"
        exit 0
    }
fi

# Method 2: Use Python with minio library
echo "Using Python minio client..."

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python3 << EOF
import os
import sys

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    print("Installing minio library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "minio", "-q"])
    from minio import Minio
    from minio.error import S3Error

# Configuration
bucket = "${S3_BUCKET}"
object_name = "${S3_PREFIX}${MODEL_FILE}"
local_path = "${MODEL_PATH}"
endpoint = "${S3_ENDPOINT}"
access_key = "${AWS_ACCESS_KEY}"
secret_key = "${AWS_SECRET_KEY}"
region = "${S3_REGION}"

print(f"Connecting to S3 endpoint: {endpoint}")

try:
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region
    )
    
    print(f"Downloading: {bucket}/{object_name}")
    print(f"To: {local_path}")
    
    client.fget_object(bucket, object_name, local_path)
    
    file_size = os.path.getsize(local_path)
    print(f"\n✓ Download complete! Size: {file_size / (1024*1024):.1f} MB")
    
except S3Error as e:
    print(f"\n❌ S3 Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model downloaded successfully: $MODEL_PATH"
else
    echo ""
    echo "❌ Model download failed!"
    echo "Please download manually from:"
    echo "  https://drive.google.com/file/d/1ruJg4lqR_jgQPj-9K1j6XmTd3yisGXq7/view"
    echo "  Save as: $MODEL_PATH"
    exit 1
fi
