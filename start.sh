#!/bin/bash
# Start Qwen Mask Component Service

# Determine working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables (if not already set)
export MASK_ASSET_URL=${MASK_ASSET_URL:-"http://51.20.86.26:9009/v1/mask/upload"}
export MASK_COMPONENT_SECRET=${MASK_COMPONENT_SECRET:-"supersecret-internal-token"}
export PORT=${PORT:-8003}
export HOST=${HOST:-"0.0.0.0"}
export QUEUE_MAX_SIZE=${QUEUE_MAX_SIZE:-100}
export WORKER_COUNT=${WORKER_COUNT:-2}
export TEMP_FOLDER=${TEMP_FOLDER:-"/tmp/qwen"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export SCHP_DEVICE=${SCHP_DEVICE:-"cpu"}

# S3 Model Storage Configuration (for downloading models from AWS S3)
# IMPORTANT: Set these environment variables before running!
#   export MODEL_S3_ACCESS_KEY="your-access-key"
#   export MODEL_S3_SECRET_KEY="your-secret-key"
export MODEL_S3_ENABLED=${MODEL_S3_ENABLED:-"true"}
export MODEL_S3_ENDPOINT=${MODEL_S3_ENDPOINT:-"s3.amazonaws.com"}
export MODEL_S3_BUCKET=${MODEL_S3_BUCKET:-"xapienappassets"}
export MODEL_S3_PREFIX=${MODEL_S3_PREFIX:-"models/"}
export MODEL_S3_REGION=${MODEL_S3_REGION:-"ap-south-1"}

# Validate required S3 credentials
if [ "$MODEL_S3_ENABLED" = "true" ]; then
    if [ -z "$MODEL_S3_ACCESS_KEY" ] || [ -z "$MODEL_S3_SECRET_KEY" ]; then
        echo "⚠️  WARNING: S3 model download is enabled but credentials are not set!"
        echo "   Set the following environment variables before running:"
        echo "   export MODEL_S3_ACCESS_KEY=\"your-access-key\""
        echo "   export MODEL_S3_SECRET_KEY=\"your-secret-key\""
        echo ""
    fi
fi

echo "=========================================="
echo "Starting Qwen Mask Component"
echo "=========================================="
echo "Port: $PORT"
echo "Host: $HOST"
echo "Workers: $WORKER_COUNT"
echo "Queue Max Size: $QUEUE_MAX_SIZE"
echo "Mask Asset URL: $MASK_ASSET_URL"
echo "S3 Model Enabled: $MODEL_S3_ENABLED"
echo "=========================================="
echo ""

# Start the service using python -m uvicorn (more reliable than direct uvicorn command)
python -m uvicorn app:app --host "$HOST" --port "$PORT"
