#!/bin/bash
# Start Qwen Mask Component Service

cd /home/fashionx/qwenmask/qwen-mask-api

# Activate virtual environment
source venv/bin/activate

# Set environment variables (if not already set)
export MASK_ASSET_URL=${MASK_ASSET_URL:-"http://13.62.127.241:9002/v1/mask/upload"}
export MASK_COMPONENT_SECRET=${MASK_COMPONENT_SECRET:-"supersecret-internal-token"}
export PORT=${PORT:-8003}
export HOST=${HOST:-"0.0.0.0"}
export QUEUE_MAX_SIZE=${QUEUE_MAX_SIZE:-100}
export WORKER_COUNT=${WORKER_COUNT:-2}
export TEMP_FOLDER=${TEMP_FOLDER:-"/tmp/qwen"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export SCHP_DEVICE=${SCHP_DEVICE:-"cpu"}

echo "=========================================="
echo "Starting Qwen Mask Component"
echo "=========================================="
echo "Port: $PORT"
echo "Host: $HOST"
echo "Workers: $WORKER_COUNT"
echo "Queue Max Size: $QUEUE_MAX_SIZE"
echo "Mask Asset URL: $MASK_ASSET_URL"
# MASK_COMPONENT_SECRET defaults to 'supersecret-internal-token' if not set
echo "=========================================="
echo ""

# Start the service
uvicorn app:app --host "$HOST" --port "$PORT"

