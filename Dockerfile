# Qwen Mask Component Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    git \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Using PyTorch 1.12.0 (required for SCHP compatibility)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==1.12.0+cpu torchvision==0.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Clone SCHP repository if not exists
RUN if [ ! -d "Self-Correction-Human-Parsing" ]; then \
        git clone https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git || true; \
    fi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs outputs checkpoints /tmp/qwen

# Set environment variables
ENV PYTHONPATH=/app:/app/Self-Correction-Human-Parsing
ENV SCHP_DEVICE=cpu
ENV PORT=8003
ENV HOST=0.0.0.0
ENV QUEUE_MAX_SIZE=100
ENV WORKER_COUNT=2
ENV TEMP_FOLDER=/tmp/qwen
ENV LOG_LEVEL=INFO

EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
