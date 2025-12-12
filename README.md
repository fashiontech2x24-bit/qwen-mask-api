# Qwen Mask Component

A simplified FastAPI microservice for generating upper-body and lower-body masks from human images using Qwen-based segmentation (SCHP).

## Features

- ✅ **FIFO Queue**: Internal async queue with guaranteed FIFO ordering
- ✅ **Background Workers**: Configurable worker pool (default: 2)
- ✅ **Mask Generation**: Creates `upper_body_mask.png` and `lower_body_mask.png`
- ✅ **Callback Service**: Automatic upload to Mask Asset Service with retry logic
- ✅ **Production Ready**: Error handling, logging, health checks
- ✅ **Scalable**: Horizontally scalable, stateless design

## Quick Start

```bash
# Setup (first time only)
./setup.sh
source venv/bin/activate

# Run service (option 1: using startup script)
./start.sh

# Run service (option 2: manually)
source venv/bin/activate
export MASK_COMPONENT_SECRET=your-secret-key  # Required for callbacks
uvicorn app:app --host 0.0.0.0 --port 8003
```

## Environment Variables

```bash
PORT=8003                    # Server port
HOST=0.0.0.0                 # Server host
QUEUE_MAX_SIZE=100            # Maximum queue size
WORKER_COUNT=2                # Number of background workers
MASK_ASSET_URL=https://apiprod.xapien.in:9002/v1/mask/upload  # Mask Asset Service URL (default)
MASK_COMPONENT_SECRET=supersecret-internal-token  # Secret for X-Internal-Auth header (default)
TEMP_FOLDER=/tmp/qwen         # Temporary folder for processing
LOG_LEVEL=INFO                # Logging level
SCHP_DEVICE=cpu               # Device: cpu, cuda, or auto
SCHP_CONFIG_PATH=config/config.yaml
```

## API Endpoints

### POST /process

Accepts job and adds to queue.

**Request:**
- `user_image`: Image file (multipart/form-data)
- `user_id`: User ID (form field)
- `job_id`: Job ID (form field)
- `provider`: "qwen" (form field, default: "qwen")

**Response:**
```json
{
  "status": "accepted",
  "queued": true,
  "job_id": "f20d-441e-a7db...",
  "message": "Job added to Qwen mask queue"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "queue_size": 3,
  "segmentation_ready": true
}
```

### GET /stats

System metrics endpoint.

**Response:**
```json
{
  "queue_max": 100,
  "queue_current": 3,
  "workers": 2,
  "jobs_processed": 441,
  "uptime_sec": 8420
}
```

## Usage Example

```bash
curl -X POST "http://localhost:8003/process" \
  -F "user_image=@image.jpg" \
  -F "user_id=user123" \
  -F "job_id=job456" \
  -F "provider=qwen"
```

## Architecture

1. **Request** → `/process` endpoint accepts job instantly
2. **Queue** → Job added to FIFO queue (asyncio.Queue)
3. **Worker** → Background worker picks job in FIFO order
4. **Processing**:
   - Load and preprocess image
   - Run SCHP segmentation
   - Generate `upper_body_mask.png` and `lower_body_mask.png`
5. **Callback** → Upload masks to Mask Asset Service (with retry)
6. **Cleanup** → Remove temporary files

## Mask Generation

The service generates two binary mask PNG files:

- **upper_body_mask.png**: Shoulders → chest → arms → torso (labels: 4, 7)
- **lower_body_mask.png**: Hips → legs → calves (labels: 5, 6, 7)

Masks are binary (255 for target regions, 0 otherwise).

## Callback Service

After generating masks, the service automatically uploads them to the Mask Asset Service:

- **URL**: Configured via `MASK_ASSET_URL`
- **Auth**: Uses `X-Internal-Auth` header with `MASK_COMPONENT_SECRET`
- **Retry**: 5 attempts with exponential backoff (1s → 2s → 4s → 8s → 16s)

## Docker

```bash
# Build
docker build -t qwen-mask-component .

# Run
docker run -p 8003:8003 \
  -e MASK_ASSET_URL=https://maskassets.xapien.in/mask/upload \
  -e MASK_COMPONENT_SECRET=your-secret \
  qwen-mask-component
```

## Configuration

See [config/config.yaml](config/config.yaml) for model and processing configuration.

## Performance Targets

- Preprocessing: < 80ms
- Segmentation (CPU): 200-350ms
- Mask construction: < 50ms
- Callback upload: < 200ms
- **Total per job**: 0.5-0.8 seconds

With 2 workers: ~2 jobs per second

## License

MIT License
