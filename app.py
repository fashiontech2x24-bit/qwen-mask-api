#!/usr/bin/env python3
"""
Qwen Mask Component - Simplified FastAPI Service
Human Parsing & Body-Region Mask Generator with FIFO Queue
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from routers.process import router as process_router
from workers.queue_worker import start_workers, stop_workers
from services.qwen_segmentation import QwenSegmentationService

# Setup logging
logging.basicConfig(
    level=os.environ.get('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Mask Component", version="1.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
segmentation_service: Optional[QwenSegmentationService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global segmentation_service
    
    try:
        # Initialize segmentation service
        config_path = os.environ.get('SCHP_CONFIG_PATH', 'config/config.yaml')
        device = os.environ.get('SCHP_DEVICE', 'cpu')
        
        logger.info(f"Initializing Qwen Segmentation Service (device={device})...")
        segmentation_service = QwenSegmentationService(config_path=config_path, device=device)
        logger.info("Segmentation service initialized successfully")
        
        # Start background workers
        worker_count = int(os.environ.get('WORKER_COUNT', 2))
        logger.info(f"Starting {worker_count} background workers...")
        await start_workers(worker_count, segmentation_service)
        logger.info("Background workers started")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down workers...")
    await stop_workers()
    logger.info("Shutdown complete")


@app.get("/health")
async def health():
    """Health check endpoint."""
    from workers.queue_worker import get_queue_stats
    
    stats = get_queue_stats()
    return {
        "status": "ok",
        "queue_size": stats.get("queue_current", 0),
        "segmentation_ready": segmentation_service is not None
    }


@app.get("/stats")
async def stats():
    """System metrics endpoint."""
    from workers.queue_worker import get_queue_stats, get_worker_stats
    
    queue_stats = get_queue_stats()
    worker_stats = get_worker_stats()
    
    return {
        "queue_max": queue_stats.get("queue_max", 100),
        "queue_current": queue_stats.get("queue_current", 0),
        "workers": worker_stats.get("worker_count", 2),
        "jobs_processed": worker_stats.get("jobs_processed", 0),
        "uptime_sec": int(time.time() - worker_stats.get("start_time", time.time()))
    }


# Include routers
app.include_router(process_router, prefix="", tags=["processing"])


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8003))
    host = os.environ.get('HOST', '0.0.0.0')
    uvicorn.run(app, host=host, port=port, log_level="info")

