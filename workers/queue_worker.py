"""
Queue Worker - FIFO queue with background workers for processing jobs.
"""

import asyncio
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import aiofiles

from services.qwen_segmentation import QwenSegmentationService
from services.mask_builder import generate_masks
from services.callback_service import CallbackService

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Job data structure."""
    job_id: str
    user_id: str
    image_content: bytes
    image_filename: str
    enqueued_at: float


# Global queue and stats
_queue: Optional[asyncio.Queue] = None
_workers: list = []
_worker_stats = {
    "worker_count": 0,
    "jobs_processed": 0,
    "start_time": time.time()
}
_queue_stats = {
    "queue_max": 100,
    "queue_current": 0
}


def get_queue() -> asyncio.Queue:
    """Get or create the global queue."""
    global _queue
    if _queue is None:
        max_size = int(os.environ.get('QUEUE_MAX_SIZE', 100))
        _queue = asyncio.Queue(maxsize=max_size)
        _queue_stats["queue_max"] = max_size
    return _queue


def enqueue_job(job_id: str, user_id: str, image_content: bytes, image_filename: str):
    """Enqueue a job (non-blocking)."""
    queue = get_queue()
    
    if queue.full():
        raise RuntimeError("Queue is full")
    
    job = Job(
        job_id=job_id,
        user_id=user_id,
        image_content=image_content,
        image_filename=image_filename,
        enqueued_at=time.time()
    )
    
    queue.put_nowait(job)
    _queue_stats["queue_current"] = queue.qsize()
    logger.debug(f"Job {job_id} enqueued. Queue size: {queue.qsize()}")


def is_queue_full() -> bool:
    """Check if queue is full."""
    queue = get_queue()
    return queue.full()


def get_queue_stats() -> dict:
    """Get queue statistics."""
    queue = get_queue()
    _queue_stats["queue_current"] = queue.qsize()
    return _queue_stats.copy()


def get_worker_stats() -> dict:
    """Get worker statistics."""
    return _worker_stats.copy()


async def process_job(job: Job, segmentation_service: QwenSegmentationService):
    """
    Process a single job.
    
    Pipeline:
    1. Save image to temp folder
    2. Load and preprocess image
    3. Run segmentation
    4. Generate masks
    5. Upload to Mask Asset Service
    6. Cleanup
    """
    temp_dir = Path(os.environ.get('TEMP_FOLDER', '/tmp/qwen'))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = None
    output_dir = temp_dir / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # [1] Save uploaded image
        input_path = temp_dir / f"{job.job_id}_input_{job.image_filename}"
        async with aiofiles.open(input_path, 'wb') as f:
            await f.write(job.image_content)
        
        logger.info(f"Processing job {job.job_id} for user {job.user_id}")
        
        # [2] Load and preprocess image (already done by saving)
        # [3] Run segmentation
        parse_image, metadata = segmentation_service.parse_image(str(input_path))
        
        # [4] Generate green bounding box masks
        upper_path, lower_path = generate_masks(
            parse_image=parse_image,
            output_dir=output_dir,
            job_id=job.job_id,
            original_image_path=str(input_path)
        )
        
        # Optional: Save seg_map
        seg_map_path = output_dir / f"{job.job_id}_seg_map.png"
        parse_image.save(seg_map_path)
        
        # [5] Upload to Mask Asset Service
        callback_service = CallbackService()
        await callback_service.upload_masks(
            user_id=job.user_id,
            job_id=job.job_id,
            upper_body_mask_path=upper_path,
            lower_body_mask_path=lower_path,
            seg_map_path=seg_map_path
        )
        
        logger.info(f"Job {job.job_id} completed successfully")
        _worker_stats["jobs_processed"] += 1
        
    except Exception as e:
        logger.error(f"Error processing job {job.job_id}: {e}", exc_info=True)
        # Don't re-raise - continue processing other jobs
    
    finally:
        # [6] Cleanup
        try:
            if input_path and input_path.exists():
                input_path.unlink()
            
            # Cleanup output directory
            if output_dir.exists():
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
            
        except Exception as e:
            logger.warning(f"Error cleaning up job {job.job_id}: {e}")


async def worker_loop(worker_id: int, segmentation_service: QwenSegmentationService):
    """Worker loop - processes jobs from queue in FIFO order."""
    queue = get_queue()
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get job from queue (FIFO order guaranteed by asyncio.Queue)
            job = await queue.get()
            _queue_stats["queue_current"] = queue.qsize()
            
            logger.debug(f"Worker {worker_id} processing job {job.job_id}")
            
            # Process job
            await process_job(job, segmentation_service)
            
            # Mark task as done
            queue.task_done()
            _queue_stats["queue_current"] = queue.qsize()
            
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled")
            break
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
            # Continue processing


async def start_workers(count: int, segmentation_service: QwenSegmentationService):
    """Start background workers."""
    global _workers, _worker_stats
    
    if segmentation_service is None:
        raise RuntimeError("Segmentation service not initialized")
    
    _worker_stats["worker_count"] = count
    _worker_stats["start_time"] = time.time()
    
    # Create workers
    for i in range(count):
        task = asyncio.create_task(
            worker_loop(i + 1, segmentation_service)
        )
        _workers.append(task)
    
    logger.info(f"Started {count} workers")


async def stop_workers():
    """Stop all workers."""
    global _workers
    
    # Cancel all workers
    for worker in _workers:
        worker.cancel()
    
    # Wait for cancellation
    if _workers:
        await asyncio.gather(*_workers, return_exceptions=True)
    
    _workers = []
    logger.info("All workers stopped")

