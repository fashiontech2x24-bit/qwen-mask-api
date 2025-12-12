"""
Process endpoint router.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from workers.queue_worker import enqueue_job, is_queue_full

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process")
async def process_image(
    user_image: UploadFile = File(...),
    user_id: str = Form(...),
    job_id: str = Form(...),
    provider: str = Form("qwen")
):
    """
    Accept job and add to FIFO queue.
    
    Returns immediately with acceptance status.
    """
    # Validate provider
    if provider != "qwen":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {provider}. Only 'qwen' is supported."
        )
    
    # Validate image file
    if not user_image.content_type or not user_image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=422,
            detail="Invalid file type. Expected image file."
        )
    
    # Check if queue is full
    if is_queue_full():
        raise HTTPException(
            status_code=429,
            detail="Queue is full. Please try again later."
        )
    
    # Generate job ID if not provided
    if not job_id or job_id == "":
        job_id = str(uuid.uuid4())
    
    try:
        # Read image content
        image_content = await user_image.read()
        
        # Enqueue job
        enqueue_job(
            job_id=job_id,
            user_id=user_id,
            image_content=image_content,
            image_filename=user_image.filename or "image.jpg"
        )
        
        logger.info(f"Job {job_id} enqueued for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "accepted",
                "queued": True,
                "job_id": job_id,
                "message": "Job added to Qwen mask queue"
            }
        )
        
    except Exception as e:
        logger.error(f"Error enqueueing job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue job: {str(e)}"
        )

