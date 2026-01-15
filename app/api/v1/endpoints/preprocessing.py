from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.state import get_model_manager
from app.core.config import settings
from app.services.preprocessing import PreprocessingService

logger = logging.getLogger(__name__)
router = APIRouter()

class ChunkingRequest(BaseModel):
    """Request model for text chunking."""
    data: List[Dict[str, Any]]
    chunking_strategy: Optional[str] = "auto"
    target_chunk_size: Optional[int] = None

class ChunkingResponse(BaseModel):
    """Response model for text chunking."""
    chunks: List[Dict[str, Any]]
    total_chunks: int
    chunking_stats: Dict[str, Any]

class PreprocessingStatus(BaseModel):
    """Status model for preprocessing operations."""
    status: str
    message: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

def get_preprocessing_service():
    """Dependency to get preprocessing service."""
    try:
        tokenizer, _, _, _ = get_model_manager().get_embedding_model()
        return PreprocessingService(tokenizer)
    except Exception as e:
        logger.error(f"Failed to get preprocessing service: {e}")
        raise HTTPException(status_code=503, detail="Preprocessing service not available")

@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_data(
    request: ChunkingRequest,
    preprocessing_service = Depends(get_preprocessing_service)
):
    """Chunk data using the preprocessing service."""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Set chunking parameters
        target_size = request.target_chunk_size or settings.TARGET_CHUNK_SIZE
        
        # Process data
        chunks = await preprocessing_service.chunk_medical_dialogues(
            df, 
            target_chunk_size=target_size
        )
        
        # Calculate statistics
        chunking_stats = {
            "total_input_rows": len(df),
            "total_chunks": len(chunks),
            "average_chunk_length": sum(len(chunk['content']) for chunk in chunks) / len(chunks) if chunks else 0,
            "chunking_strategies": {}
        }
        
        # Count chunking strategies
        for chunk in chunks:
            strategy = chunk['metadata'].get('content_type', 'unknown')
            chunking_stats["chunking_strategies"][strategy] = chunking_stats["chunking_strategies"].get(strategy, 0) + 1
        
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_stats=chunking_stats
        )
        
    except Exception as e:
        logger.error(f"Error chunking data: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

@router.post("/upload-and-chunk", response_model=ChunkingResponse)
async def upload_and_chunk_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("auto"),
    target_chunk_size: int = Form(settings.TARGET_CHUNK_SIZE),
    preprocessing_service = Depends(get_preprocessing_service)
):
    """Upload a file and chunk its contents."""
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Parse file content
        if file_extension == '.csv':
            df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file_extension == '.json':
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Process data
        chunks = await preprocessing_service.chunk_medical_dialogues(
            df, 
            target_chunk_size=target_chunk_size
        )
        
        # Calculate statistics
        chunking_stats = {
            "filename": file.filename,
            "file_size": len(content),
            "total_input_rows": len(df),
            "total_chunks": len(chunks),
            "average_chunk_length": sum(len(chunk['content']) for chunk in chunks) / len(chunks) if chunks else 0,
            "chunking_strategies": {}
        }
        
        # Count chunking strategies
        for chunk in chunks:
            strategy = chunk['metadata'].get('content_type', 'unknown')
            chunking_stats["chunking_strategies"][strategy] = chunking_stats["chunking_strategies"].get(strategy, 0) + 1
        
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_stats=chunking_stats
        )
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@router.get("/preprocessing/info")
async def get_preprocessing_info():
    """Get preprocessing service information."""
    return {
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
        "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
        "allowed_file_types": settings.ALLOWED_FILE_TYPES,
        "max_file_size": settings.MAX_FILE_SIZE,
        "batch_size": settings.BATCH_SIZE
    }

@router.post("/validate-chunks")
async def validate_chunks(
    chunks: List[Dict[str, Any]],
    preprocessing_service = Depends(get_preprocessing_service)
):
    """Validate chunks for consistency and quality."""
    try:
        validation_results = await preprocessing_service.validate_chunks(chunks)
        
        return {
            "valid_chunks": len(validation_results["valid_chunks"]),
            "invalid_chunks": len(validation_results["invalid_chunks"]),
            "validation_errors": validation_results["errors"],
            "statistics": validation_results["statistics"]
        }
        
    except Exception as e:
        logger.error(f"Error validating chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk validation failed: {str(e)}")
