"""Preprocessing endpoints available in both local and cloud runtimes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.state import get_model_manager
from app.services.preprocessing import PreprocessingService

logger = logging.getLogger(__name__)
router = APIRouter()


class _TiktokenAdapter:
    """Small tokenizer adapter for cloud mode, where HF weights are skipped."""

    def __init__(self):
        import tiktoken

        self._encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str, add_special_tokens: bool = True):  # noqa: ARG002
        return self._encoding.encode(str(text or ""))


class ChunkingRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(min_length=1)
    chunking_strategy: Optional[str] = "auto"
    target_chunk_size: Optional[int] = Field(None, ge=32, le=settings.MAX_SEQUENCE_LENGTH)


class ChunkingResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    total_chunks: int
    chunking_stats: Dict[str, Any]


class PreprocessingStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None


_preprocessing_service: Optional[PreprocessingService] = None


def get_preprocessing_service() -> PreprocessingService:
    """Reuse one worker-backed preprocessor and avoid requiring local HF in cloud mode."""
    global _preprocessing_service
    if _preprocessing_service is not None:
        return _preprocessing_service
    try:
        if settings.CLOUD_MODE:
            tokenizer = _TiktokenAdapter()
        else:
            tokenizer, _, _, _ = get_model_manager().get_embedding_model()
        _preprocessing_service = PreprocessingService(tokenizer)
        return _preprocessing_service
    except Exception as exc:
        logger.error("Failed to initialize preprocessing service: %s", exc)
        raise HTTPException(status_code=503, detail="Preprocessing service not available")


def _stats(chunks: List[Dict[str, Any]], **extra) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        **extra,
        "total_chunks": len(chunks),
        "average_chunk_length": (
            sum(len(str(chunk.get("content") or "")) for chunk in chunks) / len(chunks)
            if chunks else 0
        ),
        "chunking_strategies": {},
    }
    for chunk in chunks:
        strategy = (chunk.get("metadata") or {}).get("content_type", "unknown")
        stats["chunking_strategies"][strategy] = stats["chunking_strategies"].get(strategy, 0) + 1
    return stats


@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_data(request: ChunkingRequest):
    try:
        service = get_preprocessing_service()
        frame = pd.DataFrame(request.data)
        target_size = request.target_chunk_size or settings.TARGET_CHUNK_SIZE
        chunks = await service.chunk_medical_dialogues(frame, target_chunk_size=target_size)
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_stats=_stats(chunks, total_input_rows=len(frame)),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error chunking data: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunking failed: {exc}")


@router.post("/upload-and-chunk", response_model=ChunkingResponse)
async def upload_and_chunk_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("auto"),  # preserved for API compatibility
    target_chunk_size: int = Form(settings.TARGET_CHUNK_SIZE),
):
    del chunking_strategy
    try:
        if target_chunk_size < 32 or target_chunk_size > settings.MAX_SEQUENCE_LENGTH:
            raise HTTPException(status_code=400, detail="target_chunk_size is outside the supported range")
        filename = file.filename or "upload"
        extension = Path(filename).suffix.lower()
        if extension not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type {extension} not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}",
            )
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes",
            )
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 encoded") from exc

        if extension == ".csv":
            from io import StringIO

            frame = pd.read_csv(StringIO(decoded))
        elif extension == ".json":
            frame = pd.DataFrame(json.loads(decoded))
        else:
            # ALLOWED_FILE_TYPES may contain types supported by other upload
            # routes; this dialogue preprocessor intentionally accepts CSV/JSON.
            raise HTTPException(status_code=400, detail="This preprocessing endpoint supports CSV or JSON")

        service = get_preprocessing_service()
        chunks = await service.chunk_medical_dialogues(frame, target_chunk_size=target_chunk_size)
        return ChunkingResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            chunking_stats=_stats(
                chunks,
                filename=filename,
                file_size=len(content),
                total_input_rows=len(frame),
            ),
        )
    except HTTPException:
        raise
    except (json.JSONDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid upload content: {exc}") from exc
    except Exception as exc:
        logger.error("Error processing uploaded file: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {exc}")


@router.get("/preprocessing/info")
async def get_preprocessing_info():
    return {
        "target_chunk_size": settings.TARGET_CHUNK_SIZE,
        "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
        "allowed_file_types": settings.ALLOWED_FILE_TYPES,
        "max_file_size": settings.MAX_FILE_SIZE,
        "batch_size": settings.BATCH_SIZE,
        "tokenizer_backend": "tiktoken" if settings.CLOUD_MODE else "local-hf",
    }


@router.post("/validate-chunks")
async def validate_chunks(chunks: List[Dict[str, Any]]):
    try:
        service = get_preprocessing_service()
        validation = await service.validate_chunks(chunks)
        return {
            "valid_chunks": len(validation["valid_chunks"]),
            "invalid_chunks": len(validation["invalid_chunks"]),
            "validation_errors": validation["errors"],
            "statistics": validation["statistics"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error validating chunks: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chunk validation failed: {exc}")
