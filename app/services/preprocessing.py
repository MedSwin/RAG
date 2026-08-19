"""Dialogue preprocessing/chunking service used by the admin API.

This module is intentionally self-contained. Older revisions imported a
repository-external ``preprocessing.chunker`` module even though the imported
function was never used; that made the endpoint fail at runtime on a clean
checkout. The implementation below keeps the public chunk schema while using
the configured tokenizer directly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Token-aware dialogue chunking with one bounded worker pool."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="medswin-preprocess")

    async def chunk_medical_dialogues(
        self,
        df: pd.DataFrame,
        target_chunk_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        target_size = int(target_chunk_size or settings.TARGET_CHUNK_SIZE)
        if target_size <= 0 or target_size > settings.MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"target_chunk_size must be between 1 and {settings.MAX_SEQUENCE_LENGTH}"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._chunk_medical_dialogues_sync,
            df,
            target_size,
        )

    def _chunk_medical_dialogues_sync(
        self,
        df: pd.DataFrame,
        target_chunk_size: int,
    ) -> List[Dict[str, Any]]:
        frame = self._preprocess_data(df)
        chunks: List[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            chunks.extend(self._chunk_row(row, target_chunk_size))
        return self._validate_chunks(chunks)

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        frame = df.copy()
        qca = {"question", "context", "answer"}.issubset(frame.columns)
        io = {"input", "output"}.issubset(frame.columns)
        if not qca and not io:
            raise ValueError(
                "Input data must contain question/context/answer or input/output columns"
            )

        if qca:
            for column in ("question", "context", "answer"):
                frame[column] = frame[column].fillna("").astype(str)
            frame["input"] = frame["question"] + "\n\nContext: " + frame["context"]
            frame["output"] = frame["answer"]
        else:
            frame["input"] = frame["input"].fillna("").astype(str)
            frame["output"] = frame["output"].fillna("").astype(str)

        frame = frame[
            (frame["input"].str.strip() != "")
            & (frame["output"].str.strip() != "")
        ].copy()
        if "id" not in frame.columns:
            frame["id"] = range(len(frame))
        if "source" not in frame.columns:
            frame["source"] = "unknown"
        if "task" not in frame.columns:
            frame["task"] = "medical_dialogue"
        return frame

    def _row_text_and_qca(self, row: pd.Series) -> tuple[str, Optional[Dict[str, str]]]:
        if all(name in row.index for name in ("question", "context", "answer")):
            question = str(row.get("question", "")).strip()
            context = str(row.get("context", "")).strip()
            answer = str(row.get("answer", "")).strip()
            parts = [f"Question: {question}"]
            if context:
                parts.append(f"Context: {context}")
            parts.append(f"Answer: {answer}")
            return "\n\n".join(parts), {
                "question": question,
                "context": context,
                "answer": answer,
            }
        return (
            f"Input: {str(row.get('input', '')).strip()}\n\n"
            f"Output: {str(row.get('output', '')).strip()}",
            None,
        )

    def _chunk_row(self, row: pd.Series, target_chunk_size: int) -> List[Dict[str, Any]]:
        content, qca = self._row_text_and_qca(row)
        if not content.strip():
            return []
        parts = self._split_text_by_token(content, target_chunk_size)
        if not parts:
            return []

        source = str(row.get("source", "unknown"))
        parent_id = str(row.get("id", ""))
        task = str(row.get("task", "medical_dialogue"))
        total = len(parts)
        output: List[Dict[str, Any]] = []
        for index, part in enumerate(parts, start=1):
            if total == 1:
                chunk_id = f"{source}_{parent_id}_single"
                content_type = "complete_dialogue"
            else:
                chunk_id = f"{source}_{parent_id}_{index:02d}"
                content_type = f"dialogue_part{index}"
            metadata: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "source": source,
                "sequence": index,
                "total_chunks": total,
                "content_type": content_type,
                "task": task,
                "chunk_length": len(part),
                "token_count": self._count_tokens(part),
                "created_timestamp": datetime.now(timezone.utc),
                "related_chunks": [
                    (
                        f"{source}_{parent_id}_{other:02d}"
                        if total > 1
                        else f"{source}_{parent_id}_single"
                    )
                    for other in range(1, total + 1)
                    if other != index
                ],
            }
            if qca:
                metadata.update(qca)
            output.append({"content": part, "metadata": metadata})
        return output

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        except TypeError:
            # tiktoken-style adapters do not need/accept HF kwargs unless the
            # endpoint wrapper supplies them; keep this service tokenizer-agnostic.
            return len(self.tokenizer.encode(text))

    def _split_text_by_token(self, text: str, max_tokens: int) -> List[str]:
        if self._count_tokens(text) <= max_tokens:
            return [text.strip()]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
            if sentence.strip()
        ]
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current}\n\n{sentence}".strip() if current else sentence
            if self._count_tokens(candidate) <= max_tokens:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            if self._count_tokens(sentence) <= max_tokens:
                current = sentence
            else:
                chunks.extend(self._force_split_by_words(sentence, max_tokens))
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk.strip()]

    def _force_split_by_words(self, text: str, max_tokens: int) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and self._count_tokens(candidate) > max_tokens:
                chunks.append(current)
                current = word
            elif not current and self._count_tokens(word) > max_tokens:
                # A pathological token/word still needs deterministic progress.
                chunks.append(word)
                current = ""
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = str(chunk.get("content") or "").strip()
            metadata = chunk.get("metadata") or {}
            if not content:
                continue
            token_count = int(metadata.get("token_count") or self._count_tokens(content))
            if token_count > settings.MAX_SEQUENCE_LENGTH:
                logger.warning(
                    "Dropping chunk %s with %s tokens (> %s)",
                    metadata.get("chunk_id"),
                    token_count,
                    settings.MAX_SEQUENCE_LENGTH,
                )
                continue
            valid.append(chunk)
        return valid

    async def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_chunks: List[Dict[str, Any]] = []
        invalid_chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        required = {
            "chunk_id",
            "parent_id",
            "source",
            "task",
            "sequence",
            "total_chunks",
            "content_type",
        }
        for index, chunk in enumerate(chunks):
            chunk_errors: List[str] = []
            if not str(chunk.get("content") or "").strip():
                chunk_errors.append("Empty content")
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                chunk_errors.append("Missing metadata")
            else:
                for field in sorted(required - set(metadata)):
                    chunk_errors.append(f"Missing metadata field: {field}")
                if int(metadata.get("token_count") or 0) > settings.MAX_SEQUENCE_LENGTH:
                    chunk_errors.append(
                        f"Token count exceeds limit: {metadata.get('token_count')}"
                    )
            if chunk_errors:
                invalid_chunks.append(chunk)
                errors.extend(f"Chunk {index}: {error}" for error in chunk_errors)
            else:
                valid_chunks.append(chunk)
        return {
            "valid_chunks": valid_chunks,
            "invalid_chunks": invalid_chunks,
            "errors": errors,
            "statistics": {
                "total_chunks": len(chunks),
                "valid_chunks": len(valid_chunks),
                "invalid_chunks": len(invalid_chunks),
                "validation_errors": len(errors),
            },
        }

    def cleanup(self) -> None:
        self.executor.shutdown(wait=True)
