"""Section-aware passage chunking with token overlap."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import settings

try:
    import tiktoken
except ModuleNotFoundError:
    class tiktoken:
        @staticmethod
        def get_encoding(_name):
            class _Enc:
                def encode(self, text):
                    return text.split()
            return _Enc()

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def section_chunks(
    doc_id: str,
    text: str,
    default_section: Optional[str] = None,
    target_tokens: Optional[int] = None,
    overlap_tokens: int = 40,
) -> List[Dict[str, Any]]:
    """Split documents into 350–450 token section-aware passages."""
    if not text:
        return []
    target = target_tokens or max(350, min(450, settings.TARGET_CHUNK_SIZE))
    chunks: List[Dict[str, Any]] = []
    offset = 0
    section = default_section
    buffer: List[str] = []
    buffer_start = 0
    chunk_idx = 0

    def flush(end: int) -> None:
        nonlocal chunk_idx, buffer, buffer_start
        if not buffer:
            return
        body = "\n\n".join(buffer).strip()
        if not body:
            buffer = []
            return
        # Split oversized sections with overlap.
        words = body.split()
        if _token_len(body) <= target + 50:
            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                "text": body,
                "content": body,
                "section": section,
                "offset_start": buffer_start,
                "offset_end": end,
                "metadata": {},
            })
            chunk_idx += 1
            buffer = []
            return
        start = 0
        while start < len(words):
            window = []
            tokens = 0
            i = start
            while i < len(words) and tokens < target:
                window.append(words[i])
                tokens = _token_len(" ".join(window))
                i += 1
            part = " ".join(window).strip()
            if part:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "text": part,
                    "content": part,
                    "section": section,
                    "offset_start": buffer_start,
                    "offset_end": end,
                    "metadata": {"overlap_tokens": overlap_tokens},
                })
                chunk_idx += 1
            if i >= len(words):
                break
            start = max(i - overlap_tokens, start + 1)
        buffer = []

    for paragraph in [part for part in text.split("\n\n") if part.strip()]:
        stripped = paragraph.strip()
        start = text.find(paragraph, offset)
        if start < 0:
            start = offset
        offset = start + len(paragraph)
        is_heading = len(stripped) <= 120 and not stripped.endswith(".") and len(stripped.split()) <= 12
        if is_heading and buffer:
            flush(start)
        if is_heading:
            section = stripped
            buffer_start = offset
            continue
        if not buffer:
            buffer_start = start
        buffer.append(stripped)
        if _token_len("\n\n".join(buffer)) >= target:
            flush(offset)
    flush(len(text))
    return chunks
