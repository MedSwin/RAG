"""LIT-only TREC run exporter. Not /chat. No pack, gate, EMR ingest, or fusion."""

from __future__ import annotations

import logging
from typing import Iterable

from app.core.config import settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import LexicalRetriever
from app.schemas.enums import SourceType
from app.services.adapters.embedding import EmbeddingClient
from app.services.adapters.reranker import RerankerClient

from .contract import CASCADE_RERANK, RRF_K, RRF_K_BM25, RRF_K_DENSE, RUN_DEPTH
from .ranking import bare_pmcid, cascade_fill, max_pool_pmcids, rrf_combine
from .topics import CdsTopic, TopicField

logger = logging.getLogger(__name__)

LIT_CONSTRAINTS = {
    "source_policy": "LIT_ONLY",
    "disable_patient_retrieval_context": True,
}


class LitOnlyExporter:
    def __init__(self) -> None:
        self.dense = DenseRetriever()
        self.lexical = LexicalRetriever()
        self.embedder = EmbeddingClient(
            settings.EMBEDDING_URL,
            model_path=settings.EMBEDDING_MODEL_PATH,
        )
        self.reranker = RerankerClient(settings.RERANKER_URL)

    async def close(self) -> None:
        await self.embedder.client.aclose()
        await self.reranker.client.aclose()

    async def _embed_query(self, query: str):
        vectors = await self.embedder.embed([query], input_type="query")
        if not vectors:
            raise RuntimeError("Query embedding returned no vector")
        return vectors[0]

    async def retrieve_system(
        self,
        topic: CdsTopic,
        system: str,
        *,
        org_id: str,
        field: TopicField = "note",
    ) -> list[tuple[str, float]]:
        query = topic.t1_query(field)
        if system == "bm25":
            chunks = await self.lexical.retrieve(
                query, org_id, RRF_K_BM25, SourceType.LIT, None, LIT_CONSTRAINTS
            )
            return max_pool_pmcids((chunk.doc_id, chunk.lexical_score or 0.0) for chunk in chunks)
        if system == "dense":
            embedding = await self._embed_query(query)
            chunks = await self.dense.retrieve(
                embedding, org_id, RRF_K_DENSE, SourceType.LIT, None, LIT_CONSTRAINTS
            )
            return max_pool_pmcids((chunk.doc_id, chunk.dense_score or 0.0) for chunk in chunks)
        if system == "rrf":
            return await self._rrf(query, org_id)
        if system == "cascade":
            return await self._cascade(query, org_id)
        raise ValueError(f"Unknown T1 system: {system}")

    async def _rrf(self, query: str, org_id: str) -> list[tuple[str, float]]:
        embedding = await self._embed_query(query)
        lexical = await self.lexical.retrieve(
            query, org_id, RRF_K_BM25, SourceType.LIT, None, LIT_CONSTRAINTS
        )
        dense = await self.dense.retrieve(
            embedding, org_id, RRF_K_DENSE, SourceType.LIT, None, LIT_CONSTRAINTS
        )
        lex_docs = [doc_id for doc_id, _score in max_pool_pmcids(
            (chunk.doc_id, chunk.lexical_score or 0.0) for chunk in lexical
        )]
        dense_docs = [doc_id for doc_id, _score in max_pool_pmcids(
            (chunk.doc_id, chunk.dense_score or 0.0) for chunk in dense
        )]
        return rrf_combine([lex_docs, dense_docs], k=RRF_K, depth=RUN_DEPTH)

    async def _cascade(self, query: str, org_id: str) -> list[tuple[str, float]]:
        embedding = await self._embed_query(query)
        lexical = await self.lexical.retrieve(
            query, org_id, RRF_K_BM25, SourceType.LIT, None, LIT_CONSTRAINTS
        )
        dense = await self.dense.retrieve(
            embedding, org_id, RRF_K_DENSE, SourceType.LIT, None, LIT_CONSTRAINTS
        )
        lex_docs = [doc_id for doc_id, _score in max_pool_pmcids(
            (chunk.doc_id, chunk.lexical_score or 0.0) for chunk in lexical
        )]
        dense_docs = [doc_id for doc_id, _score in max_pool_pmcids(
            (chunk.doc_id, chunk.dense_score or 0.0) for chunk in dense
        )]
        rrf = rrf_combine([lex_docs, dense_docs], k=RRF_K, depth=RUN_DEPTH)
        best_chunk: dict[str, object] = {}
        for chunk in [*lexical, *dense]:
            pmcid = bare_pmcid(chunk.doc_id)
            if pmcid is None:
                continue
            score = max(chunk.lexical_score or 0.0, chunk.dense_score or 0.0)
            current = best_chunk.get(pmcid)
            if current is None or score > current[0]:
                best_chunk[pmcid] = (score, chunk)
        head_ids = [doc_id for doc_id, _score in rrf[:CASCADE_RERANK]]
        passages = []
        kept: list[str] = []
        for doc_id in head_ids:
            item = best_chunk.get(doc_id)
            if item is None:
                continue
            kept.append(doc_id)
            passages.append(item[1].text)
        if not passages:
            return rrf
        raw = await self.reranker.rerank(query, passages, return_logits=True)
        reranked: list[tuple[str, float]] = []
        for result in raw:
            index = int(result.get("index", -1))
            if 0 <= index < len(kept):
                reranked.append((kept[index], float(result.get("score") or result.get("p_hat") or 0.0)))
        reranked.sort(key=lambda item: -item[1])
        return cascade_fill(reranked, rrf, CASCADE_RERANK, RUN_DEPTH)


def write_run(
    path,
    run_name: str,
    rankings: Iterable[tuple[int, list[tuple[str, float]]]],
) -> None:
    lines: list[str] = []
    for qid, ranked in sorted(rankings, key=lambda item: item[0]):
        if not ranked:
            continue
        max_score = ranked[0][1]
        min_score = ranked[-1][1]
        span = max(max_score - min_score, 1e-9)
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            # Strictly non-increasing official scores: use rank as a tiny tie-break.
            adjusted = (score - min_score) / span + (len(ranked) - rank) * 1e-9
            lines.append(f"{qid} Q0 {doc_id} {rank} {adjusted:.9f} {run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
