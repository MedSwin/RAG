# Indexing

Naive uses cloud dense retrieval. Two-stage and full use the same cloud dense+BM25 hybrid candidate pool and Cohere reranking. Every benchmark pipeline uses an isolated LIT-only corpus and retains a document-level top-20 ranking before evidence packing. Chunk-to-document aggregation uses the maximum final chunk score and stable document-ID tie breaking; packing does not change this audit ranking.
