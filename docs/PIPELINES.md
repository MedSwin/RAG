# Pipelines

`naive`: raw query → cloud dense retrieval → normal production top-K/token packing → selected generator. It never invokes BM25, reranking, MAC, a gate, or retrieve-more loops.

`two-stage`: query normalization → cloud dense+BM25 hybrid retrieval → Cohere rerank → facet fusion/packing → sufficiency gate and retrieve-more loops → selected generator. It has no MAC agents and logs `skipped_by_pipeline`.

`full`: two-stage plus MAC specialist dispatch, claim alignment, and MAC-informed fusion/contradiction handling. Both two-stage and full use cloud `embed-v-4-0` and `Cohere-rerank-v4.0-fast` under either generator.
