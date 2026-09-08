# Runtime architecture

The v2 runtime has one typed `PipelineId`: `naive`, `two-stage`, or `full`. Naive is a separate dense-only orchestrator. Two-stage and full share the policy-driven full orchestrator; only `enable_mac` differs. Free-form MAC/gate ablation switches are not accepted by the public API. The selected generator owns query normalization, enabled MAC calls, benchmark answer generation, and final synthesis.
