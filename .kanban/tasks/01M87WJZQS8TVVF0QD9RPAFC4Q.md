---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'ClusterLanguageModel seam: remote forward and cache ops'
---
Let a `ModelContainer` hold a model whose forward pass and KV/recurrent cache operations run in a remote worker process (the rank-0 worker in FoundationModelsCluster). Everything around them stays in-process and unchanged: sampling, logit processors, grammar, stop strings, tool and reasoning parsing, and prompt-cache policy in `MLXLanguageModel`. Design: `cluster.md` in swissarmyhammer/FoundationModelsCluster (Architecture § "The app drives; workers replay").

- [ ] Define a transport-agnostic protocol (e.g. `RemoteModelTransport`) with ops: begin/end request, forward(tokens, cache handle) → last-position logits, newCache, trim, copy, reset. No networking in this repo; FoundationModelsCluster implements the transport (SSH stdio + shared-memory logits).
- [ ] `ClusterLanguageModel`: a `LanguageModel` whose `callAsFunction`/`prepare` and `newCache` go through the transport. Its caches are handles (a `KVCache` type the cache code can trim/copy/reset through ops).
- [ ] Audit: every cache operation must go through the model. Check `PromptCacheReusePolicy`, `ChatSession`, `KVCacheTree`, prompt-cache snapshots and spooling, `MLXLanguageModel` warmup/guided/tool phases (MLXLanguageModel.swift ~647-655, 987, 1757, 2116). Anything that reads cache tensors directly must be disabled for clustered models or routed through an op.
- [ ] v1 limits, each rejected with a clear error: VLM inputs (workers need embeddings), MTP/speculative decoding, prompt-cache snapshots to disk, CompiledDecodeSegments (compile off). Prefill chunked to ≤ 1024 tokens per step.
- [ ] Tests: an in-process fake transport that wraps a real local model. A clustered container must produce identical tokens to the local one across generation, a tool-call turn, a guided-generation turn and prompt-cache reuse/trim.

Depends on nothing in this repo. FoundationModelsCluster's worker consumes the same op set, so agree on the op enum with that session (foundationmodelscluster-02). #cluster