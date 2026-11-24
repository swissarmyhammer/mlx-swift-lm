---
assignees:
- claude-code
depends_on: []
position_column: todo
position_ordinal: '8380'
title: Review + docs for tensor-parallel sharding (Qwen35, Inkling2)
---
Make the sharding work mergeable into `stable`.

- [ ] `/review` the cluster commits (de0b6969..HEAD) and fix the findings.
- [ ] `scripts/verify-docs.sh` passes. `TensorParallelShardable`, `TensorParallelShardingError` and the `TensorParallel` helpers are new public API in MLXLMCommon.
- [ ] DocC article or section, plus `skills/mlx-swift-lm/references/` (and the supported-models list) covering which families can shard, and how.
- [ ] `swift-format format --in-place --configuration .swift-format --recursive .` per CLAUDE.md, using CI's pinned version (see .github/workflows/pull_request.yml).
- [ ] All 5 test bundles green (MLXLMTests, MLXGuidedGenerationTests, MLXFoundationModelsTests, CXGrammarTests, MLXHuggingFaceMacrosTests), via `swift build --build-tests` + `xcrun xctest` per CLAUDE.md. Two known parallel-load flakes pass when run alone: "a chunked MTP prefill of a long prompt peaks lower…" and "a fork of a sliding-window model reuses the instructions past a wrapped window". Fix or stabilise them if they're real.
- [ ] Move pkdw8fc and e8smzkz to done.

Depends on e8smzkz. #cluster