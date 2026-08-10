---
assignees:
- claude-code
depends_on:
- 01KZGMXEJ4A72EE95T2MJRZKGM
position_column: todo
position_ordinal: 8c80
title: 'Document deferred DeepSeek-V4 scope: DSpark, JANGTQ/mxtq, V4-Pro, fused mxfp4 kernels'
---
## What

Record what the DSV4 port deliberately does **not** cover, so the gaps are discoverable rather than rediscovered. Write it to `docs/deepseek-v4-support.md` alongside a short statement of what *is* supported.

Deferred, with the reason each was cut:

1. **DSpark speculative decode** — the reference `scouzi1966/mlx-swift-lm` `Libraries/MLXLLM/Models/DeepseekV4.swift` carries `DeepseekV4DSparkMarkovHead` (line 757), `DeepseekV4DSparkConfidenceHead` (777), `DeepseekV4DSparkProposal` (792), `DeepseekV4DSparkStage` (800), and `DeepseekV4DSparkGenerator` (2355). We decode the `dspark_*` config keys but implement none of the behavior. This repo already has its own speculative-decoding machinery (`Libraries/MLXLMCommon/SpeculativeDecoding.swift`, `MTPSpeculativeTokenIterator.swift`) — any future DSpark work should be evaluated against that rather than ported wholesale.
2. **Activation quantization (`DeepseekV4ActivationQuant`)** — the reference ships `Libraries/MLXLMCommon/DeepseekV4ActivationQuant.swift` (153 lines): e4m3 activation round-trip and symmetric-Q8 matvec. `osaurus-ai/osaurus` carries a `deepseekV4ActivationQAT` load flag, which suggests it is an opt-in accuracy/throughput path rather than a load requirement. **Verify** whether `mlx-community/DeepSeek-V4-Flash-4bit` actually needs it — if the integration test in `e7b24ws` passes without it, record that as evidence and keep it deferred; if not, file a task.
3. **JANGTQ / `mxtq` quantization variants** — `osaurus-ai/vmlx-swift-lm` ships `DeepseekV4JANGTQ.swift` and `DeepseekV3JANGTQ.swift` for `weight_format == "mxtq"` (TurboQuant codebook) bundles. We do not support that format at all; the `mxtq_bits` config keys are skipped.
4. **DeepSeek-V4-Pro** (`mlx-community/DeepSeek-V4-Pro`, 1.6T total / 49B active) — same `model_type`, so it may load, but it is untested and the memory footprint is impractical here. State that it is unvalidated rather than implying support.
5. **Fused mxfp4 Metal kernels** — `deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and `deepseek_v4_native_mxfp4_down_sum6` from the reference's `SwitchLayers.swift`, a throughput optimization we skipped in favor of the generic `gatherQuantizedMM` path (see task `wkv5j6f`). Note the env knobs the reference uses (`VMLX_DSV4_NATIVE_MXFP4`, `VMLX_DSV4_MXFP4_ROWS_PER_SIMD`, `VMLX_DSV4_MXFP4_SIMD_GROUPS`) so a future performance task has a starting point.
6. **`deepseek_v32`** — we still have no port of it, even though upstream `mlx-lm` does (`mlx_lm/models/deepseek_v32.py`). Worth noting because it is the closest relative of DSV4's sparse attention.
7. **Application-layer pieces from `scouzi1966/maclocal-api`** — `Sources/AFMKitMLX/Models/DeepseekV4CheckpointConverter.swift`, `Sources/AFMKitDwarfStar/AFMDwarfStarCheckpoint.swift`, `Sources/AFMKitMLX/AFMMLXRuntimeAdapter.swift`, `Sources/AFMKitMLX/AFMMLXMetalSchedulingPolicy.swift`. These are server/CLI concerns in a downstream app, not library concerns — out of scope by design, but named here because their checkpoint-conversion and Metal-scheduling logic is the best available reference if we ever need it.

Also record the provenance chain once, in one place: `osaurus-ai/vmlx-swift-lm` (origin, MIT, Osaurus AI copyright) then `scouzi1966/mlx-swift-lm` (MIT) then us; plus `scouzi1966/maclocal-api` (MIT) as the integration reference. Neither upstream is a GitHub fork (`fork=false`, `parent=none` on both), so there is no shared git history and every port was a manual transcription.

## Provenance
- All paths and line numbers above were read from `scouzi1966/mlx-swift-lm` @ `main`, `scouzi1966/maclocal-api`, and `osaurus-ai/vmlx-swift-lm` during planning research.

## Acceptance Criteria

- [ ] `docs/deepseek-v4-support.md` exists, stating what is supported (short-prompt DSV4-Flash inference) and enumerating all seven deferred items with reasons.
- [ ] Item 2 records the verification outcome for activation quantization — either "not needed, evidence: integration test passes" or a filed follow-up task id.
- [ ] The provenance chain and MIT licensing of all three reference repos is recorded.
- [ ] The doc links the kanban short ids of the tasks that would pick each deferred item up, or says "unfiled".
- [ ] No source-code changes.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4DocsTests.swift`: asserts `docs/deepseek-v4-support.md` exists and contains the strings `DSpark`, `ActivationQuant`, `mxtq`, `DeepSeek-V4-Pro`, `mxfp4`, `deepseek_v32`, and `maclocal-api` — a cheap guard against the doc being gutted later.
- [ ] Test: asserts the doc mentions `MIT` and all three reference repo names.
- [ ] Run: `swift test --filter DeepseekV4DocsTests` — all pass.

## Workflow
- Documentation task; the tests are content guards, not behavior tests.
#deepseek-v4