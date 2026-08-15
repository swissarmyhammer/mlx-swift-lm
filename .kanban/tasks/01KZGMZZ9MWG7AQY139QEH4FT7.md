---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kztjprmgj4f57352wc6cpfwz
  text: |-
    Research results:

    - Model of the content-guard test: `Tests/MLXLMTests/LicenseNoticeTests.swift`. It finds the repository root from `#filePath` and reads the document as text. The new suite copies that pattern.
    - The suite name is `DeepSeekV4DocsTests` (canonical DeepSeek spelling, the same as the renamed family). The card wrote `DeepseekV4DocsTests`; the family files are all `DeepSeekV4*.swift`.
    - Kanban references for the deferred items: item 2 is `^e7b24ws` (verification pending; the ~91 GB checkpoint is absent and the gated suite `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift` skips). Item 5 was deferred in `^wkv5j6f` (done); a performance follow-up is unfiled. Items 1, 3, 4, 6, 7 are unfiled — a task search for DSpark, mxtq, deepseek_v32 found no task.
    - Facts confirmed in the tree: `Libraries/MLXLMCommon/SpeculativeDecoding.swift`, `Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift`, `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift`, `Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift`, and the decoded-but-unused `dspark_*` keys in `Libraries/MLXLLM/Models/DeepSeekV4Configuration.swift`.
    - `THIRD-PARTY-NOTICES.md` at the repository root already holds the three MIT license texts; the new document points to it.
  timestamp: 2026-08-12T08:51:29.040157+00:00
- actor: claude-code
  id: 01kztjsyr0yrrnef01rqwb3kw8
  text: |-
    ### implement — changed
    - evidence: 2 new files — docs/deepseek-v4-support.md, Tests/MLXLMTests/DeepSeekV4DocsTests.swift. TDD sequence: the suite failed 3 of 3 before the document existed, and passes 3 of 3 after it (`swift test --filter DeepSeekV4DocsTests`). `git status` shows no change under Libraries/, thus AC5 holds.
    - next: review. The task stays in doing for the review step.
  timestamp: 2026-08-12T08:53:13.600820+00:00
- actor: claude-code
  id: 01kztk3yg45nnnk6gyd3dtcs7z
  text: |-
    ### review — clean
    - evidence: The engine ran `review sha HEAD~1..HEAD` and gave 0 findings with 0 attempted files (the engine does not read Markdown or test files). The substitute proof is on the card: the guard suite failed 3 of 3 before `docs/deepseek-v4-support.md` existed, and passes 3 of 3 after (`swift test --filter DeepSeekV4DocsTests`). The spot-check of the document found all seven deferred items with reasons, the provenance chain, and the MIT license for all three reference repositories; `THIRD-PARTY-NOTICES.md` exists at the repository root. The document does not overclaim: it says that real-weights validation is not complete and points to task `^e7b24ws`, and it says that DeepSeek-V4-Pro is not validated. The task ids `^e7b24ws`, `^wkv5j6f`, `^r92pjcr`, and `^tty95f4` in the document match tasks on the board. The diff of 3c54a9a touches only the document, the test file, and kanban files — no change under `Libraries/`, thus AC5 holds.
    - next: none. The task moves to done.
  timestamp: 2026-08-12T08:58:41.028334+00:00
- actor: claude-code
  id: 01kztk4hp74njg5r11wgq5370r
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — docs/deepseek-v4-support.md and Tests/MLXLMTests/DeepSeekV4DocsTests.swift; TDD red 3/3 then green 3/3
    - test: green — swift test, exit 0, 848 + 246 tests, 0 failures
    - commit: 3c54a9a
    - review: clean — engine 0 findings 0 attempted (Markdown/test delta); substitute TDD proof verified; reviewer spot-check found the doc complete and honest; task moved to done
  timestamp: 2026-08-12T08:59:00.679320+00:00
depends_on:
- 01KZGMXEJ4A72EE95T2MJRZKGM
position_column: done
position_ordinal: ef80
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

- [x] `docs/deepseek-v4-support.md` exists, stating what is supported (short-prompt DSV4-Flash inference) and enumerating all seven deferred items with reasons. (Done. The document states the supported functions with more precision: registry load, chat encoder, DSML parser, and reasoning modes, proven on synthetic weights; real-weights validation is not complete, task `^e7b24ws`.)
- [x] Item 2 records the verification outcome for activation quantization — either "not needed, evidence: integration test passes" or a filed follow-up task id. (The filed follow-up branch: task `^e7b24ws`. The gated suite skips because the ~91 GB checkpoint is absent, thus no test result exists yet.)
- [x] The provenance chain and MIT licensing of all three reference repos is recorded.
- [x] The doc links the kanban short ids of the tasks that would pick each deferred item up, or says "unfiled". (Item 2: `^e7b24ws`. Item 5: deferred in `^wkv5j6f`, performance follow-up unfiled. Items 1, 3, 4, 6, 7: unfiled.)
- [x] No source-code changes. (`git status` shows only the new document and the new test file.)

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4DocsTests.swift`: asserts `docs/deepseek-v4-support.md` exists and contains the strings `DSpark`, `ActivationQuant`, `mxtq`, `DeepSeek-V4-Pro`, `mxfp4`, `deepseek_v32`, and `maclocal-api` — a cheap guard against the doc being gutted later. (The file and the suite use the canonical spelling: `Tests/MLXLMTests/DeepSeekV4DocsTests.swift`, suite `DeepSeekV4DocsTests`.)
- [x] Test: asserts the doc mentions `MIT` and all three reference repo names.
- [x] Run: `swift test --filter DeepSeekV4DocsTests` — all pass. (3 of 3 pass.)

## Workflow
- Documentation task; the tests are content guards, not behavior tests.
#deepseek-v4