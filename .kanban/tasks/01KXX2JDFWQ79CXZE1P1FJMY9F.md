---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: 'Integration test: real end-to-end prompt-cache reuse on hybrid Qwen3.6 (mlx-community/Qwen3.6-27B-mxfp4)'
---
## What

The hybrid Mamba/attention prompt-cache work (task ^r9rf5g7, commits `7bb20a8`/`a1c1385`/`7c0522f`) is proven at the unit level with real `Qwen35Model`/`Qwen3NextModel` architectures but only synthetic random weights (`Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift`). There is NO end-to-end proof that a real hybrid Qwen3.6 session actually reports cache reuse through the full `respond()` → `commitPromptCache` → `store()` → `resolve()` pipeline — the pipeline the pure-attention `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseTests.swift` already covers for `mlx-community/Qwen2.5-3B-Instruct-4bit`.

Create `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift`, mirroring `PromptCacheReuseTests.swift`'s two-round shape exactly (same suite traits: `.serialized`, `.timeLimit`, macOS 27 `.enabled(if:)` gate, `#if FoundationModelsIntegration`, `@available` on the test function; same helpers: `makeTestModel`/`makeMLXExecutor`/`makeExecutorRequest`/`respondCollectingTextAndUsage` from `Support/FMTestHelpers.swift`, `releaseAllGPUMemory()` at the end), but against **`mlx-community/Qwen3.6-27B-mxfp4`** — a real hybrid Mamba/attention model (`model_type: qwen3_5` → `Qwen35Model` per `Libraries/MLXLLM/LLMModelFactory.swift`). Add the model id as a new fixture constant in `TestFixtures` (e.g. `static let qwen36HybridModelID = "mlx-community/Qwen3.6-27B-mxfp4"`) in `Support/FMTestHelpers.swift`. The model is already present in the local HuggingFace cache (`models--mlx-community--Qwen3.6-27B-mxfp4`); the machine has 512 GB unified memory. Use a generous `.timeLimit` (the 27B model is much bigger than the 3B the existing test loads).

Also assert the capability signal inside the test: `MLXLanguageModel.supportsPromptCacheReuse(model:parameters:)` must be `true` for the loaded container's model.

### Critical nuance — do not weaken the assertion to make it pass

Hybrid checkpoints cannot be trimmed (`MambaCache.isTrimmable == false`), so a round whose cache lands in `PromptCache.reconcileCacheAdvance`'s `.trimCacheByOne` case is DROPPED entirely (not trimmed-and-stored like pure attention) — see `PromptCache.cacheAdvanceOffset`'s "KNOWN, ACCEPTABLE DEGRADATION" doc in `Libraries/MLXFoundationModels/PromptCacheChunks.swift` and `commitPromptCache` in `MLXLanguageModel.swift`. Mirror the existing test's magnitude-bounded assertion (`second.cachedTokenCount >= sharedPrefixTokens - 1` where `sharedPrefixTokens = first.promptTokenCount + first.outputTokenCount`). If the test FAILS because real EOS-terminated Qwen3.6 rounds systematically hit the trim path and nothing is ever stored (cachedTokenCount stays 0), that is exactly the real-world gap this test exists to expose: STOP and report the finding to the user — do NOT weaken the assertion, mark the model unsupported, or scope the failure away (user has explicitly forbidden unilateral out-of-scope resolutions).

## Acceptance Criteria

- [ ] `PromptCacheHybridReuseTests.swift` exists in `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/` and compiles as part of the `IntegrationTesting` scheme
- [ ] `TestFixtures` (in `Support/FMTestHelpers.swift`) gains the `mlx-community/Qwen3.6-27B-mxfp4` model-id constant; no other suite's fixture usage changes
- [ ] The test asserts `MLXLanguageModel.supportsPromptCacheReuse` is `true` for the loaded Qwen3.6 model
- [ ] Round 2's `cachedTokenCount` is asserted magnitude-bounded against round 1's full stored prefix (`>= first.promptTokenCount + first.outputTokenCount - 1`), not merely `> 0`
- [ ] The test passes against the real model — or, if it fails on the hybrid EOS-trim degradation, the failure is reported to the user verbatim as a product finding (no assertion weakening, no scope-out)

## Tests

- [ ] New: `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheHybridReuseTests.swift` — two-round `respond()` reuse test against `mlx-community/Qwen3.6-27B-mxfp4`
- [ ] Run: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheHybridReuseTests` → passes
- [ ] Regression: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheReuseTests` → still passes (pure-attention path unaffected)
- [ ] Regression: `swift test --filter MLXFoundationModelsTests` → still green (253 tests as of `7c0522f`)

## Workflow

- Use `/tdd` — write the failing test first (it fails until the real model round-trips reuse), then confirm it passes against the real model.