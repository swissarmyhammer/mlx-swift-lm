---
assignees:
- claude-code
depends_on:
- 01KXY0Z94XT2HF9RPM3XGVTH41
position_column: todo
position_ordinal: 8b80
title: 'MiniMax-M3: text-only processor, factory registration, and real-weights coherence test'
---
## What

Make `mlx-community/MiniMax-M3-4bit` actually loadable and generating text end-to-end:

1. Register `"minimax_m3_vl"` → `MiniMaxM3Configuration`/model init in `Libraries/MLXVLM/VLMModelFactory.swift`'s type registry.
2. **Text-only processor**: port the text path of mlx-vlm's `processing_minimax_m3_vl.py` as the model's `UserInputProcessor` registration in `VLMProcessorTypeRegistry` (check the repo's `preprocessor_config.json`/`processor_config.json` for the processor type string). Image/video input throws a clear "not yet supported" error until the vision task ^(vision) lands — do NOT silently ignore images.
3. Add a `VLMRegistry` entry (e.g. `minimaxM34bit` for `mlx-community/MiniMax-M3-4bit` — follow the acronym-casing conventions established in task ^9mv1q33's rework) and update `skills/mlx-swift-lm/references/supported-models.md` + `Libraries/MLXVLM/README.md`.
4. Quantization: the checkpoint quantizes MoE gates at 8-bit group-64 while the rest is 4-bit — verify the existing per-layer `quantization` config handling in the factory covers this heterogeneous layout (it should, via config-driven per-module quant predicates); fix up if not.

Weights are ~120 GB (not yet in the local HF cache) — the machine has 512 GB unified memory; the integration test downloads on first run like other gated real-weights suites.

## Acceptance Criteria

- [ ] `mlx-community/MiniMax-M3-4bit` loads through `VLMModelFactory` with zero unconsumed/missing weight keys
- [ ] A text-only prompt generates coherent text end-to-end (real weights, gated integration test — same pattern as `IntegrationTesting/IntegrationTestingTests/CoherenceIntegrationTests.swift`)
- [ ] Image input throws a descriptive unsupported error (unit test, no weights needed)
- [ ] `VLMRegistry` entry + docs updated; registry characterization tests still green

## Tests

- [ ] Extend `Tests/MLXLMTests/MiniMaxM3Tests.swift`: processor text-path unit test, image-throws test, registry entry test
- [ ] New gated integration test in `IntegrationTesting/IntegrationTestingTests/CoherenceIntegrationTests.swift` (or sibling): MiniMax-M3-4bit generates coherent text
- [ ] Run: `swift test --filter MLXLMTests` → green; `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/CoherenceIntegrationTests` → M3 case passes

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax