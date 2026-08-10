---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzp968sph15apw2exds16qh3
  text: |-
    ### Research

    Read the fixture `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json` and the plumbing. Facts:

    - The `quantization` block holds 3 scalar keys (`group_size: 64`, `bits: 4`, `mode: "affine"`) and 641 per-layer keys. No `false` (skip) entry.
    - 129 keys are `{group_size: 32, bits: 4, mode: "mxfp4"}` = 43 layers x 3 projections, all named `model.layers.N.ffn.switch_mlp.{gate,up,down}_proj`.
    - 512 keys are `{group_size: 64, bits: 4, mode: "affine"}`: `model.embed_tokens`, `lm_head`, `model.layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}`, `model.layers.N.attn.compressor.{wgate,wkv}`, `model.layers.N.attn.indexer.{wq_b,weights_proj}`, `model.layers.N.attn.indexer.compressor.{wgate,wkv}`, `model.layers.N.ffn.shared_experts.{gate,up,down}_proj`.
    - The router (`model.layers.N.ffn.gate`) is absent from the block, thus it stays in high precision.
    - `QuantizationMode` is `String, Codable` with raw values `affine`/`mxfp4`/`mxfp8`/`nvfp4`, so `mode` decodes with no new code.
    - `MLX.quantized(_:groupSize:bits:mode:)` returns `biases` as `MLXArray?` and gives `nil` for mxfp4 (the C vector holds 2 arrays, not 3). `QuantizedSwitchLinear.biases` is already `MLXArray?`, thus it stores `nil`.
    - `Load.swift` `loadWeights` gates on `weights["<path>.scales"] != nil` and then hands `perLayerQuantization.quantization(layer: path)?.asTuple` to `quantize(model:)`, which threads `mode` through.

    The path strings in the block are checkpoint key paths. The Swift module tree must give the same flattened paths (`ffn`, `switch_mlp`, `attn`, `wq_a`, ...) for the plan to resolve. This is a constraint the MoE/model task must meet; the tests here examine it with a probe module tree that uses those `@ModuleInfo` keys.
  timestamp: 2026-08-10T16:48:13.622503+00:00
- actor: claude-code
  id: 01kzp9peaa7ep0ymrt60sb5efx
  text: |-
    ### Result: the generic path already resolves the mixed plan -- no production change

    Eight new tests in `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift`. Seven passed on the first run against unchanged production code, which is the finding the card asked for. `Libraries/` holds no diff.

    **One test went red first, and it was a fixture fact, not a code gap.** The first draft examined `model.layers.0.attn.compressor.wkv`. That key is absent: layers 0 and 1 hold a compress ratio of 0 and thus carry no compressor, and layer 2 is the first that does. The block names a compressor for layers 2 to 42 (41 keys) and an indexer for the even layers 2, 4, ... 42 (21 keys), which is where `compress_ratios` holds 4. The test now names layer 2 and passes. The MoE and attention tasks must build the compressor and the indexer conditionally on the same rule.

    **No reference conflict.** The card, the fixture and the code agree on every point examined. Nothing needed adjudication against the Python.

    **Deferred: fused mxfp4 Metal kernels.** `scouzi1966/mlx-swift-lm`'s `deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and `deepseek_v4_native_mxfp4_down_sum6` are a throughput optimization, not a correctness requirement. The generic `gatherQuantizedMM` path gives correct mxfp4 output, as `mxfp4SwitchLinearMatchesItsOwnDequantizedWeights` shows. `Libraries/MLXLMCommon/SwitchLayers.swift` is unmodified (0 lines, against the ~40 line limit), and no Metal source was added. File a performance task if profiling later justifies the kernels.

    **Access level.** `QuantizedSwitchLinear` keeps `weight`, `scales` and `biases` internal, thus the test file uses `@testable import MLXLMCommon`, which is the pattern `MiniMaxM3Tests` already uses. No production access level was widened.
  timestamp: 2026-08-10T16:57:03.562907+00:00
- actor: claude-code
  id: 01kzp9pwdwpx54h0454ezh961t
  text: |-
    ### Mutation proof (the tests are not vacuous)

    The review engine leaves test files out, thus each new test was examined against a broken production file. Two mutations, each reverted after the run.

    **Mutation 1** -- `Libraries/MLXLMCommon/BaseConfiguration.swift`: `public var mode: QuantizationMode { _mode ?? .affine }` changed to `{ .affine }`, which drops the decoded `mode`.

    Result: 2 of 8 tests failed with 136 issues.
    - `fixtureGivesEveryExpertProjectionMxfp4` -- 130 issues, one for each of the 129 `resolved.mode == .mxfp4` expectations plus the count.
    - `planAppliesMxfp4ToExpertsAndAffineToEverythingElse` -- 6 issues: `layer.mode == .mxfp4` and `layer.biases == nil` on all three expert projections.

    **Mutation 2** -- `Libraries/MLXLMCommon/SwitchLayers.swift`: `MLX.quantized(other.weight, groupSize: groupSize, bits: bits, mode: mode)` changed to `mode: .affine`, which packs the weights the wrong way while the layer keeps reporting `.mxfp4`.

    Result: 3 tests failed.
    - `planAppliesMxfp4ToExpertsAndAffineToEverythingElse` -- 3 issues, `layer.biases == nil` on each expert projection.
    - `mxfp4SwitchLinearRunsWithScalesAndNoBiases` -- `layer.biases == nil` failed, then mlx aborted the run: `[gather_qmm] Scale type must be uint8 but received type float32`.
    - `mxfp4SwitchLinearMatchesItsOwnDequantizedWeights` -- reached by the same abort.

    Both mutations reverted. `git status --porcelain -- Libraries/ Tests/` then gave one line, the new untracked test file, and `git diff --stat -- Libraries/` gave nothing.

    ### Verification

    - `swift build` -- exit 0.
    - `swift test --filter DeepseekV4QuantizationPlanTests` -- 8 tests in 1 suite passed.
    - `swift test --filter Quantiz` -- 13 tests in 2 suites passed (this suite plus the pre-existing quantization tests), no regression.
    - `swift test` -- exit 0. Swift Testing: 378 + 80 + 282 + 7 tests passed. XCTest: 0 failures across every suite. The only 2 skips are the pre-existing `CompiledDecodeCorrectnessTests` skips (kanban 01KYD3ZCWTZ414Y79RSAKVQXXZ), untouched.
    - `swift format` run on the one touched file only, with the repo `.swift-format`. It made no change.
  timestamp: 2026-08-10T16:57:18.012820+00:00
- actor: claude-code
  id: 01kzp9qv8x696hp0ztx322fh05
  text: |-
    ### implement — changed
    - evidence: 1 file — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift (new, 8 tests). No production change; `git diff --stat -- Libraries/` is empty. Mutation proof: BaseConfiguration `mode` forced to `.affine` killed 2 tests with 136 issues; SwitchLayers `MLX.quantized(mode:)` forced to `.affine` killed 3 tests (one through an mlx `[gather_qmm] Scale type must be uint8` abort). Both reverted, tree clean. `swift build` exit 0; `swift test --filter DeepseekV4QuantizationPlanTests` 8/8 pass; `swift test --filter Quantiz` 13/13 pass; full `swift test` exit 0, zero failures.
    - next: /review
  timestamp: 2026-08-10T16:57:49.597527+00:00
depends_on:
- 01KZGMPECN4FA7T3BFX6F6QMF7
position_column: doing
position_ordinal: '80'
title: Verify mixed affine+mxfp4 per-layer quantization resolves for deepseek_v4
---
## What

DeepSeek-V4-Flash-4bit is quantized as **4-bit affine, group size 64, plus `mxfp4` on some FFN layers**. Establish — with tests — that our existing quantization plumbing already resolves that mixed plan, and fix whatever gaps the tests expose.

**Scope note (revised after research):** an earlier read suggested `mxfp4` was a missing loader feature. It is not. Verified facts:

- `QuantizationMode` in mlx-swift already has `.mxfp4` (and `.mxfp8`, `.nvfp4`) — `.build/checkouts/mlx-swift/Source/MLX/Ops.swift:1097`.
- `BaseConfiguration.Quantization` already decodes `mode` and exposes `asTuple = (groupSize, bits, mode)` — `Libraries/MLXLMCommon/BaseConfiguration.swift:22-54`.
- `PerLayerQuantization.quantization(layer:)` already resolves per-layer overrides and `.skip` — `Libraries/MLXLMCommon/BaseConfiguration.swift:71-118`.
- `Load.swift:64-69` already threads the resolved per-layer tuple (including `mode`) into `quantize(model:)`.
- `QuantizedSwitchLinear` already forwards `mode:` to `MLX.gatherQuantizedMM`, and its `biases` is already `MLXArray?` — correct for mxfp4, which has scales but no biases (`Libraries/MLXLMCommon/SwitchLayers.swift:379-445`).
- Our `mlx-swift` resolves to exactly `0.31.6` (`Package.resolved`), matching what `scouzi1966/maclocal-api` pins. No `Package.swift` change needed.

So the generic path should already work. What this task must prove:

1. The real `config.json` mixed plan decodes into a `PerLayerQuantization` whose mxfp4-designated FFN layers resolve to `mode == .mxfp4` while everything else resolves to `(64, 4, .affine)`.
2. A `QuantizedSwitchLinear` built with `mode: .mxfp4` produces finite, correctly-shaped output through `gatherQuantizedMM` with `biases == nil`.
3. Any layer-path mismatch between DSV4's checkpoint key naming and our `quantize(model:)` path strings is identified and handled.

**Explicitly out of scope:** the reference's custom fused Metal kernels. `scouzi1966/mlx-swift-lm`'s `Libraries/MLXLMCommon/SwitchLayers.swift` is 1025 lines vs our 444 (1105 changed diff lines), and that delta is mostly two hand-written Metal kernels — `deepseek_v4_ds4_mxfp4_gate_up_scored_swiglu` and `deepseek_v4_native_mxfp4_down_sum6` — gated behind env knobs `VMLX_DSV4_NATIVE_MXFP4`, `VMLX_DSV4_MXFP4_ROWS_PER_SIMD`, `VMLX_DSV4_MXFP4_SIMD_GROUPS`. Those are a **throughput optimization, not a correctness requirement**. The slow-but-correct generic `gatherQuantizedMM` path is the target here; file a separate performance task if profiling later justifies the kernels. Do **not** wholesale-copy their `SwitchLayers.swift` — ours has diverged and the copy would regress other models.

## Provenance
- Reference for the kernel approach we are deliberately *not* porting: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLMCommon/SwitchLayers.swift` (MIT).
- Load-side gap list: `osaurus-ai/vmlx-swift-lm` — `Libraries/MLXLLM/Models/DSV4-PORT-STATUS.md` (names FP4 e2m1fn routed-expert dequant and FP8 e4m3fn + UE8M0 128x128 dequant).

## Acceptance Criteria

- [x] A test proves the real DSV4-Flash-4bit `config.json` quantization block decodes to a `PerLayerQuantization` with at least one `.mxfp4` layer and a `(64, 4, .affine)` default.
- [x] A test proves `QuantizedSwitchLinear(mode: .mxfp4)` returns finite output of the expected shape with `biases == nil`.
- [x] If any gap is found, it is fixed in the smallest possible diff and the fix is covered by a test. (No gap was found: seven of the eight tests passed against unchanged production code. The one red test named a layer the fixture does not hold -- see the comments.)
- [x] `Libraries/MLXLMCommon/SwitchLayers.swift` is either unmodified or changed by fewer than ~40 lines; no Metal kernel source added. (Unmodified.)
- [x] Existing SwitchLayers/quantization tests still pass (no regression to other models).
- [x] A note is recorded (in the task or a follow-up) stating that fused mxfp4 kernels were deliberately deferred.

## Tests

- [x] New `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift`.
- [x] Test: decode the checked-in DSV4 `config.json` fixture; assert the resolved per-layer modes as described.
- [x] Test: construct a small `SwitchLinear` (e.g. 4 experts, 32x64), call `toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)`, run it, assert output shape and `allFinite`.
- [x] Test: same construction with `.affine` still works — guards against a regression in the shared path.
- [x] Run: `swift test --filter DeepseekV4QuantizationPlanTests` — all pass.
- [x] Run the pre-existing quantization suite to prove no regression: `swift test --filter Quantiz` — all pass.

## Workflow
- Use `/tdd` — write the plan-resolution and mxfp4 forward tests first; they may pass immediately, which is itself the finding. Only write code for gaps the tests actually expose.
#deepseek-v4