---
assignees:
- claude-code
depends_on:
- 01KZGMPECN4FA7T3BFX6F6QMF7
position_column: todo
position_ordinal: '8380'
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

- [ ] A test proves the real DSV4-Flash-4bit `config.json` quantization block decodes to a `PerLayerQuantization` with at least one `.mxfp4` layer and a `(64, 4, .affine)` default.
- [ ] A test proves `QuantizedSwitchLinear(mode: .mxfp4)` returns finite output of the expected shape with `biases == nil`.
- [ ] If any gap is found, it is fixed in the smallest possible diff and the fix is covered by a test.
- [ ] `Libraries/MLXLMCommon/SwitchLayers.swift` is either unmodified or changed by fewer than ~40 lines; no Metal kernel source added.
- [ ] Existing SwitchLayers/quantization tests still pass (no regression to other models).
- [ ] A note is recorded (in the task or a follow-up) stating that fused mxfp4 kernels were deliberately deferred.

## Tests

- [ ] New `Tests/MLXLMTests/DeepseekV4QuantizationPlanTests.swift`.
- [ ] Test: decode the checked-in DSV4 `config.json` fixture; assert the resolved per-layer modes as described.
- [ ] Test: construct a small `SwitchLinear` (e.g. 4 experts, 32x64), call `toQuantized(groupSize: 32, bits: 4, mode: .mxfp4)`, run it, assert output shape and `allFinite`.
- [ ] Test: same construction with `.affine` still works — guards against a regression in the shared path.
- [ ] Run: `swift test --filter DeepseekV4QuantizationPlanTests` — all pass.
- [ ] Run the pre-existing quantization suite to prove no regression: `swift test --filter Quantiz` — all pass.

## Workflow
- Use `/tdd` — write the plan-resolution and mxfp4 forward tests first; they may pass immediately, which is itself the finding. Only write code for gaps the tests actually expose.
#deepseek-v4