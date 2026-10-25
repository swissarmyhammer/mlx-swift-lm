---
assignees:
- claude-code
depends_on:
- 01M5SKBCEWG4D25SZZNGG7BVM7
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8780'
title: 'MTP: multi-row quantized matvec that reads the weights one time for 2–8 verify rows'
---
## What
Measured on M5 Max (scratchpad rows.py, Qwen3.8-27B 4-bit g64 shapes, the matmuls of one step): M=1 25.0 ms, M=2 27.5 ms (1.10×), M=3 51.3 ms (2.05×), M=4 48.5 ms (1.94×). For M ≥ 3, MLX reads the weights again for each row, so a verify of 3–4 tokens costs about two plain steps. Deeper drafts are not useful until this is fixed. MLX issue 3661 reports the same problem (closed without a fix).

### References
- llama.cpp `kernel_mul_mv_ext_q4_f32_impl` (ggml/src/ggml-metal/kernels/mul_mv.metal ~603; dispatch in ggml-metal-ops.cpp ~2625). It is used for 2 ≤ M ≤ 8 for q4_0, q4_1, q8_0 and mxfp4. Layout: 2 SIMD groups for each threadgroup; `nxpsg` = 16 threads along a weight row when K % 256 == 0 and M < 3, else 8; each SIMD group does 32/nxpsg weight rows; each thread dequantizes 4 float4 chunks one time into registers and does a dot product with `r1ptg` activation rows (r1ptg = 2 for M=2, 3 for M=3 or 6, 4 for M=4, 7 or 8, 5 for M=5). The grid is (ceil(N / rows per TG), ceil(M / r1ptg)). With this kernel the cost for each row stays almost flat up to M = 4–8.
- osaurus vmlx-swift PR 613 `RowExactAffineQMV`, a copy of MLX `qmv_fast_impl` where all rows share one weight stream.

- [ ] Add Libraries/MLXLMCommon/MultiRowQuantizedMatvec.swift: an `MLXFast.metalKernel` for M ≤ 8 rows (r1ptg 2–5 as in llama.cpp), affine (bits 4 and 8, group 32/64/128, with biases) and mxfp4, bf16/f16 input. The output of each row must be bit-identical to `quantizedMM` of that row alone (same reduction order as `qmv_fast`). If bit-identical output costs too much speed, measure both and record the result.
- [ ] Add a dispatch helper `quantizedMatmulRows(_ x:, _ layer: QuantizedLinear)`. It uses the kernel when 3 ≤ M ≤ 8 (also M = 2 if the measurement shows a gain), and otherwise `QuantizedLinear.callAsFunction`. M = 1 always keeps the MLX kernel.
- [ ] Use the helper in the Qwen35 verify path: the fused GDN projection, out_proj, the MLP, the attention projections, and lm_head (Libraries/MLXLLM/Models/Qwen35.swift; Libraries/MLXLLM/Models/Qwen3Next.swift `Qwen3NextMLP`, which other models share).

## Acceptance Criteria
- [ ] A microbenchmark of all Qwen3.8 shapes: M=4 costs ≤ 1.3× M=1, and M=6 costs ≤ 1.6× M=1 (an XCTest in Release, or a Python harness that runs the same kernel source).
- [ ] Bit-identical row outputs against `quantizedMM` for every supported mode, bits, and group size.
- [ ] Plain decode (M=1) of Qwen3.8 and of a Qwen3Next model is unchanged: the same logits, and tok/s within noise.

## Tests
- [ ] Tests/MLXLMTests/MultiRowQuantizedMatvecTests.swift: bitwise equality for M = 1..8, all modes, odd output sizes (48, 96, 248320).
- [ ] Tests/MLXLMTests: a Qwen3Next fixture test where plain-decode logits are bit-identical before and after the `Qwen3NextMLP` change.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #perf