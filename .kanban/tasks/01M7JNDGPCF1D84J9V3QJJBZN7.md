---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7k5s160msg64js43jxwwm68
  text: 'Research: Qwen35DecodeHeadsKernel guard accepts any headDim that is a multiple of 128 and <= 4096, so 384 runs (3 groups). FlintDecodeHeadsKernel guard is `headDim == 128` only (one SIMD group of 4 numbers per lane; a wider head would sum in another order than MLX''s multi-SIMD-group rms_single_row), so Flint accepts no non-power-of-two width. Plan: Qwen gets a 384 parity test (RED then GREEN); Flint gets the reciprocal (bit-identical at 128, covered by existing parity tests + flintcheck) and a test that 384 is rejected (the card''s "or reject such widths").'
  timestamp: 2026-11-16T21:56:25.408796+00:00
- actor: claude-code
  id: 01m7k64km0k6k7f8t2d0hyknvg
  text: 'RED: new `aHeadWidthThatIsNotAPowerOfTwoGivesTheBitsOfTheOpPath` (headDim 384) fails in all 8 float32 cases (12 issues), e.g. `queries[768]: kernel -0.29497716, op path -0.29497713` — one ulp from the division. bfloat16/float16 cases pass (the rounding to T hides it). Flint `aHeadWidthThatIsNotAPowerOfTwoHasNoKernel` passes before the change: the Flint guard already rejects 384, so Flint has no reachable non-power-of-two width to show RED on.'
  timestamp: 2026-11-16T22:02:44.736634+00:00
- actor: claude-code
  id: 01m7k6sxsh0hrtzxc4qrkv71p2
  text: 'GREEN: both kernels now compute `rsqrt(sum * (1.0f / DIM) + eps)`, as ResidualRMSNormKernel does, and the header comments describe the fast-math reciprocal. The Qwen 384 parity test passes in all 24 cases, and the existing 256 parity tests (Qwen 24, Flint 48) and layer tests stay bit-identical. All five bundles exit 0: MLXLMTests 1128 XCTest (0 failures; 4 env-gated LFM2 skips, the same before the change) + 2371 Swift Testing passed (3 known issues, the same as before); MLXGuidedGenerationTests 78; MLXFoundationModelsTests 496; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. qwencheck vs qwen_golden_mn44v3n.json: 8/8 MATCH (short/code/32k/toolchat × plain/mtp), SESSION PASS, both TOOLCHAT PASS, CHECK PASS. flintcheck vs golden_base.json: 4/4 MATCH, all 256 ids identical, tool call get_weather(Paris) emitted.'
  timestamp: 2026-11-16T22:14:23.281247+00:00
- actor: claude-code
  id: 01m7k6t0ca562hwe78m7yg6rfh
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXVLM/Models/Qwen35DecodeHeadsKernel.swift, Libraries/MLXLLM/Models/FlintDecodeHeadsKernel.swift, Tests/MLXLMTests/Qwen35DecodeHeadsKernelTests.swift, Tests/MLXLMTests/FlintDecodeHeadsKernelTests.swift; 5 bundles green; qwencheck CHECK PASS; flintcheck 4/4 MATCH
    - next: /review. Flint's guard (headDim == 128) rejects 384, so its 384 test is a rejection test that also passed before the change. No Flint parity case at a non-power-of-two width can exist.
  timestamp: 2026-11-16T22:14:25.930693+00:00
position_column: doing
position_ordinal: '80'
title: 'Decode-heads kernels: divide by the norm width the way MLX fast math does'
---
## Why
Found on ^ftj9mae. SwiftPM builds the MLX `rms_*` kernels with fast math, which turns `acc / axis_size` into `acc * (1 / axis_size)`. For a width that is not a power of two this rounds differently from a true division: a probe of 256 float32 rows found 50-65 rows whose inverse differed by one ulp at widths 72, 5120 and 6000 (0 rows at 2048), and `total * (1.0f / WIDTH)` matched every row.

`Qwen35DecodeHeadsKernel` (`simd_sum(groupSum) / DIM`) and `FlintDecodeHeadsKernel` (`squares / DIM`) divide. They are bit-exact today only because their head widths are powers of two (256). A head width such as 384, which both init guards accept, would not be bit-exact.

## What
Write `* (1.0f / DIM)` in both kernels, as `ResidualRMSNormKernel` does. Add a parity case with a head width that is not a power of two (if the kernels accept it), or narrow the init guards to powers of two.

## Acceptance
- [ ] Both kernels multiply by the reciprocal, or reject such widths.
- [ ] A parity case covers it. All five bundles are green. #qwen-perf