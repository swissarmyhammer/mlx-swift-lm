---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m505s8v383kx3kxvnk9zjdv2
  text: |-
    Research and micro-benchmark (scratchpad/qwen38bench mode `pattn`, prototype scratchpad/qwen38bench/Sources/qwen38bench/PattnProto.swift; one call = q (1,24,2048,256), k/v (1,4,32768,256) bf16, causal; 16 layers, idle GPU, sourcekit-lsp ~90 % CPU as noise):
    - MLX SDPA (unfused for D 256 on M3): 88-92 ms/layer.
    - First kernel (one threadgroup = 6 query heads x 8 tokens of one KV head, two SIMD groups split the head width and add their partial scores in threadgroup memory, online softmax in float32, K/V tiles of 16 keys in threadgroup memory): 94-97 ms/layer. Slower.
    - What did NOT help: K stored transposed in threadgroup memory (bank conflicts, 109 ms); scores as K*Q^T with a shuffle transpose (no change); more accumulator chains (slower); 2 token fragments per SIMD group (register pressure, 158 ms); 4-way head-width split (113-122 ms); K/V frags read straight from device memory (250-380 ms); register prefetch of the next tile (worse: +9 to +20 ms, register pressure); one shared K/V buffer with 32 keys (158-188 ms); skip of the output rescale with simd_all (115 ms).
    - What helped: (1) all tile loads of a thread first, then all threadgroup stores (the old loop waited for each load). (2) Extra "loader" SIMD groups that load the next K/V tile into the second value buffer while the compute SIMD groups work (two barriers for each tile). (3) Three query heads for each threadgroup (2 threadgroups for each KV head and token block) instead of six. (4) Probabilities in the input type for the P*V product (the same rounding as the MLX unfused path).
    - Best: 3 heads x 8 tokens per threadgroup, split 2, 16 keys per tile, 4 loader SIMD groups: 83.5 ms/layer at 2048 x 32768 (MLX 92), 2.9 vs 5.5 ms at 2048 x 2048, 18.7 vs 21.6 at 2048 x 8192, 40.3 vs 43.2 at 2048 x 16384, 21.7 vs 23.0 at 512 x 32768, equal at 128 x 32768, 3.4 vs 4.1 at 64 x 32768. float32 needs 8 keys per tile (threadgroup memory limit 32 KB).
    - Max |diff| vs the float32 reference: 1.7e-4 (bf16), 2e-5 (f16), 5e-7 (f32); MLX bf16 SDPA: 4e-4.
    - Expected gain over a 32k prefill: about 1 s (attention is about 15 % of prefill), plus no 24 x 2048 x 32k score tensor (about 3 GB transient memory less).
  timestamp: 2026-10-15T16:19:35.395211+00:00
- actor: claude-code
  id: 01m5094p2537dewb8qgabq9z2p
  text: |-
    Implementation and measurement (not committed).

    Code:
    - New `Libraries/MLXLMCommon/PrefillAttentionKernel.swift`: one Metal flash-attention kernel for prefill (head width 256, query heads of one KV head in groups of 3, causal or no mask, online softmax in float32, 8x8 SIMD-group matrices, 4 loader SIMD groups, two value tiles). The tuned prototype configuration: 3 heads x 8 tokens per threadgroup, 2 SIMD groups for each head, key tiles of 16 keys (8 for float32). Keys with spaced head numbers take narrow loads. Route: `PrefillAttentionKernel.scaledDotProductAttention` runs the kernel for 64 or more queries, else it gives the call to `DecodeAttentionKernel.scaledDotProductAttention` (decode is not changed).
    - `Qwen35Attention.attend` calls the new route.
    - New `Libraries/MLXLMCommon/MetalMatrixLane.swift`: the lane layout of an 8x8 SIMD-group matrix (`lane_coordinate` and its constants), shared with `DeepSeekV4SparseAttentionKernel` (that header had the same lines).
    - Tests: `PrefillAttentionKernelTests` (tolerance against the SDPA of MLX in float32 and bfloat16, causal and no mask, 3 shapes; spaced keys; positive controls: wrong mask and missing token break the tolerance; canRun limits; route bit for bit), `Qwen35PrefillAttentionTests` (the attention body gives the kernel output bit for bit). Shared helpers: `AttentionKernelTolerance` (the decode test uses it now), `Qwen35AttentionLayerFixture` (the decode route test uses it now).
    - TDD: the Qwen35 route test failed before the route and passed after it. A mutant kernel (causal limit + 1) failed 7 tolerance checks.

    Measurement (Qwen3.8-27B mxfp4, 32k, idle GPU at each start; background CPU: fseventsd about 105 %, two sourcekit-lsp about 88 % and 77 %, SourceKitService 27-94 %). Baseline = same tree with the old route.

    | | base 1 | base 2 | new 1 | new 2 |
    |---|---|---|---|---|
    | profile prefill total | 82.27 s | 82.39 s | 82.14 s | 82.69 s |
    | profile block 0 | 4.530 s | 4.531 s | 4.736 s | 4.635 s |
    | profile blocks 8-15 (16k-32k) | 43.85 s | 43.91 s | 43.70 s | 43.74 s |
    | profile last block | 5.813 s | 5.827 s | 5.777 s | 5.783 s |
    | profile peak after prefill | 20.9 GiB | 20.9 GiB | 18.1 GiB | 18.1 GiB |
    | profile decode B (pipelined) | 29.33 ms | 29.45 ms | 29.34 ms | 29.40 ms |
    | full prefill | 81.40 s (395.6 tok/s) | 81.56 s (394.8 tok/s) | 87.45 s (368.2 tok/s, CPU noise) | 81.16 s (396.7 tok/s) |
    | full peak | 20.7 GiB | 20.7 GiB | 17.9 GiB | 17.9 GiB |
    | full decode | 29.23 ms | 29.47 ms | 30.05 ms | 29.55 ms |
    | cold turn 2 peak | 23.07 GiB | 23.07 GiB | 20.33 GiB | 20.33 GiB |
    | HARD reuse check (thinking off, on) | PASS, PASS | PASS, PASS | PASS, PASS | PASS, PASS |

    Result:
    - Peak memory: 2.8 GiB less (20.7 to 17.9 GiB).
    - Prefill: blocks 8-15 are 0.16 s faster in both runs. Each block from 10k keys on is 0.01-0.04 s faster. Block 0 is 0.1-0.2 s slower: most probably the first compile of the new Metal kernel. The full prefill gain is about 0.2-0.4 s of 82 s. This is inside the noise of full mode, and it is much less than the 3 s target.
    - The micro-benchmark predicted about 0.11 s less at the last block; the model gives 0.04 s. The micro-benchmark sends 16 independent calls, thus the GPU can overlap them. In the model, each call waits for the layer before it.
    - Decode: no change (the same kernel).

    Checks: `swift build --build-tests` with 0 compiler warnings (only the old SwiftPM note "missing creator for mutated node"). `xcrun xctest` MLXLMTests: 716 XCTest tests and 1497 Swift Testing tests, 0 failures. xcodebuild Qwen35SessionPromptCacheTests + Qwen35CheckpointIntegrationTests: 7 tests passed. swiftlint no_magic_numbers: 0 on the new files.

    Note: only models whose query heads per KV head are a multiple of 3 take the kernel (Qwen3.8: 6). Qwen3.5-4B (4 per KV head) keeps the old route.

    Logs: scratchpad/pattn/jah-{base,new}-{profile,full}-{1,2}.{txt,load}, jah-xctest-full.log, jah-integ.log, jah-mutant.log.

    step: implement / outcome: changed / evidence: peak 20.7 to 17.9 GiB; prefill blocks 8-15 0.16 s faster in 2 of 2 runs, total gain about 0.2-0.4 s (target 3 s not met); all tests pass / task: ^jahaypg
  timestamp: 2026-10-15T17:18:15.109616+00:00
position_column: doing
position_ordinal: '80'
title: 'Qwen3.8 prefill speed: fused attention kernel for head dim 256 (no full score tensor)'
---
## What

At the last 2048 block of a 32k prefill, attention takes 1.50 s (25 % of the block), about 12 s for a 32k prefill. On the M3 Ultra (no NAX), MLX uses the unfused path for head dim 256. It writes the full score tensor (24 x 2048 x 32k in bf16, about 3.2 GB for each layer). The MLX fused kernel with `forceFused` is slower (1.59 s).

## Approach

1. Write one Metal flash-attention kernel for prefill, head dim 256, GQA 6 (24 query heads, 4 KV heads), causal, with online softmax in float32. Use simdgroup matrix tiles. The DeepSeek-V4 sparse attention kernel and the llama.cpp head-dim-256 flash-attention kernel are references.
2. Use it only where it is faster (prefill blocks); decode keeps the MLX fused kernel.
3. Target: about 3 s less for a 32k prefill, and less peak memory.

## Measure

scratchpad/qwen38bench at 32k: a micro-benchmark of one call at the last block, then full mode for prefill tok/s, peak memory, decode and the reuse check (two runs each, idle GPU).

## Acceptance Criteria

- [ ] Prefill at 32k is faster, measured.
- [ ] A tolerance test compares the kernel with MLX SDPA in float32 and bfloat16, with a positive control.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance