---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8crxbe86w4eccf8p5fsh62p
  text: |-
    Research (throwaway XCTest probe in MLXLMTests, since deleted):

    Cause: the tiny config's WEIGHT GENERATION, not a model bug. `Inkling2TinyModels.target` quantized every NVFP4 expert projection with `MLX.quantized(dense, mode: .nvfp4)` and NO global scale, then set every `global_scale` to 1. MLX's nvfp4 dequant with a global scale g is `dequant(no gs) * g / 2688` (2688 = 6 * 448, largest E2M1 value times largest E4M3 scale). Measured: dense absmax 1.28; dequant(gs nil) err 0.165, absmax 1.31; dequant(gs 1) absmax 0.00049. gatherQuantizedMM with gs=ones gives |y| 0.0017 vs reference 4.41. Two projections in a row -> every tiny layer's `Inkling2SparseMoE(x)` had max |out| ~1.3e-9 (float32 and bf16). So the experts vanished and zeroing down scales / global scales / router / post_mlp_convolution changed the tiny logits by exactly 0.0, while o_proj zero -> 2.06, post_attention_convolution zero -> 2.41.

    Real checkpoint (mlx-community/Inkling-2-Mini-4bit from the local HF cache, loaded on this Mac only via VLMModelFactory.load(from: dir)): its experts are NVFP4 too (down_proj scales uint8 [128,2304,72], global_scale float32 [128], values 0.15..0.74). Its convention: global_scale = the expert's absmax (expert 0 of layer 5: g 0.2304, dequant(g) absmax 0.2305 std 0.029; dequant(nil) absmax 2688). 8-token prompt, max |dlogits| vs baseline: zero layer-5 down_proj.scales -> 19.33; all down_proj.scales -> 22.87; all mlp.gate.weight -> 16.01; all post_mlp_convolution -> 24.68; cached prefill (7 tokens) 22.93 and cached decode step 17.74 with all down scales zeroed. Restore checks returned 0.0 (note: Module.update mutates the MLXArray objects that parameters() returned in place, so a restore snapshot must be copied first). => the model keeps the MoE output on every path; not-a-model-bug.

    Fix: the tiny target quantizes each expert with global scale = its absmax (the checkpoint's convention) and stores that scale. After the fix the tiny MoE output max is 22..40 per layer, and zeroing down scales changes the logits by 1.86 (fp32) / 1.89 (bf16).
  timestamp: 2026-11-26T20:31:50.728478+00:00
- actor: claude-code
  id: 01m8cs5hawtjnpr25kzkzba9rh
  text: |-
    TDD + shard-check evidence:
    - RED: new Swift Testing test `Inkling2ModelTests.expertsReachTheLogits(bfloat16:)` (Tests/MLXLMTests/Inkling2Tests.swift) zeroes every `mlp.experts.down_proj.scales` and expects max |dlogits| > 0.1 on an 8-token prompt. Before the fix both cases failed (`Expectation failed: maxAbsoluteDifference(logits, withoutExperts) > Self.expertLogitEffect`, bfloat16 false and true). GREEN after the fix in Tests/MLXLMTestSupport/Inkling2TinyModels.swift (2 cases passed).
    - `scripts/shard-check.sh --tiny inkling2 inkling2_loop` with the all-sum of `Inkling2SparseMoE.callAsFunction` dropped (`return sum`): exit 1, FAIL inkling2 teacher-forced / announce, FAIL inkling2_loop teacher-forced / announce (inkling2 teacher-forced: top-5 gate agree 5, mismatch 2, max abs logit diff up to 1.12; inkling2_loop decode 1..4 argmax MISMATCH, diff up to 2.6). build.log `error:` lines: 0.
    - Restored (Inkling2MoE.swift identical to HEAD, git diff empty): exit 0, PASS for all four checks, top-5 gate agree 9/9 mismatch 0. build.log `error:` lines: 0.
  timestamp: 2026-11-26T20:36:18.908288+00:00
- actor: claude-code
  id: 01m8csf8q2dw68rw38dvqt3jtc
  text: |-
    ### implement — changed
    - evidence: 2 files — Tests/MLXLMTestSupport/Inkling2TinyModels.swift (per-expert NVFP4 global scale = absmax, passed to MLX.quantized and stored; removed the global_scale = 1 loop), Tests/MLXLMTests/Inkling2Tests.swift (new `expertsReachTheLogits(bfloat16:)`, RED then GREEN). Throwaway probe deleted; Libraries/ unchanged. swift-format run on both. All five bundles via `xcrun xctest`: MLXLMTests 2783 Swift Testing tests passed (3 known issues, which are existing withKnownIssue tests in AttentionKernelToleranceTests/ToolTests) + 1261 XCTest 0 failures; MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 618 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 11 XCTest 0 failures. ClusterInklingParityTests passed on this run (no NaN flake). shard-check --tiny inkling2 inkling2_loop: 4 FAIL with all-sum dropped, 4 PASS restored, 0 `error:` lines in both build logs.
    - next: /review
  timestamp: 2026-11-26T20:41:37.762703+00:00
- actor: claude-code
  id: 01m8cv1vtcwm65g6em8dnq0md1
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (91d85604): 0 findings (confirmed 0, refuted 0, attempted 7, failed 0); 2 files reviewed, 4 .kanban files matched no validator
    - next: none — moved to done
  timestamp: 2026-11-26T21:09:15.724492+00:00
- actor: claude-code
  id: 01m8cv27tt3qpd9erq78jjdyrc
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Tests/MLXLMTestSupport/Inkling2TinyModels.swift, Tests/MLXLMTests/Inkling2Tests.swift (not a model bug; tiny NVFP4 global scale)
    - test: green — MLXLMTests twice (2783 ST + 1261 XCTest, 0 failures, ClusterInklingParityTests passed both), FM 618, Guided 96, CXGrammar 7, Macros 11; --tiny all 9 families PASS
    - commit: 91d85604
    - review: clean — 0 findings; task done
  timestamp: 2026-11-26T21:09:28.026006+00:00
position_column: done
position_ordinal: ffffff8380
title: 'Inkling 2 tiny target: the MoE branch does not reach the logits, so tiny shard-check cannot see an Inkling MoE sharding break'
---
Found in ^727pxfp. With `Inkling2TinyModels.target(json: Inkling2TinyModels.targetJSON)` (and the tiny `inkling2` / `inkling2_loop` checkpoints that `ShardCheck tiny-checkpoint` writes from it), the MoE output has no effect on the logits:

- In memory: zeroing every `mlp.experts.down_proj.global_scale` leaves the logits of an 8-token prompt bit-identical (with and without a cache).
- In the checkpoint: zeroing every `mlp.experts.down_proj.scales`, every `mlp.experts.gate_up_proj.scales`, or every `mlp.gate.weight` leaves the unsharded ShardCheck logits bit-identical; zeroing `self_attn.o_proj.weight` changes them by 2.4.
- Consequence: dropping the all-sum of `Inkling2SparseMoE.callAsFunction` (the path ShardCheck's sharded run takes: `prefill` -> `callAsFunction`, 54 calls with the group set over 9 steps) leaves the sharded logits bit-identical, so `scripts/shard-check.sh --tiny` cannot catch an Inkling MoE sharding break. Dropping the attention-output all-sum IS caught.

- [x] Find why the MoE branch is invisible: a property of the tiny config (attention-residual blocks of 2 layers, `local_layer_ids`, softcaps, random weight scale 0.3), or a model bug where the MoE output is dropped. Check whether a real Inkling 2 checkpoint's logits depend on its experts. — The tiny weight generation: NVFP4 experts were quantized without a global scale and then given `global_scale = 1`, which scales each projection down by 6 * 448 = 2688, so the MoE output was about 1e-9. The real Inkling-2-Mini-4bit checkpoint depends on its experts on every path (see comments).
- [x] If it is the tiny config, change `Inkling2TinyModels.targetJSON` (or the weight scale) so the experts reach the logits, and show the tiny shard-check then fails when the `Inkling2SparseMoE` all-sum is dropped. — `Inkling2TinyModels.target` now quantizes each expert with global scale = its absmax (the checkpoint's convention). shard-check fails 4/4 with the all-sum dropped and passes 4/4 with it restored.
- [x] If it is a model bug, fix it with a failing test first. — NOT APPLICABLE: not a model bug. A real checkpoint's logits move by 19.3 when one layer's expert down scales are zeroed, and by 22.9 (cached prefill) / 17.7 (decode) when all are zeroed. #cluster