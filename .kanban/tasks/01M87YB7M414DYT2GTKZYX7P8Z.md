---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m882139hz7x6bhy7ygn9eybc
  text: |-
    Research (picked up):
    - oQ4e-mtp is an oMLX layout (attn.in_proj fused, experts.w13/w2) that Inkling2OMLXLayout remaps. Per-layer quantization: lm_head, embed, attn in_proj/q/k/v/o 8-bit affine gs64. Experts of layers 0-19 nvfp4 gs16 (with global_scale); experts of layers 20-23 are NOT in config's per-layer list and are 4-bit affine gs64 (w2.biases [128,2304,18]). So the expert format differs between layers.
    - Full-layer q_proj in the checkpoint is 6552 rows (q 3072 + r 384 + gate 3072 + sink 24), k_proj 1056 (k 1024 + weight_proj 32); sliding in_proj 8600. After remap each port projection is its own QuantizedLinear.
    - For N=2: o_proj inputs 3072 = 48 gs64 groups -> 24/rank OK; expert down inputs 1152 = 72 nvfp4 groups -> 36 OK, = 18 affine gs64 groups -> 9 OK. N=4 would fail at the affine experts (18 groups).
    - MLXNN shardLinear already checks sharded-to-all quantization groups but throws ShardingError.quantizationGroup without naming the layer; row splits are only checked inside shardInPlace, i.e. during mutation.
    - Plan: TensorParallel.requireWholeRows / requireWholeInputs (+ per-projection wrapper) in MLXLMCommon, called from each model's per-layer validation before makeShards builds anything; replaces Inkling2TensorParallelPlan.requireWholeQuantizationGroups.
  timestamp: 2026-11-25T00:34:58.481601+00:00
- actor: claude-code
  id: 01m882638h9ktj0xdyc4dn9wms
  text: |-
    Baseline shard-check (before code changes, default prompt = ShardCheck's synthetic 1000+37i ids):
    - Inkling-2-Mini-oQ4e-mtp: PASS, max KL 0.04832 at step 1 (thin margin to 0.05), argmax 9/9.
    - Qwen3.8-27B-mxfp4: FAIL. Prefill KL 0.00030 but argmax flips: reference top-2 is 276 = 6.59375 vs 261 = 6.5625, a single bf16 ulp (spacing 0.03125 in [4,8)). Sharded gives 261 6.5625 / 276 6.53125. After the flip each run decodes its own tokens, so the later KLs (up to 1.8) compare different contexts.
    Bisect by layer type (temporary SHARD_BISECT patch, reverted, never committed; default prompt): attention-only prefill KL 0.00030, flips the tie (FAIL); GDN-only 0.00029, PASS; MLP-only 0.00016, PASS. Each type adds the same order of bf16 partial-sum noise; attention's share happens to tip the tie.
    Teacher-forced past the tie (prompt + 276): attention-only PASS max KL 0.00033; full sharding PASS max KL 0.00033 at step 5, argmax 9/9, tokens equal.
    Prose prompt (34 tokens of English text): full sharding PASS max KL 0.00174 at step 4, 9/9.
    Cause: a one-ulp near-tie in the default synthetic prompt, not a sharding defect. Gate not loosened.
  timestamp: 2026-11-25T00:37:42.289132+00:00
- actor: claude-code
  id: 01m882ywq3h26jvcpxve28gmyf
  text: |-
    Shard-check results (one Mac, 2 ring processes on localhost, M-series Mac17,15 96 GB; final code with the group-size validation). Gate: tokens + argmax every step + KL <= 0.05, unchanged.

    | checkpoint | mode | prompt | result | max KL (step) | argmax | ms/token sharded vs local |
    |---|---|---|---|---|---|---|
    | Qwen3.8-27B-mxfp4 | ring | default synthetic | FAIL | 1.798 (7) | 6/9 | 50.1 vs 18.2 |
    | Qwen3.8-27B-mxfp4 | ring | prose (34 tok) | PASS | 0.00174 (4) | 9/9 | 51.3 vs 18.1 |
    | Qwen3.8-27B-mxfp4 | ring + MLX_METAL_FAST_SYNCH=1 | default | FAIL | 1.798 (7) | 6/9 | 28.4 vs 18.1 |
    | Qwen3.8-27B-mxfp4 | ring + FAST_SYNCH | prose | PASS | 0.00174 (4) | 9/9 | 28.6 vs 18.1 |
    | Inkling-2-Mini-oQ4e-mtp | ring | default | PASS | 0.04832 (1) | 9/9 | 36.5 vs 13.3 |
    | Inkling-2-Mini-oQ4e-mtp | ring | prose (39 tok) | PASS | 0.00444 (2) | 9/9 | 42.7 vs 13.2 |
    | Inkling-2-Mini-oQ4e-mtp | ring + FAST_SYNCH | default | PASS | 0.04832 (1) | 9/9 | 23.2 vs 13.2 |
    | Inkling-2-Mini-oQ4e-mtp | ring + FAST_SYNCH | prose | PASS | 0.00444 (2) | 9/9 | 23.1 vs 13.3 |

    Notes:
    - Qwen default-prompt FAIL: one-bf16-ulp near tie at prefill (276 = 6.59375 vs 261 = 6.5625), prefill KL 0.0003; see the bisect comment above. Teacher-forced past the tie, full sharding passes with max KL 0.00033. Not a sharding defect; gate not loosened. The default ShardCheck prompt (ids 1000+37i) is junk text, so near-ties are likely; follow-up suggestion: a prose default prompt.
    - Inkling default-prompt KL 0.048 at step 1 sits close to the limit because that step's reference distribution is nearly flat (entropy 5.36 nats); on prose, max KL is 0.0044.
    - Loader handles oQ4e's per-layer modes: 8-bit affine gs64 attention/embed/lm_head, nvfp4 gs16 experts in layers 0-19, 4-bit affine gs64 experts in layers 20-23 (absent from config's per-layer list, detected from tensor shapes). Sharding validates and splits all of them for N=2.
    - Not done here: the 2-Mac JACCL + FAST_SYNCH repeat (FoundationModelsCluster / te3zhxt runner) — that card item is for that session.
  timestamp: 2026-11-25T00:51:14.787871+00:00
- actor: claude-code
  id: 01m883e104es7renyzqpz92jct
  text: |-
    ### implement — changed
    - evidence: e2f26822; 5 files — Libraries/MLXLMCommon/TensorParallel.swift (QuantizationGroups, requireWholeRows/requireWholeInputs, SplitProjections), Libraries/MLXLLM/Models/Qwen35TensorParallel.swift, Libraries/MLXVLM/Models/Inkling2TensorParallel.swift (requireWholeShares per layer, called in makeShards before any mutation; old plan.requireWholeQuantizationGroups removed), Tests/MLXLMTests/TensorParallelShardingTests.swift, Tests/MLXLMTests/Inkling2TensorParallelTests.swift. All 5 bundles green: MLXLMTests 1260 XCTest + 2724 Swift Testing (3 pre-existing known issues), MLXGuidedGenerationTests 95, MLXFoundationModelsTests 611, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 XCTest. Shard-check: Inkling oQ4e PASS both modes; Qwen mxfp4 FAIL on default synthetic prompt (1-ulp prefill tie, bisected, not a sharding defect), PASS on prose prompt both modes.
    - next: the 2-Mac JACCL repeat (unchecked item) needs the FoundationModelsCluster session; consider a prose default prompt for ShardCheck so the default gate does not trip on junk-text ties.
  timestamp: 2026-11-25T00:59:30.692403+00:00
position_column: doing
position_ordinal: '8180'
title: Shard-check the Router's actual checkpoints (mxfp4, oQ4e-mtp/nvfp4)
---
Sharding was validated only on the plain affine 4-bit checkpoints (Qwen3.8-27B-4bit, Inkling-2-Mini-4bit). The Router's model lists also use mlx-community/Qwen3.8-27B-mxfp4 and mlx-community/Inkling-2-Mini-oQ4e-mtp. The latter has mixed quantization: 8-bit affine gs64 attention, embed and lm_head, plus nvfp4 gs16 experts. Its per-layer quantization entries in config.json also differ between in_proj and q/k/v layers.

- [x] Run `scripts/shard-check.sh` (one Mac, ring) on Qwen3.8-27B-mxfp4 and Inkling-2-Mini-oQ4e-mtp. Gate: tokens + argmax at every step + KL ≤ 0.05.
- [x] Check that each quantized split respects that format's group size: after splitting, (in/N) % groupSize == 0 for sharded-to-all and the row split for all-to-sharded, for mxfp4, nvfp4 and affine. Make `shard()` throw `.indivisible` naming the layer when it doesn't hold.
- [x] Unit-test the plan for a mixed-quantization config.
- [ ] Ask the FoundationModelsCluster session (te3zhxt runner) to repeat each on 2 Macs over JACCL with FAST_SYNCH.
- [x] Record results on this task. #cluster