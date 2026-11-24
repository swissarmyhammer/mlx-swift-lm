---
comments:
- actor: wballard
  id: 01m87hw96ym9r5dn89r63a8c2g
  text: |-
    Research findings:
    - Experts in Inkling-2-Mini-4bit are NVFP4 (gs 16, global_scale per expert), not affine gs64; dense attention projections are affine 4-bit gs64. Per-rank expert width 576: 576%16==0 and 576%64==0.
    - gate_up_proj rows interleave gate/up per channel (kernel reads row 2c and 2c+1), so a contiguous split on axis 1 (.count(1)) keeps each channel's pair together. OMLX w13 maps straight to gate_up_proj, same layout.
    - rel_logits_proj.proj is [d_rel, extent], shared across heads: stays whole.
    - Dynamic conv static_weight is [K, kvDim]: channels on axis 1. weight_proj rows are kvHeads*K, head-major.
    - Fused query/key/value caches are views of per-projection params; shard the views, the attention update override invalidates, then Inkling2.prepare() re-fuses.
    - Expert decode kernels fuse combine+conv+residual. Sharded path: run them with zero residual/state, take the last state row (the expert sum), allSum, then inkling2ExpertConvolution. The sequential verify path already uses this trick.
    - Inkling2TextModel is private in Inkling2.swift, so the model-level conformance lives there.
  timestamp: 2026-11-24T19:52:43.486283+00:00
- actor: wballard
  id: 01m87jjr2r2begmzwz2r8mg4ng
  text: |-
    Progress:
    - Implementation landed (plan, attention/MoE sharding, Inkling2 conformance, ShardCheck VLM load). 19 plan/split/group-of-one tests + 3 share-sum tests green.
    - Dead end: first share-sum test reseeded the global MLXRandom per model, and parallel test cases raced on it (6.0 abs diff, varying run to run). Fixed by copying the whole layer's parameters into each share and using a keyed RNG for the input. After that shares sum to the whole within 1e-4 (f32) for sliding attention, full attention (dynamic conv) and MoE.
    - Shard check on Inkling-2-Mini-4bit: tokens 9/9, max raw logit diff 1.48 (fails old 0.5 abs gate). Bisect: attention-only 1.18, MoE-only 1.25. Float32 activations drop it to 0.23. Unsharded prefill-vs-decode of the same tokens differs by 0.30-0.74 per step, so this is bf16 noise. Coordinator switched the gate to KL: max KL 0.00484 at prefill (limit 0.05), PASS. Same with MLX_METAL_FAST_SYNCH=1.
    - mlx-swift fork re-resolved to 6eb4ccc (stale untracked Package.resolved + workspace-state had pinned 98f4a90).
  timestamp: 2026-11-24T20:04:59.608846+00:00
- actor: claude-code
  id: 01m87wjkff5e4005cx8enr6abw
  text: |-
    ### implement — stuck (session died)
    - The implementing subagent was lost when the terminal (zellij) died, around 2026-11-24 20:05Z. Its work is **uncommitted** in the working tree on `cluster`:
      - modified: Libraries/MLXLMCommon/TensorParallel.swift, Libraries/MLXVLM/Models/Inkling2.swift, Inkling2Attention.swift, Inkling2Layers.swift, Inkling2MoE.swift, Package.swift, tools/ShardCheck/ShardCheck.swift
      - new: Libraries/MLXVLM/Models/Inkling2TensorParallel.swift, Tests/MLXLMTests/Inkling2TensorParallelTests.swift
    - Unknown: whether it builds, and whether shard-check passed. The next step is to build, run `scripts/shard-check.sh <Inkling-2-Mini-4bit snapshot>` (the gate is now KL ≤ 0.05 + argmax, commit 965d6424), then run the full unit suite, then commit.
    - Pick up the mlx-swift fork at jaccl-no-ipv4 6eb4ccc (FAST_SYNCH fence fix): re-resolve packages. Also run the check with MLX_METAL_FAST_SYNCH=1.
    - Keep ShardCheck's JACCL env selection (b75c680d).
    - Per the updated CLAUDE.md, tests can also run as `swift build --build-tests` then `xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest`. Run all 5 bundles before calling it green.
  timestamp: 2026-11-24T22:59:40.655756+00:00
- actor: claude-code
  id: 01m87xhx0awvbstkan8an36g3x
  text: |-
    Picked up after the session died. The uncommitted work built. Changes on top of it:
    - Inkling2TextModel now conforms to TensorParallelShardable (a card item that was missing). Inkling2.shard calls it, then prepare() and invalidateCompiledTraces().
    - Inkling2.kvHeads reads each visited layer's attention.kvHeads, so the stored plan optional is gone. New test: "the key-value heads of each layer visit are the heads its layer keeps" (RED [2 x8] before the change, GREEN after).
    - Fixed an unused-result warning in the share-sum test, and added a doc comment to TensorParallel.Split.init.
    - mlx-swift fork checkouts are at 6eb4ccc (both .build/checkouts and .build/shard-check).
  timestamp: 2026-11-24T23:16:46.218951+00:00
- actor: claude-code
  id: 01m87xv6tsqxgqkq11tg8zf644
  text: |-
    ### implement — changed
    - evidence: commits 3f8ea0ba feat(inkling2) (sharding + plan tests), b821849b test(shard-check) (VLM load). Files: TensorParallel.swift, Inkling2.swift, Inkling2Attention.swift, Inkling2Layers.swift, Inkling2MoE.swift, Inkling2TensorParallel.swift (new), Inkling2TensorParallelTests.swift (new), Package.swift, ShardCheck.swift.
    - shard-check on Inkling-2-Mini-4bit (one Mac, 2-process ring, mlx-swift 6eb4ccc): PASS. Tokens 9/9, argmax 9/9, max KL 0.00484 at step 0 (prefill); decode steps 0.0012-0.0029. Sharded 36.8 ms/token vs 12.8 unsharded. With MLX_METAL_FAST_SYNCH=1: PASS, same KL, 22.5 ms/token. Two processes on one GPU are slower than one; the speedup has to come from 2 Macs.
    - Tests (xctest, all 5 bundles): MLXLMTests XCTest 1260/0 failures + Swift Testing 2640 passed (3 known issues, all withKnownIssue in other suites); MLXGuidedGenerationTests 95; MLXFoundationModelsTests 605; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. 0 failures. The two known parallel-load flakes did not occur.
    - Open: not run on 2 real Macs over JACCL yet. Unit tests cover only groups of one; the shard-sum tests check the split math in one process.
    - next: review
  timestamp: 2026-11-24T23:21:51.193631+00:00
- actor: claude-code
  id: 01m87y183egxmrmpypph909a8r
  text: |-
    ### 2-Mac JACCL check — PASS
    - ShardCheck at a1aa97cf (mlx-swift 6eb4ccc); rank 0 on m5u2 rdma_en6, rank 1 on m5u1 rdma_en5; MLX_METAL_FAST_SYNCH=1; 32 decode steps.
    - 33/33 tokens and argmax match. Max KL 0.00484 at prefill. Decode 1–13: 0.0010–0.0029. Decode 14–32: 0.0002–0.0016.
    - Top-10 logprob diff rises from ~0.07 to ~0.51 by step 32, and max abs logit diff stays ≤ 1.48. Watch this in 17p77c9.
    - Speed, warm: 11.0 ms/token on one Mac, 11.4 on two. The MoE's small active compute means the allSums cost about what the split saves. Inkling Mini runs local unless it doesn't fit (the planner's head-fits rule).
  timestamp: 2026-11-24T23:25:09.102815+00:00
position_column: done
position_ordinal: ffffee80
title: Tensor-parallel sharding for Inkling 2 + ShardCheck VLM load
---
Shard the Inkling 2 text stack (mlx-community/Inkling-2-Mini-4bit, `inkling2_loop`) across a distributed group, as Qwen35 does.

- [x] Plan: per-rank head counts for full and sliding layers, expert inner width, parameter splits (axis + segments) for attention and experts, error cases
- [x] Attention: shard by head (q/k/v/r/gate/sink rows, QK clip scales, short convs incl. dynamic static_weight axis 1 and weight_proj), o_proj sharded-to-all, re-prepare fused projections
- [x] MoE: shard inside each expert (gate_up rows on axis 1, down inputs on last axis), one allSum after the expert combine, router whole
- [x] Loop: shard each distinct layer once
- [x] Inkling2 + text model conform to TensorParallelShardable; kvHeads reports the share
- [x] ShardCheck loads VLM model types (keep JACCL env selection)
- [x] shard-check.sh on Inkling-2-Mini-4bit passes (tokens agree, logits within tolerance)
- [x] Unit tests for the plan; full unit suite green