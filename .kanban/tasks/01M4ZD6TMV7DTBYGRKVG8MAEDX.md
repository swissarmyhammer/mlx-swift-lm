---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m50eqxpyv55r1y0t733x0yk0
  text: |-
    Research and micro-benchmark (scratchpad/qwen38bench mode `fuse`, file FuseBench.swift; random mxfp4 weights at the real shapes, group 32, bf16; each layer feeds the next layer). Logs: scratchpad/qwen38/fuse-decode-1.txt, fuse-decode-2.txt, fuse-prefill-1.txt. GPU idle before each run. CPU load: sourcekit-lsp 2 x ~85 %, SourceKitService ~70 %, fseventsd ~50 %, mobileassetd ~55 %.

    Decode (T = 1), median of 40:
    - MLP 64 layers (gate + up + silu*up + down): separate 14.73-14.97 ms, joined 15.00-15.11 ms. Joined is SLOWER by 0.14-0.28 ms (+0.9 to +1.9 %). With a compiled activation: +0.06 to +0.10 ms. The gate+up matmuls alone are 0.17 ms faster joined (10.51 -> 10.34 ms), but the activation that reads the two halves of the joined output costs more than that.
    - q/k/v 16 layers: separate 1.77-1.81 ms, joined 1.46-1.48 ms: -0.30 to -0.32 ms per token (-17 %).

    Prefill (T = 2048), median of 6:
    - MLP 64 layers: separate 3015-3017 ms, joined 3002 ms: -13 to -15 ms per block (-0.4 to -0.5 %), about 0.2 s for a 32k prefill (0.25 %).
    - q/k/v 16 layers: -0.8 to -4.6 ms per block (-0.4 to -2.2 %).

    Decision: the MLP join gives no decode gain (it is slower) and a prefill gain below the noise of full mode (81 to 86 s between runs). Thus no change for the MLP. The q/k/v join gives a decode gain; implement it with the method of the GDN join (FusedQuantizedLinearProjectionCache in MLXLMCommon/FusedQuantizedLinear.swift: one joined QuantizedLinear, the three named modules stay as views into the same storage, thus the checkpoint keys do not change; prepared by `Qwen35TextModel.prepare()`; the joined module goes into the compile state of the traces).
  timestamp: 2026-10-15T18:56:08.414091+00:00
- actor: claude-code
  id: 01m50jxnqgaqxt4xp8rhdwg4sa
  text: |-
    q/k/v join: implemented, measured on the real model, NOT faster, thus reverted. The tree is back at HEAD bc0f93e (only this card changed).

    What was built (kept as a patch for reference: scratchpad/qwen38/g8maedx-qkv-join.patch, tests: scratchpad/qwen38/g8maedx-Qwen35FusedAttentionProjectionTests.swift.txt): Qwen35Attention got a FusedQuantizedLinearProjectionCache for q_proj/k_proj/v_proj (views keep the checkpoint keys), prepared in Qwen35TextModel.prepare(), the joined module in the compile state of compiledAttentionPre and of the decode segments. Shared helpers went into FusedQuantizedLinearProjectionCache (sourceKeys, invalidating(afterUpdating:), invalidating(afterUpdatingModule:), prepare(installingInto:), sourceOutputs(_:)), and the LLM and VLM GDN used them. 18 new Swift Testing tests (bit-identical q/gate/k/v for affine 4-bit, affine 8-bit and mxfp4, each in bf16/f16/f32, decode and prefill shapes; whole-model logits for prompt + traced decode steps in each dtype; eligibility; checkpoint keys; update invalidation with a positive control) all passed.

    Same-session A/B, 32k, base = HEAD binary, new = q/k/v join (logs scratchpad/qwen38/g8m-*.txt). GPU idle (0 %) before each run. CPU load in each run: sourcekit-lsp 2 x ~88 %, SourceKitService 80-150 %, fseventsd 0-120 %, swift-package 206 % in prof-new-3.
    - profile (order base, new, new, base): B pipelined 29.21 / 29.23 / 29.17 / 29.18 ms; A sync 30.42 / 30.55 / 30.60 / 30.70 ms. Prefill 82.70 / 92.25 / 93.13 / 87.14 s (heat drift in runs 2-4: the first chunks of new-2 equal base-1, 4.568 vs 4.558 s, then all runs drift up); peak 18.1 GiB all.
    - step (order new, base, base, new), round 1: B 29.34 / 29.23 / 29.29 / 29.28 ms; C generate 28.86 / 28.77 / 28.71 / 29.00 ms; prefill 82.0 / 81.8 / 81.8 / 82.2 s.
    - full (order base, new, new, base): prefill 80.98 / 80.95 / 80.92 / 80.98 s (397.6-397.9 tok/s); decode 29.10 / 29.05 / 29.01 / 30.14 ms/tok; peak 17.9 GiB all; HARD reuse check PASS and warm == cold (tie-aware) PASS, thinking off and on, in all 4 runs.

    Conclusion: no measurable gain in decode (the micro-benchmark gain of 0.3 ms per token does not appear in the model: the three qmv of a full-attention layer are small and the decode step is compiled) and no gain in prefill (within 0.06 s of 81 s). The MLP join was not built: the micro-benchmark showed it slower at decode and only 0.4 % faster at prefill (see the research comment). Thus no code change.

    Final tree (= HEAD): swift build --build-tests 0 compiler warnings (one llbuild note "missing creator for mutated node", not a compiler warning, present before), xcrun xctest MLXLMTests.xctest: 716 XCTest tests 0 failures, 1503 Swift Testing tests in 122 suites passed. No model code changed, thus the real-weights integration tests were not necessary.

    Note for the next agent: a filtered Swift Testing run works with `swiftpm-testing-helper --test-bundle-path <MLXLMTests binary> --testing-library swift-testing --filter <Suite>` with DYLD_FRAMEWORK_PATH and DYLD_LIBRARY_PATH set to the MacOSX platform Developer Library/Frameworks and usr/lib, from a directory that holds a `default.metallib` link to .build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib.
  timestamp: 2026-10-15T20:09:11.152249+00:00
- actor: claude-code
  id: 01m50jxwre9pgq09p3r59559v7
  text: |-
    ### implement — no-change
    - evidence: micro-benchmark (scratchpad/qwen38/fuse-decode-1.txt, fuse-decode-2.txt, fuse-prefill-1.txt): MLP join +0.06 to +0.28 ms slower at decode, -0.4 % at prefill; q/k/v join -0.3 ms at decode in isolation. q/k/v join built and measured in the model, same-session A/B at 32k (scratchpad/qwen38/g8m-prof-*, g8m-step-*, g8m-full-*): profile B 29.21/29.18 base vs 29.23/29.17 new ms; step C 28.77/28.71 base vs 28.86/29.00 new ms; full prefill 80.98/80.98 base vs 80.95/80.92 new s, peak 17.9 GiB both, reuse check PASS in all runs. No gain, thus reverted; tree = HEAD bc0f93e. Final tree: swift build --build-tests 0 compiler warnings; xcrun xctest MLXLMTests.xctest 716 XCTest + 1503 Swift Testing tests, 0 failures.
    - next: review. The card acceptance "faster, or the card records why not" is met by the record above. The equality test was written and passed but was reverted with the code (kept in scratchpad/qwen38/g8maedx-Qwen35FusedAttentionProjectionTests.swift.txt).
  timestamp: 2026-10-15T20:09:18.350620+00:00
- actor: claude-code
  id: 01m50k02c6r569bq8qzzznjgxz
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (1ee08ff): 0 findings, 0 confirmed, 0 refuted. The change has only the two card files (.kanban/tasks/01M4ZD6TMV7DTBYGRKVG8MAEDX.md and .jsonl). No validator matches these files. No source file changed. The card comments record the measured evidence: the MLP join is slower at decode and only 0.4 % faster at prefill. The q/k/v join gives no gain at 32k in a same-session A/B (profile, step and full modes). Thus the code was reverted.
    - next: none. The task is in done.
  timestamp: 2026-10-15T20:10:29.638260+00:00
- actor: claude-code
  id: 01m50k0m8aq5ajctk25wbfa7ms
  text: |-
    ### finish iteration 1 — clean
    - implement: no-change — the MLP join is slower at decode; the q/k/v join was built, measured at 32k (A/B), and gave no gain, so it was reverted
    - test: green — run on the final tree by implement: xctest 716 + 1503 tests, 0 failures, 0 compiler warnings
    - commit: 1ee08ff (card files only)
    - review: clean — 0 findings; task moved to done
  timestamp: 2026-10-15T20:10:47.946689+00:00
position_column: done
position_ordinal: ffd880
title: 'Qwen3.8 speed: fuse the MLP gate and up projections, and the attention q/k/v projections'
---
## What

The MLP matmuls are the largest cost: 13.6 ms of a 30.5 ms decode token, and 2.96 s of a 5.84 s prefill block. The MLP `gate_proj` and `up_proj` read the same input in two quantized matmuls. The full-attention layers do q, k and v in separate matmuls. The GDN input projections are already fused.

## Approach

1. Join the gate and up weights (and q/k/v weights) into one quantized weight at load time (mxfp4, same group size), and split the output. Do the same for the scales.
2. Measure decode and prefill. Keep only what is faster.

## Measure

scratchpad/qwen38bench at 32k: profile and step modes for decode, full mode for prefill, peak memory and the reuse check (two runs each, idle GPU).

## Acceptance Criteria

- [ ] Decode or prefill at 32k is faster, measured, or the card records why not.
- [ ] A test shows the fused projection gives the same values as the separate projections.
- [ ] All MLXLMTests pass, with 0 compiler warnings. #qwen3-8 #performance