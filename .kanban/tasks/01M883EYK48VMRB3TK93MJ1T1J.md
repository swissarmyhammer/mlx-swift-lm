---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m883r90yr5btq77hga3ygrvm
  text: |-
    Research (picked up):
    - No tokenizer loads in ShardCheck without a new dependency: the HF tokenizer comes only through MLXHuggingFace's macros (#huggingFaceTokenizerLoader), which need swift-transformers in the consumer; the root package does not depend on it (only IntegrationTesting does). So the default prompt is pre-tokenized per model family and checked in.
    - Sentence: "The lighthouse keeper climbed the narrow stairs every evening to light the lamp, and from the top he watched the fishing boats return safely to the harbor before the storm arrived." Encoded with HF `tokenizers` (uv scratch env, not a project dep), add_special_tokens=False; decode round-trips. qwen3_5: 34 ids, identical for Qwen3.5-2B-mxfp4, Qwen3.8-27B-4bit and -mxfp4 (2B's tokenizer.json differs in bytes but encodes the same). inkling2_loop: 40 ids, Inkling-2-Mini-4bit and oQ4e-mtp share tokenizer.json (same blob).
    - ShardCheck selects the default by config.json model_type (decoded through BaseConfiguration); an unknown model type throws "no default prompt ... pass --tokens" instead of falling back to made-up ids.
    - Qwen3.8-27B-4bit was not in the HF cache; downloaded mlx-community/Qwen3.8-27B-4bit (snapshot b935172e) with huggingface_hub in a uv scratch env.
    - A 2-Mac JACCL repeat from another session (zyx7p8z comment, 01:02) already used a 34-id English Qwen prompt; this card's prompt is a different sentence.
  timestamp: 2026-11-25T01:05:06.590832+00:00
- actor: claude-code
  id: 01m884ac4tnc0w7eyj0fshhx6z
  text: |-
    Shard-check results with the final code (Release binary built after the last source edit). One Mac, 2 ring processes on localhost, default English prompt, teacher-forced (the new default), gate unchanged (KL ≤ 0.05; argmax mismatch fails unless the reference margin ≤ max(0.05, 2 bf16 ulps)).

    | checkpoint | FAST_SYNCH | result | max KL (step) | argmax agree | ties skipped | ms/token sharded vs local |
    |---|---|---|---|---|---|---|
    | Qwen3.8-27B-4bit (b935172e) | off | PASS | 0.00148 (3) | 9/9 | 0 | 51.1 vs 19.2 |
    | Qwen3.8-27B-4bit | 1 | PASS | 0.00148 (3) | 9/9 | 0 | 29.2 vs 19.1 |
    | Qwen3.8-27B-mxfp4 (bbdfd10f) | off | PASS | 0.00268 (7) | 9/9 | 0 | 50.0 vs 18.2 |
    | Qwen3.8-27B-mxfp4 | 1 | PASS | 0.00268 (7) | 9/9 | 0 | 28.5 vs 18.3 |
    | Inkling-2-Mini-4bit (9856dc53) | off | PASS | 0.00722 (2) | 8/9 | 1 (decode 6, margin 0.031) | 36.6 vs 13.2 |
    | Inkling-2-Mini-4bit | 1 | PASS | 0.00722 (2) | 8/9 | 1 | 22.5 vs 12.7 |
    | Inkling-2-Mini-oQ4e-mtp (3eea0427) | off | PASS | 0.01638 (2) | 9/9 | 0 | 36.9 vs 13.4 |
    | Inkling-2-Mini-oQ4e-mtp | 1 | PASS | 0.01638 (2) | 9/9 | 0 | 23.1 vs 13.4 |

    Other modes (Qwen3.8-27B-mxfp4, no FAST_SYNCH):
    - --free-running, English prompt: PASS, max KL 0.00268, tokens equal.
    - --announce: PASS, tokens equal; rank 1 applied 15 ops, 1 park, 0 failed.
    - Old synthetic prompt (1000+37i), teacher-forced: PASS. The prefill prints "argmax tie (margin 0.03125), skipped" (276 vs 261, the zyx7p8z tie); max KL 0.00048.
    - Old synthetic prompt, --free-running: FAIL as before (KL 1.798 at step 7, tokens differ). This is expected: free running still needs equal tokens, and the tie cascades.
    Note: Inkling-2-Mini-4bit has its own near tie at decode 6 on the English prompt, with KL 0.006 at that step. A free-running run there would fail on the token check, and teacher forcing plus the tie gate handles it.

    Comparator self-check: python3 -I scripts/test_shard_check_compare.py, 11 tests OK. ruff (D1, PLR2004, E/F/W) is clean.
    Test bundles (swift build --build-tests + xcrun xctest): MLXLMTests 1260 XCTest + 2724 Swift Testing (3 known issues), MLXGuidedGenerationTests 95, MLXFoundationModelsTests 611, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11. All passed, with 0 failures.
    No Swift unit test covers ShardCheck. It is an executable with no test target, and adding a sixth bundle would change the five-bundle setup in CLAUDE.md. The real-model runs above are its verification.

    ### implement — changed
    - evidence: 97f9ad18, 07a1d3ec; 4 files: scripts/shard_check_compare.py, scripts/test_shard_check_compare.py, tools/ShardCheck/ShardCheck.swift, scripts/shard-check.sh
    - next: /review
  timestamp: 2026-11-25T01:14:59.610348+00:00
position_column: done
position_ordinal: fffff280
title: 'ShardCheck: teacher-force reference tokens and tie-aware argmax gate'
---
Qwen3.8-27B-mxfp4 "failed" the shard check on the default prompt because of a near-exact tie: top-2 logits of 6.59375 vs 6.5625 (one bf16 step apart, KL 0.0003). Sharding flipped the winner, and every later step then compared different contexts (KL 1.8). That isn't a sharding defect (zyx7p8z bisect). Make the check robust:
- [x] **Teacher forcing:** the sharded run feeds the reference run's greedy tokens (from the reference .json) instead of its own argmax, so one flipped tie can't cascade. Per-step KL then compares the same contexts.
- [x] **Tie-aware argmax gate:** an argmax mismatch fails only when the reference margin (top1 − top2 logit) is above a small epsilon, e.g. 2 bf16 ulps at that magnitude or 0.05 logits. Print ties as "tie, skipped".
- [x] **Default prompt:** switch to real English token ids (pre-tokenized per model family, or have ShardCheck accept text and use the model's tokenizer if one is available), so made-up ids don't invite ties.
- [x] Re-run Qwen3.8-27B-4bit/mxfp4 and Inkling 4bit/oQ4e-mtp; all must pass with the gate unchanged (KL ≤ 0.05). #cluster