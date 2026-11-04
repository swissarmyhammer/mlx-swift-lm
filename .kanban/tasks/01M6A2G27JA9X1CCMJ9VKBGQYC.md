---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m6asxc29qzgdw22am5bxmnag
  text: 'The same small change fixed the code here, under ^84b33jx (uncommitted, branch mtp): `SpeculativeTokenIterator` stores `GenerationBufferPoolClearing` and clears through one `emitPendingToken()`, which both hand-out sites now call, so it counts emitted tokens. Test `draftModelIteratorClearsEveryIntervalOfEmittedTokens` (RED then GREEN; 2 mutants, no clear and round-start only, both fail it). The second box is checked on that basis. The first box stays open: no long draft-model decode was measured, so whether the pool grew on this path, and the tok/s effect, are not known.'
  timestamp: 2026-11-01T05:39:27.433037+00:00
- actor: claude-code
  id: 01m6b201j0kb9kz5g9dg39xydk
  text: |-
    Measured the long draft-model decode. The pool grows on this path, and the clear pays for itself.

    Pair: target mlx-community/Qwen3-4B-4bit, draft mlx-community/Qwen3-1.7B-4bit (same vocabulary hash), both from the local HF cache, nothing downloaded. Qwen3.5-4B as a draft for Qwen3.8-27B was tried first and cannot run: `SpeculativeTokenIterator.init` throws "Speculative decoding requires a trimmable draft KV cache.", because the linear_attention layers of qwen3_5 are not trimmable. So no 3.5 or 3.8 checkpoint can be a draft model, and the MTP head is the only speculation for those targets.

    Method: ChatSession with SpeculativeDecodingConfig(draftModel:) (numDraftTokens 5), temperature 0, prompt 100,158 tokens, maxTokens 12,000, Release via xcodebuild, separate process per arm, same binary, runtime flag guarding the clear in emitPendingToken (temporary hook as GuidedPoolProbe), order ABBA, bench.lock around each run, wait for 1-minute load below 4.0.

    | run | arm | load at start (1m / 5m) | tok/s | peak cache GiB | peak active+cache GiB | swap delta GiB | swapouts | min free GiB | peak compressor GiB | pressure |
    | 1 | clear | 2.42 / 2.95 | 35.19 | 27.68 | 58.11 | -0.01 | 0 | 0.2 | 0.7 | 1 |
    | 2 | noclear | 2.51 / 3.54 | 21.49 | 46.20 | 73.87 | +10.16 | 8,859,502 | 0.0 | 51.7 | 4 |
    | 3 | noclear | 3.90 / 4.94 | 14.93 | 46.21 | 73.87 | +9.74 | 22,937,003 | 0.0 | 60.2 | 4 |
    | 4 | clear | 3.19 / 6.70 | 34.25 | 27.68 | 58.11 | -9.24 | 38,591 | 0.1 | 1.0 | 1 |

    Mean 34.72 tok/s with the clear, 18.21 without: 1.91 times. Peak active+cache 58.1 against 73.9 GiB. Each peak is the highest value of its run, the final `decoded` snapshot included; the clear-arm peaks are that snapshot, which the `PEAKS` lines do not cover. All four runs wrote the same text (hash 55a01bdcb27ff228, 12,000 tokens, stop length) with the same telemetry: 2,020 rounds, 10,098 drafted, 9,981 accepted. Run 3 started after a 9-minute load wait, with 23.40 GiB of swap. That is about 0.9 GiB above the baseline of about 22.5 GiB (runs 1 and 2 started at 22.52 and 22.50), although run 2 ended at 32.66 GiB. Run 4 started with 32.56 GiB of swap, and its -9.24 GiB is that swap draining, to 23.32 GiB at the end of the run. Run 4's 38,591 swapouts are a measured count and are unexplained: they came in three steps at t56, t60 and t62 with swapUsed flat at about 24.3 GiB, and the log does not show their cause. Run 4 also started with a 5-minute load of 6.70. A calibration run of the clear arm, outside ABBA, gave 34.94 tok/s, same peaks, 0 swapouts.

    Limits of these numbers. 100,158 tokens is past the 40,960 max_position_embeddings of Qwen3-4B, and the text ends in a repeated sentence, so the 98.8% acceptance and the tok/s are not those of useful output; the memory and swap comparison is what this shows. tok/s is token rate over the whole decode including the swap of the noclear arms. Free memory falls to 0.1-0.2 GiB in the clear arm too, mostly reclaimable file and inactive pages (pressure 1, no swap growth). The clear arm still holds 27.7 GiB of pool at its peak.

    Noted: `GenerateCompletionInfo.proposedDraftTokens` and `acceptedDraftTokens` are nil on this path (they come from the MTP stats); the counts are in `speculativeDecodingTelemetry`.

    Test coverage confirmed: `draftModelIteratorClearsEveryIntervalOfEmittedTokens` in Tests/MLXLMTests/GenerationBufferPoolClearingTests.swift already covers the clear, so no test was added. Mutants run again in Debug on the full MLXLMTests bundle: clear removed fails it, clear only on the first token of a round fails it, and each fails only that test (indices == [0, 256, 512]). Restored tree: 2092 tests passed with 1 known issue.

    Tree: hook, probe and mutants removed; no source differs from HEAD. Only this card's files are modified. Probe kept in the scratchpad as PoolProbeVK.swift.keep, runner vk-run.sh, log vk-probe.log, per-run load in vk-ab.load.
  timestamp: 2026-11-01T08:00:43.584948+00:00
- actor: claude-code
  id: 01m6b205cjzrcy3tvfxc3aybf8
  text: |-
    ### implement — no-change
    - evidence: no source file changed (git diff HEAD empty outside .kanban/tasks/01M6A2G27JA9X1CCMJ9VKBGQYC.*). Measurement ABBA Release: clear 35.19/34.25 tok/s, noclear 21.49/14.93 tok/s; peak cache 27.6 vs 46.2 GiB; peak active+cache 58.0 vs 73.9 GiB; swapouts 0/38,591 vs 8,859,502/22,937,003. draftModelIteratorClearsEveryIntervalOfEmittedTokens fails under 2 of 2 mutants; xcrun xctest MLXLMTests 2092 passed, 1 known issue.
    - next: /review, card stays in doing.
  timestamp: 2026-11-01T08:00:47.506901+00:00
- actor: claude-code
  id: 01m6b25hpaqdhh8m2ztpk3z985
  text: |-
    ### review — findings
    - evidence: 3 findings by hand, recorded in ## Review Findings (2026-11-01 02:03), all on the measurement comment against scratchpad vk-probe.log: clear-arm peaks omit the decoded row (27.68 / 58.11), run 3 swap start is 23.40 GiB not the +10.16 of run 2, run 4 swapouts are not shown to be the drain. Engine not run: no source changed (git diff HEAD is this card only), and the clear was reviewed under ^84b33jx. Matching: every other table value, load columns against vk-ab.load, means 34.72 / 18.21, 1.91 times, 98.8%, hash 55a01bdcb27ff228, telemetry 2,020 / 10,098 / 9,981 on all four runs, calibration 34.94, 40,960 max_position_embeddings. Limits stated. draftModelIteratorClearsEveryIntervalOfEmittedTokens exists and tests the 256-token cadence on SpeculativeTokenIterator; mutants give indices [] and [0], each the only failure of 2092 tests; base run 2092 passed, 1 known issue.
    - next: correct the three statements in a new comment, check the boxes, /review. Card stays in review.
  timestamp: 2026-11-01T08:03:43.946860+00:00
- actor: claude-code
  id: 01m6b2cqndsve18zbffcmenm8d
  text: |-
    ### implement — changed
    - evidence: measurement comment 01m6b201j0kb9kz5g9dg39xydk corrected against vk-probe.log (decoded rows: cache 27.68 runs 1 and 4, sum 58.11 run 4; no clear-arm row above either; swap 32.66 end of run 2, 23.40 start of run 3, 22.52 / 22.50 baseline; run 4 swap 32.56 -> 23.32, swapouts +15,227 / +11,674 / +11,690 at t56 / t60 / t62 = 38,591). 3 findings checked. `blockSize` doc comment in Tests/MLXLMTests/GenerationBufferPoolClearingTests.swift corrected: it was true of the MTP tests only. In the draft-model test `prepare` returns `.tokens` for this prompt, so no prefill token precedes the first round, and 256 is the middle and 512 the last token of their rounds. vk-mut2.MLXLMTests.log supports this: clear on the first token of a round gives indices [0]. `swift build --build-tests`: Build complete, comment-only source change. No GPU, no test run, no commit.
    - next: /review. The previous step record still reads peak cache 27.6 and peak active+cache 58.0; it was left as written history. Card stays in review.
  timestamp: 2026-11-01T08:07:39.437134+00:00
- actor: claude-code
  id: 01m6b2jxsg6x9z4w1206w20nxe
  text: |-
    ### review — clean
    - evidence: second round, by hand as instructed: 0 new findings, 3 of 3 prior findings checked. Engine not run. Comment 01m6b201j0kb9kz5g9dg39xydk recomputed from vk-probe.log: every table row of runs 1 to 4 matches RESULT, PEAKS, decoded and the snapshot rows (tok/s, swap, swapouts, min free, peak compressor, pressure, load). Peak cache and active+cache: no snapshot or same-tick STEP row exceeds 27.68 / 58.11 (clear) or 46.21 / 73.87 (noclear). PEAKS precedes the decoded snapshot (sampler.finish(), then snapshot) in PoolProbeVK.swift.keep. Means 34.72 / 18.21, 1.91 times, 98.8% match. Swap: 32.66 end of run 2, 23.40 start of run 3, 22.52 / 22.50 baseline. Run 4: 32.56 -> 23.32, swapouts +15,227 / +11,674 / +11,690 at t56 / t60 / t62 = 38,591, swapUsed 24.32 to 24.29 across them, no rise in either clear run. Doc comment on `blockSize` in Tests/MLXLMTests/GenerationBufferPoolClearingTests.swift against the code: MockMainModel.prepare returns .tokens, so the MTP .tokens path queues one prefill token and rounds start at index 1 (256 = 1 + 3*85 first, 512 = 1 + 3*170 + 1 middle). LLMModel.prepare returns .tokens for a 3-token prompt (default step 512) for main and draft, no pending token, the first verify pass takes the prompt and rounds start at 0 (256 = 3*85 + 1 middle, 512 = 3*170 + 2 last). A round emits blockSize tokens on both iterators (numDraft accepted + bonus). A clear at one end only misses a multiple on both. vk-mut2.MLXLMTests.log gives indices [0], as the doc implies. The passthrough test has no rounds and the doc is scoped to a speculating iterator.
    - next: none. Moved to done. The doc comment in Tests/MLXLMTests/GenerationBufferPoolClearingTests.swift is uncommitted.
  timestamp: 2026-11-01T08:11:02.320580+00:00
position_column: done
position_ordinal: ffff9b80
title: SpeculativeTokenIterator never clears the MLX buffer pool (the draft-model path, as MTP did before ^y1hk1vz)
---
Found in ^y1hk1vz. `TokenIterator` and, after ^y1hk1vz, `MTPSpeculativeTokenIterator` clear the MLX pool every 256 emitted tokens through `GenerationBufferPoolClearing`. `SpeculativeTokenIterator` in `Libraries/MLXLMCommon/Evaluate.swift` (draft-model speculation) has no clear, by reading the code. It is not measured. A long draft-model decode may grow the pool the same way, for its target cache and its draft cache.

- [x] Measure a long draft-model decode at long context: pool, swap, tok/s.
- [x] If the pool grows, store `GenerationBufferPoolClearing` in the iterator and count emitted tokens, with a test as in `GenerationBufferPoolClearingTests`.

#perf

## Review Findings (2026-11-01 02:03)

- [x] `vk-probe.log` `hand/card-numbers` — The measurement comment's peak values for the clear arm are lower than the log's maximum. The table gives peak cache 27.55 (run 1) and 27.56 (run 4) and peak active+cache 57.98 (run 4), and the text gives 27.6 and 58.0. These are the `PEAKS` lines, which the probe prints before the final snapshot. The `decoded` row of runs 1 and 4 records cache 27.68 and sum 58.11, and that row is the highest of each run. State peak cache 27.68 and peak active+cache 58.11 for both clear runs, 27.7 and 58.1 in the text, or state that the peaks exclude the final snapshot.
- [x] `vk-probe.log` `hand/card-numbers` — The measurement comment says run 3 started "on the swap run 2 left". Run 2 ended with swap 32.66 GiB, run 3 started with 23.40 GiB, and run 1 and run 2 started with 22.52 and 22.50 GiB. So run 3 started about 0.9 GiB above the baseline, not on the +10.16 GiB of run 2. State that figure, and do not give the swap left by run 2 as a reason for the rate of run 3 without a measurement of it.
- [x] `vk-probe.log` `hand/card-numbers` — The measurement comment says run 4's 38,591 swapouts "are the same swap draining". The swapouts counter counts pages written to swap, and the log does not show that these writes are the drain. It rose in three steps at t56, t60 and t62, with swapUsed flat at about 24.3 GiB, after swap had already fallen from 32.56 to 24.32 GiB. State the -9.24 GiB as the drain, and state the 38,591 swapouts as a measured count whose cause is not shown.