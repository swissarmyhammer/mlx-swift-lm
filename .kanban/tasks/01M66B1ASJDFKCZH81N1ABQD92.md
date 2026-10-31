---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m67dd1qc5n92jcz3mc39twn8
  text: |-
    Picked up. Baseline (HEAD a99b4079, Release, M5 Ultra, before any change; logs in scratchpad 1abqd92-before-*):
    - Sweep router MTP/plain tok/s (ms/round, eff. block): 2k 79.1/54.8 (33.6, 2.90); 8k 79.4/52.0 (37.9, 3.13); 32k 52.9/46.5 = 1.14x (43.1, 2.59); 65k 44.1/36.3 = 1.22x (72.1, 3.31); 100k 36.4/30.5 = 1.20x (93.7, 3.49).
    - Sweep text: 2k 81.9/55.8; 8k 88.5/54.3; 32k 75.2/46.9 = 1.61x (46.2, 3.54); 65k 54.7/38.0 = 1.44x (52.5, 3.02); 100k 41.7/31.6 = 1.32x (65.9, 2.97).
    - CPU load router 32k MTP/plain: 0% 1.159 (42.3 ms/round), 50% 1.088 (44.2), 100% 1.094 (50.8; plain 24.4 ms/tok vs 21.5 quiet). 2k: 1.48 / 1.37 / 1.16.
    - qwen38FourBitMTPGeneration passed: bs4 pMin0.75 mean 89.7 tok/s (math 99.4, code 85.8, prose 83.8); qwen38FourBitMTPSystemPromptCache passed, edit turn cached 2169 prefilled 123, tokens match plain.
    - Round cost router 32k: verify rows 1..4 [22.7, 30.5, 40.4, 46.5] ms, drafter proposal bs2..4 [1.5, 3.1, 4.7] ms; 100k: verify [28.0, 44.4, 57.8, 69.0], drafter [1.9, 8.4, 13.2] (full drafter window = default).
    Early read: at quiet load the router-32k round (42.3 ms at eff. block 2.59) is within ~1-3 ms of its GPU work (verify ~37 + drafter ~2.5 + commit ~1.5-2); the card's 55-vs-37 predates bs4/split-K. At 65k-100k most of the round over the verify is the full-window drafter attention (GPU work, 8-13 ms), not iterator overhead. Under CPU load the gap grows to ~8-10 ms/round at 100% load. Instrumented profile running next.
  timestamp: 2026-10-30T22:03:06.348880+00:00
- actor: claude-code
  id: 01m67mepymmf0w7nyg7hfctgvh
  text: |-
    Profile of one MTP round (temporary host-side timers around each phase of speculateRound + the drafter's syncs, no extra syncs; Release, greedy, bs4 pMin0.75, processor nil, ArgMax; ms per round averaged over the sweep's 1024-token decodes; logs scratchpad 1abqd92-profile-*, 1abqd92-probe1/2):
    | path / context | round | pre (=discard of last round's GDN snapshot slots) | seed gate | chain wait | chain build+encode | verify build | verify encode+wait | commit build | commit encode | book |
    | text 32k | 41.1 | 1.64 | 0.35 | 2.04 | 0.76 | 1.23 | 34.7 | 0.10 | 0.20 | 0.07 |
    | text 100k | 61.0 | 1.59 | 0.91 | 2.52 | 4.44 (encode 4.24) | 1.11 | 49.8 | 0.10 | 0.20 | 0.15 |
    | router 32k | 42.1 | 1.21 | 0.39 | 1.27 | 0.83 | 4.18 | 31.7 | 1.77 | 0.33 | 0.33 |
    | router 100k | 89.8 | 2.20 | 1.65 | 2.82 | 6.53 (encode 6.34) | 9.32 | 60.8 | 6.00 | 0.30 | 0.09 |
    Findings:
    1. "outside" (consumer between rounds) is 0.02 ms: the executor's handler is not a cost. Host syncs are cheap; asyncEval of the verify is effectively synchronous (encode throttled by MLX's 10-in-flight-command-buffer limit), so encode and GPU run in lockstep and the verify itself is GPU-bound.
    2. Dominant common overhead: MLX buffer-pool churn. MLXLanguageModel sets Memory.cacheLimit = 256 MiB process-wide; each round frees/allocates ~0.6 GiB of GDN snapshot slots (48 layers x 4 x 3 MiB) plus KV-length attention temporaries, which the pool cannot keep, so buffers go back to the OS and fresh ones are faulted in: the "pre" discard (1.2-2.2 ms), drafter-step encode at 100k (4-6 ms), and slower verifies. A/B in one process (probe1/probe2, cacheLimit only): router 100k MTP 38.0 -> 43.2 (1 GiB) -> 44.0 (2 GiB) -> 43.4 (4 GiB) -> 43.9 tok/s (16 GiB), round 89.8 -> 77.6 ms, discard 2.2 -> 0.16 ms, chain encode 6.3 -> 0.5 ms; plain 32.2 -> 36.1 tok/s (31.1 -> 27.7 ms/token = the bare forward). text 100k MTP 45.0 -> 53.8, plain 33.3 -> 37.0. router 32k MTP 54.2 -> 56.4 (ratio 1.153 -> 1.195). 2 GiB captures all of it at 100k; 1 GiB most.
    3. Router-only (MLXVLM Qwen35 + VLM drafter): verify graph build 4.2 -> 9.3 ms and drafter commit build 1.8 -> 6.0 ms from 32k to 100k (text path flat 1.1 / 0.1 ms): an O(context) host step on the VLM path (uncompiled verify, explicit array masks), with the GPU idle. Separate from the pool; overlaps ^66K9Y's .causal-mask card; to file as a follow-up.
    4. CPU load (router 32k, 0/50/100%): round 41.5 / 47.5 / 56.9 ms; verify encode+wait 31.6 -> 40.7 under full load (encode slows), pre 1.3 -> 2.7, verify build 3.9 -> 5.3, commit 1.9 -> 2.9.
    Plan: size the FoundationModels buffer pool from the device's recommended working set (keeps 256 MiB on small devices) so a long-context decode step's freed buffers stay pooled; tests first.
  timestamp: 2026-10-31T00:06:20.884959+00:00
- actor: claude-code
  id: 01m67ts63qaktgegy8grnz5ngk
  text: |-
    Implementation landed (uncommitted), measured on final code (instrumentation removed; logs scratchpad 1abqd92-after-*):
    1. Libraries/MLXFoundationModels/MLXLanguageModel.swift: the process-global MLX buffer pool is `gpuCacheLimitBytes(workingSetBytes:)` = max(256 MiB, recommended working set / 32) instead of a fixed 256 MiB (2.43 GiB here). Tests: Tests/MLXFoundationModelsTests/MLXLanguageModelGPUCacheLimitTests.swift (RED: function missing; GREEN 3/3).
    2. Libraries/MLXLMCommon/KVCache.swift `createCausalMask`: key/query positions from `MLXArray.arange(..., dtype: .int32)` on the device, not `MLXArray(Int32(0) ..< Int32(offset + n))` (an unspecialized generic Array(range) over every cached position; `sample` of the router 100k decode put 492 of 683 verify-build samples and 421 of 437 commit samples there). Characterization tests (behaviour unchanged, so they pass before and after): Tests/MLXLMTests/CausalMaskTests.swift (4).
    Instrumented re-profile with both fixes (probe4): router 32k round 42.1 -> 36.6 ms (verify build 4.18 -> 2.03, commit build 1.77 -> 0.12, discard 1.21 -> 0.19); router 100k 89.8 -> 65.1 ms; text 100k 61.0 -> 51.2 ms.
    After (Release): sweep router MTP/plain tok/s 2k 85.3/55.1, 8k 87.0/53.1, 32k 61.1/46.5 = 1.31x (37.3 ms/round), 65k 59.7/40.5 = 1.48x (53.2), 100k 52.0/36.0 = 1.44x (65.7); text 2k 86.8/56.0, 8k 92.7/55.0, 32k 86.6/49.3 = 1.76x (40.1), 65k 65.8/42.5 = 1.55x (43.6), 100k 54.1/37.1 = 1.46x (50.8). Rounds and effective block sizes identical to the baseline at every point (same tokens). CPU load router 32k: 0% 1.340, 50% 1.288, 100% 1.227 (was 1.159/1.088/1.094); 2k 1.55/1.49/1.38. Round cost router 32k verify rows 1..4 [22.6, 26.7, 36.4, 41.7] (was [22.7, 30.5, 40.4, 46.5]); 100k drafter proposal bs4 13.2 -> 6.7 ms (the round-cost loader also goes through MLXLanguageModel's pool). qwen38FourBitMTPGeneration passed (bs4 pMin0.75 mean 89.7, unchanged: short contexts, own loader); qwen38FourBitMTPSystemPromptCache passed, edit turn cached 2169 prefilled 123, tokens exact.
    Follow-up filed: ^yh9ens5 (compile the router path's VLM verify, 2.0-2.4 ms build left per round). Correction to my earlier comment: the router .causal-mask card is ^8p0rfxx, not "^66K9Y".
    Not done / ruled out: deferring the snapshot-slot discard and pipelining across rounds were unnecessary once the pool keeps the buffers (discard 0.2 ms); speculative commit before the verify read was not pursued (commit build is now 0.1-0.4 ms).
  timestamp: 2026-10-31T01:56:55.543391+00:00
- actor: claude-code
  id: 01m67tsd7ss4nsmxy9gftg33wk
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/MLXFoundationModels/MLXLanguageModel.swift, Libraries/MLXLMCommon/KVCache.swift, Tests/MLXFoundationModelsTests/MLXLanguageModelGPUCacheLimitTests.swift (new), Tests/MLXLMTests/CausalMaskTests.swift (new). `swift build --build-tests` 0 compiler warnings; xcrun xctest all 5 bundles 0 fail 0 skip (MLXLMTests 920 XCTest + 2082 Swift Testing, MLXFoundationModelsTests 460, MLXGuidedGenerationTests 72, CXGrammarTests 7, MLXHuggingFaceMacrosTests 7). Release: sweep/cpuload/round cost/qwen38FourBitMTPGeneration/qwen38FourBitMTPSystemPromptCache all passed; router 32k 1.14x -> 1.31x plain, 100k 1.20x -> 1.44x; 50% load 1.088 -> 1.288.
    - next: /review (task left in doing; nothing committed)
  timestamp: 2026-10-31T01:57:02.841754+00:00
position_column: doing
position_ordinal: '8280'
title: 'MTP: remove the ~18 ms per-round iterator overhead at 32k context (55 ms measured vs ~37 ms of verify + draft GPU work)'
---
From ^tdj3waa iteration 4 (Release, M5 Ultra, Qwen3.8-27B-4bit + MTP-4bit, router path): at 32k context an MTP round measures ~55 ms while its GPU work (verify forward + drafter steps, from MTPRoundCostBreakdownTests) is ~37 ms; host syncs are only 0.22–0.25 ms quiet (0.55–1.05 ms fully loaded). The ~18 ms gap is iterator overhead between GPU work, and it is what decides MTP vs plain at long context and under CPU load (unguarded MTP 0.93x plain at 32k with half the cores busy).

- [x] Profile one MTP round at 32k end to end (signposts / os_signpost + Instruments Metal System Trace, or GPU-synced timers around each phase): graph building, cache bookkeeping (snapshot slots, rewind, KVCacheStorage), drafter commit, sampling/acceptance, array conversions, lazy-eval breaks that serialize CPU and GPU. Done with host timers around every phase plus `sample` of the generation thread, at 32k and 100k on both load paths and at 0/50/100% CPU load (breakdown in the comments). Dominant overheads: (1) the 256 MiB MLX buffer pool of MLXLanguageModel: every round released and re-faulted ~0.6-2 GiB of buffers (discard 1.2-2.2 ms, drafter-step encode 4-6 ms at 100k, slower verifies); (2) router path only: `createCausalMask` built its key positions as a Swift range on the host, through an unspecialized generic `Array(range)` (verify build 4.2 -> 9.3 ms, VLM drafter commit 1.8 -> 6.0 ms from 32k to 100k).
- [x] Remove the dominant overhead (keep the GPU busy: build the next drafter step / verify graph while the previous evaluates; avoid redundant evals and host copies) with output identical to today (greedy up to the near-tie rule, sampled distribution unchanged), cache/drafter alignment intact. MLXLanguageModel sizes the pool to 1/32 of the GPU's recommended working set, at least 256 MiB (2.43 GiB on the M5 Ultra; small devices keep 256 MiB); `createCausalMask` uses device-side `arange`. Both leave every value unchanged: sweep rounds and acceptance are identical before/after (router 32k 449 rounds, eff. block 2.586), prompt-cache edit turn prefills 123, tokens exact.
- [x] Re-measure: MTPRoundCostBreakdownTests, the 2k–100k sweep, and MTPCPULoadTests at 0/50/100% load; target round time close to the GPU work and MTP ≥ plain at 32k under 50% load. Router sweep MTP tok/s 79.1/79.4/52.9/44.1/36.4 -> 85.3/87.0/61.1/59.7/52.0 (2k/8k/32k/65k/100k), ratio at 100k 1.20x -> 1.44x; text 81.9/88.5/75.2/54.7/41.7 -> 86.8/92.7/86.6/65.8/54.1. CPU load router 32k MTP/plain 1.159/1.088/1.094 -> 1.340/1.288/1.227 (round 42.3/44.2/50.8 -> 36.1/37.7/42.1 ms). Router 32k round 36.6 ms vs GPU verify+drafter ~35 ms; residual host piece is the uncompiled VLM verify build (2.0 ms), filed as ^yh9ens5.

#mtp #perf