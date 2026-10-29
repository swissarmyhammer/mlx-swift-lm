---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m641kvkaqew3ma3vd78phf60
  text: 'Router''s real path (from foundationmodelsrouter-ce): FoundationModelsExtras `MLXModelLoader.loadLanguageModel` calls `MLXLanguageModel(configuration:capabilities:speculativeDecoding:weightsLocation:load:)` directly; the load closure uses `loadModelContainer(from:using:configuration:progressHandler:)`, so Qwen3.8 loads as the VLM `Qwen35`, and the drafter is `.matchingTarget(.qwen35Text, .qwen35VLM)` → the VLM drafter. All our real-model MTP benchmarks loaded the target through `LLMModelFactory` (text path). This benchmark must load through `loadModelContainer` + `.matchingTarget` to measure the path SWE-bench actually runs, and compare against the text path.'
  timestamp: 2026-10-29T14:39:23.242644+00:00
- actor: claude-code
  id: 01m6424y7tpbc6jj2e7tp68c6t
  text: |-
    Phase 1 research (worktree reset to mtp 268dab08):
    - `MLXLanguageModel.modelID` is the cache identity of the configuration only, so a text-path model and a router-path (loadModelContainer → VLM Qwen35) model of the same id SHARE one container in the model cache. The benchmark must `releaseAllGPUMemory()` (evictAll) between the two load paths, or the second path silently measures the first one's container.
    - Plain and MTP models of one id also share the prompt-cache key space (modelID + first transcript entry id), so each session needs its own instructions entry (fresh entry id).
    - Executor-level driving (`makeMLXExecutor` + `executeResponse`, as Qwen35AgenticPromptCacheAssessmentTests does) gives one measured pass per turn; a tool call ends the pass, and the driver appends a canned `runCode` tool output. `ExecutorPromptCacheStore.peek` gives the ledger/render; drafter state metadata `cache_start_position` == 0 shows the drafter was resumed from turn 1 (as MTPPromptCacheTests asserts).
    - FM path has no per-token candidates, so the near-tie check needs an independent reference forward of the target over the shared prefix (chunked prefill with a fresh cache, ~35-45k tokens) for the top-2 log probabilities, then `MTPChatTokenMatch` on the first divergent token. Only the first divergence is meaningful: after a near tie the two greedy continuations follow different prefixes.
    - Per-position decode rate: the executor emits one `appendText` chunk per decoded token at most (detokenizer may hold a partial UTF-8 / stop-string token), so chunk timestamps give a lower-bound tok/s per 1024-chunk bucket.
    - MLX peak memory: `Memory.peakMemory` (reset by assigning). Executor ignores capability `.toolCalling`; tools come from `enabledTools`.
    - Card subtask 4 (compare with ACP smoke telemetry for django__django-13964) needs that telemetry; out of phase 1 scope.
  timestamp: 2026-10-29T14:48:43.002533+00:00
- actor: claude-code
  id: 01m642vm7pb5xa5mb40e2v17at
  text: |-
    Phase 1 landed (no model loaded, nothing run on the GPU). Commit a304cf9b on branch worktree-agent-ac12e219c380a9047 (on top of mtp 268dab08).
    - New: IntegrationTesting/.../TextGeneration/MTPSWEBenchScaleTests.swift (32 KB) and MTPSWEBenchScaleTranscript.swift (15 KB, synthetic agent transcript: system prompt + runCode tool, issue, varied Python modules and pytest failures until 31k content tokens).
    - Changed: Qwen38FourBitMTPModel.make(speculates:reasons:path:) with `LoadPath { text, router }`; router = testLoad() (loadModelContainer) + .matchingTarget(.qwen35Text, .qwen35VLM). Default .text keeps existing callers as they were.
    - Test 1 `agentSessionSpeculatesAtSWEBenchScale`: for router then text path (releaseAllGPUMemory between, because both share one container): warm-up, plain and MTP sessions of 4 turns (diagnose, runCode call, short answer, long decode budget 8192) through the executor with a private unlimited-budget prompt-cache store. Logs per pass: prompt/prefilled/cached tokens, prefill tok/s, decode tok/s, stop, accepted/proposed, tokens per verify, drafter start position, fallback, tool call, peak MLX memory, 1024-chunk position buckets. Asserts: turn-1 render >= 30k, every follow-up reuses the previous ledger minus 128 (tool-call seam), long decode >= 4096 tokens, every MTP pass drafted with no passthrough/fallback and drafter start 0, MTP long decode faster than plain, equal fed counts up to the first divergent turn, and the first divergence holds MTPChatTokenMatch against a chunked uncached reference prefill (either side may take the runner-up).
    - Test 2 `decodeRateHoldsFromTwoToThirtyTwoThousandContextTokens` (card subtask 2): router path, no tools, uncached, 3072-token decodes after 2k/8k/32k content tokens; asserts MTP speculates, is faster than plain, and streams >= 2 buckets; logs rates relative to 2k.
    - Limitation found: with tools enabled the executor (runAllowedToolGeneration) streams only reasoning and emits the response text as one chunk at the end, so the position buckets of the tool-path session cover reasoning only. The per-token buckets are exact for the no-tool sweep.
    - Builds: xcodebuild build-for-testing Release → TEST BUILD SUCCEEDED, no Swift warnings; swift build --build-tests → Build complete, 0 Swift warnings (only third-party mlx-swift C++ header warnings and the SwiftPM "missing creator" note).
    - Still open: phase 2 run (bench lock); subtask 4 (compare with ACP smoke telemetry for django__django-13964) needs that telemetry.
  timestamp: 2026-10-29T15:01:06.422194+00:00
- actor: claude-code
  id: 01m642vprtrhz1ed2g5v3afcnt
  text: |-
    ### implement — changed
    - evidence: 3 files — IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/MTPSWEBenchScaleTests.swift, MTPSWEBenchScaleTranscript.swift, Qwen38FourBitMTPModel.swift; commit a304cf9b; xcodebuild Release build-for-testing succeeded, swift build --build-tests 0 Swift warnings
    - next: phase 2 (run once under the bench lock) when the orchestrator says "run phase 2"
  timestamp: 2026-10-29T15:01:09.018807+00:00
- actor: claude-code
  id: 01m6435cew2z8q05tcqgwt59em
  text: |-
    Coordinator add-on (still phase 1, nothing run): commit 32ba1416 adds MTPNearTieDivergenceTests.swift (11 KB). Router saw greedy MTP diverge from plain greedy on the VLM path (148f227e) for "Write the numbers from 1 to 80 in words, separated by commas. Output only the list." (MTP reasoned ~30 tokens longer, ended inside reasoning at 512).
    - Parameterized over LoadPath (.router, .text); no instructions, no tools, greedy, 512 tokens, reasons: true; plain and MTP each through the executor with a private store so the ledger is readable.
    - First divergence = common prefix of the two ledgers. The executor reports no logits, so the test replays plain decode (render minus its last token in one forward, then one token at a time), then forks the cache: a 1-token forward of the last shared token gives plain's row, a 2-token forward [last shared, MTP token] gives the verify row (row 0). Logs token index, both tokens, both top-2 and both gaps, and whether the replay reproduces plain's token; asserts MTPChatTokenMatch with plain's row as the candidates (nearTieLogitTolerance 0.5). A failure message starts "Bug, no near tie:" with the index and both gaps; the check is not loosened.
    - Caveat: the verify row is a reconstruction. The live verify ran on the MTP iterator's own cache and could have been at a row other than 0; the reconstruction keeps plain's cache and the two-row kernel shape.
    - Shared code: SessionRunner/Pass/PassEvents in MTPSWEBenchScaleTests are now internal (ExecutorToolCall renamed from EmittedToolCall, which collided with Qwen35AgenticPromptCacheAssessmentTests), run() takes declaresTool; the top-2 log-prob read moved to Qwen38FourBitMTPModel.topCandidates(of:).
    - Builds: IntegrationTesting Release build-for-testing TEST BUILD SUCCEEDED (no Swift warnings; only the appintents metadata note); swift build --build-tests complete, no Swift warnings.
  timestamp: 2026-10-29T15:06:26.140004+00:00
- actor: claude-code
  id: 01m6435f6dyq6grm87dg5jq9zw
  text: |-
    ### implement — changed
    - evidence: 32ba1416 (on a304cf9b) — MTPNearTieDivergenceTests.swift (new), MTPSWEBenchScaleTests.swift, Qwen38FourBitMTPModel.swift; Release build-for-testing succeeded, swift build --build-tests 0 Swift warnings
    - next: phase 2 run under the bench lock when the orchestrator says "run phase 2" (now covers MTPSWEBenchScaleTests and MTPNearTieDivergenceTests)
  timestamp: 2026-10-29T15:06:28.941360+00:00
- actor: claude-code
  id: 01m6470wh3qkawnm6x73bczgv8
  text: |-
    Phase 1, round 3 (coordinator change), commit 2a3bcb3a. Nothing has run on the GPU.
    - Sweep: `decodeRateFromTwoToOneHundredThousandContextTokens(path:)` runs on the router and text paths. It measures 2k, 8k, 32k, 64k and 100k tokens, running MTP then plain back to back with 512 tokens each. Each point gets its own prompt-cache store; MTP cold-prefills and plain reuses the cached prefix (the reuse is logged). Each line logs MTP/plain ratio, acceptance, tokens per verify, effective block size, ms per round and plain ms per token.
    - New `MTPRoundCostBreakdownTests` (15.5 KB, log category MTPRoundCost, prefix "MTP ROUND COST:") runs on both paths at 32k and 100k tokens. It prefills the target and drafter in 2048-token chunks, then times each part as the median of 8 runs with a GPU sync around each:
      - plain one-token forward;
      - verify forward at 1 to 5 rows;
      - drafter proposal at block sizes 2 to 5, with each chained step's increment;
      - one host sync.
      It also logs the drafter's cache entry count, which tests suspect 1 (drafter attention over the full context). The verify rows against plain show suspect 2 (a slower multi-row SDPA).
    - Build: xcodebuild Release build-for-testing succeeded with no Swift warnings. `swift build --build-tests` is clean.
  timestamp: 2026-10-29T16:13:53.059108+00:00
- actor: claude-code
  id: 01m647vmt166pwwjvx1411ze4v
  text: |-
    Phase 1, round 4 (adaptive MTP guard). Prepared as local WIP commit 646578ad on top of 2a3bcb3a. Not built and not rebased: waiting for the coordinator's sha of the guard landing on mtp. I wrote it against the API in the in-progress guard worktree agent-a24c62ee3c2f35eec:
    - `MTPAdaptivePolicy.default` and `.disabled`;
    - `MLXSpeculativeDecoding(drafterConfiguration:from:using:registration:...adaptivePolicy:)`;
    - the telemetry fields `adaptivePauseCount`, `adaptiveProbeCount`, `adaptivePlainTokenCount`, `adaptiveBlockSizeChangeCount` and `adaptiveBlockSize` (an optional Int).

    That worktree was mid-rebase with conflicts in MLXSpeculativeDecoding.swift (the memoryPolicy/drafterBytes parameters next to adaptivePolicy), so the final init's argument order may differ.

    Changes:
    - `Qwen38FourBitMTPModel.make` takes `adaptivePolicy` (default `.default`).
    - The breakdown uses `.disabled`.
    - The sweep runs three models on both paths at 2k/8k/32k/64k/100k, each with a 1024-token budget. Order: guard off (cold prefill with the drafter), then guard on (reuses the cache), then plain (reuses the cache).
    - It logs both ratios against plain. The guard-on line adds pauses, probes, plain tokens, block size changes and the final block size.
    - It asserts only on guard-on: rate >= plain × 0.97, it drafted, no passthrough or fallback, and it wrote at least 1024 tokens. Plain also gets the token-count and context checks. Nothing is asserted on guard-off.
    - A guard pause sets no fallback or passthrough reason in the guard code, so those checks hold with the guard on.
  timestamp: 2026-10-29T16:28:29.889028+00:00
- actor: claude-code
  id: 01m648qr4j3cbyb096mce0as7r
  text: |-
    ### implement — changed
    - evidence: rebased the 4 commits onto mtp ed863693 (clean, no conflicts). The real `MLXSpeculativeDecoding(…registration:…drafterBytes:memoryPolicy:adaptivePolicy:)` accepts the prepared call unchanged, since it skips the defaulted parameters.
    - xcodebuild Release build-for-testing: TEST BUILD SUCCEEDED. The full log has 1 warning line, the appintentsmetadataprocessor "Metadata extraction skipped" note, and 0 Swift warnings from repo sources. All repo modules and IntegrationTestingTests were recompiled.
    - swift build --build-tests: Build complete. Its only note is llbuild's "missing creator for mutated node" on the Cmlx bundle, which is not a Swift warning.
    - commits: eb4268b9, eff25c2c, 910303f7, 9fac368b (head) on worktree-agent-ac12e219c380a9047. Not pushed.
    - next: wait for "run phase 2". Estimated runtime: sweep ~30–45 min for both paths, breakdown ~10–15 min, session test ~40 min, near-tie test ~10 min; ~1.75–2.25 h total. No GPU work has run yet.
  timestamp: 2026-10-29T16:43:50.802389+00:00
- actor: claude-code
  id: 01m64a9kz8a5w8ba1hr32q6t9d
  text: |-
    ### implement — changed
    - evidence: rebased onto mtp fb641ddf, which includes the guard review fixes in 3b1e122e. The rebase was clean.
    - commits: f9e09ce0, ae5fe3c8, e3e9deee, fe3b25d8 (head).
    - xcodebuild Release build-for-testing: TEST BUILD SUCCEEDED. The log has 0 Swift warnings from repo sources; its only warning line is the appintentsmetadataprocessor note.
    - swift build --build-tests: Build complete, 0 Swift warnings. Its only note is llbuild's "missing creator".
    - the sweep is unguided text, so the new guided stand-down fallback does not touch its guard-on fallback check.
    - next: on "run phase 2", one lock hold running, in order: MTPRoundCostBreakdownTests, the sweep, MTPNearTieDivergenceTests, MTPGenerationTests, then GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache() and GuidedGenerationBenchmarkTests/qwen38GuidedJSONWithMTP(). Uptime before and after each step; the session test is skipped.
  timestamp: 2026-10-29T17:11:04.936312+00:00
- actor: claude-code
  id: 01m64dbt4a6r6b3z1gq1pd2f12
  text: |-
    ### phase 2: run (head fe3b25d8). Lock taken 12:18:22, released 13:03:12.
    Step results (uptime load averages, before → after):
    1. MTPRoundCostBreakdownTests: passed, 446 s. 8.60 → 7.97.
    2. Sweep: passed, 2110 s. 7.97 → 5.41.
    3. MTPNearTieDivergenceTests: failed on the router path, 40 s. 5.41 → 10.98.
    4. MTPGenerationTests: passed, 9 s. 10.98 → 10.22.
    5. qwen38GuidedJSONWithMTP passed; qwen38FourBitMTPSystemPromptCache failed. 10.22 → 6.79.
    A TestudoApp UI-test xcodebuild from another session was running at the same time.

    Breakdown (guard off; ms; verify by rows 1–5; drafter by block 2–5; chained steps):
    - text 32k: plain 21.44; verify 21.85/26.59/35.33/42.48/48.95; drafter 1.42/2.83/4.42/5.90; chained 1.41/1.60/1.47; sync 0.183
    - text 100k: plain 26.42; verify 27.43/36.31/48.60/58.32/68.92; drafter 1.78/8.89/21.30/24.47; chained 7.11/12.41/3.18; sync 0.249
    - router 32k: plain 22.42; verify 22.66/30.79/40.43/45.84/53.94; drafter 1.54/3.19/4.56/6.04; chained 1.65/1.36/1.49; sync 0.251
    - router 100k: plain 27.59; verify 28.88/43.59/57.25/70.02/81.98; drafter 1.99/8.77/19.09/22.31; chained 6.78/10.32/3.21; sync 0.182
    - The drafter cache holds the full context (32768 and 100000 entries).

    Sweep (tok/s; MTP/plain ratio):

    | path, context | plain | guard off | guard on | guard on: pauses / probes / plain tokens / block changes / final block |
    |---|---|---|---|---|
    | text 2k | 55.2 | 88.0 (1.59) | 84.8 (1.53) | 1/5/84/6/5 |
    | text 8k | 54.4 | 95.3 (1.75) | 82.8 (1.52) | 2/4/144/3/2 |
    | text 32k | 47.8 | 74.6 (1.56) | 60.6 (1.27) | 3/6/216/3/2 |
    | text 64k | 38.3 | 52.4 (1.37) | 43.5 (1.14) | 5/3/460/3/2 |
    | text 100k | 32.3 | 43.7 (1.35) | 33.9 (1.05) | 5/3/588/3/2 |
    | router 2k | 55.3 | 76.3 (1.38) | 65.3 (1.18) | 5/5/448/3/2 |
    | router 8k | 53.5 | 76.3 (1.43) | 60.2 (1.13) | 7/3/460/3/2 |
    | router 32k | 46.8 | 55.6 (1.19) | 48.1 (1.03) | 4/2/796/3/2 |
    | router 64k | 37.3 | 48.6 (1.30) | 37.8 (1.01) | 4/3/671/3/2 |
    | router 100k | 30.3 | 37.9 (1.25) | 29.8 (0.99) | 4/2/782/3/2 |

    Guard-off acceptance, tokens per verify and effective block, by point:
    - text: 0.88/2.92/3.18, 0.96/3.75/3.87, 0.93/3.39/3.58, 0.91/3.01/3.21, 0.90/3.26/3.51
    - router: 0.85/2.78/3.09, 0.94/3.76/3.93, 0.83/2.43/2.72, 0.95/3.84/3.97, 0.94/3.95/4.13

    Guard-off ms per round: text 33.2/39.4/45.4/57.5/74.7; router 36.4/49.3/43.8/78.9/104.3. Plain ms per token: text 18.1/18.4/20.9/26.1/31.0; router 18.1/18.7/21.4/26.8/33.0.

    Findings:
    - (a) Guard on is below guard off at all 10 points, by 4–25%. It drops to block size 2 and decodes up to 796 of 1024 tokens plainly, even though guard off beats plain everywhere (1.19–1.75). The worst point is router 100k at 0.986 of plain, 1.6% above the 0.97 floor. This needs a follow-up on the guard (^tdj3waa).
    - (b) A one-row verify costs about the same as plain decode. Each extra row costs ~6.8 ms at 32k and ~10–13 ms at 100k, so a 5-row verify is 2.3–3.0x a plain forward and is the largest part of a round. Suspect 2 is confirmed as the main cost.
    - (c) Drafter chained steps stay ~1.5 ms at 32k but jump to 3–12 ms each at 100k, so a full block-5 proposal is 22–24 ms. Suspect 1 is confirmed as the secondary cost.
    - (d) Near-tie, router path: first divergence at generated token 74 (ledger 148). Plain wrote 58614, MTP wrote 16570. The top-2 gap is 0.0 on both the plain row and the verify row, an exact tie. The replay ranks 16570 first, but plain decode wrote 58614 ("replay reproduces plain false"), so MTPChatTokenMatch fails on how the exact tie was ordered. It is a near tie by the gap. Text path: token 88, gap 0.125 on both rows, holds.
    - (e) qwen38FourBitMTPSystemPromptCache, follow-ups: tokens exact on every turn, but turn 1 is MTP 51.8 vs plain 56.0 tok/s (0.92x, 102 tokens) and turn 2 is 46.0 vs 55.9 (0.82x, 117 tokens). Turn 0 was 1.60x. The edit and thinking scenarios ran 1.44–1.77x.
    - Plain never reused the MTP cache in the sweep (cached 0 at every point), so it prefilled cold; decode rates are unaffected. promptTime in the sweep lines is not a prefill time.
  timestamp: 2026-10-29T18:04:42.506922+00:00
position_column: doing
position_ordinal: '8180'
title: 'MTP: measure Qwen3.8 MTP at SWE-bench scale (tens of thousands of context tokens, ~16k-token decodes, tool turns)'
---
The ACP agent's SWE-bench run (via Router → Extras MLXModelLoader → MLXLanguageModel, Qwen3.8-27B-4bit + Qwen3.8-27B-MTP-4bit) has multi-turn sessions with contexts of tens of thousands of tokens, hundreds of generations, and decodes up to ~16000 tokens, almost all with a `runCode` tool call. Our MTP benchmarks only cover ~2.2k-token prompts and 192-token decodes.

- [ ] Release integration benchmark: MLXLanguageModel with MTP, a ~32k-token prompt (tool schema + long history), follow-up turns, and a long decode (≥4k tokens); report MTP vs plain tok/s, acceptance, tokens per verify over the decode, prefill tokens per follow-up (only the new message), peak memory incl. drafter cache.
- [ ] Check speed does not decay with context length (tok/s at 2k / 8k / 32k positions) and no fallback occurs.
- [ ] Use the shared bench lock; do not run during the ACP smoke run.
- [ ] Compare with the ACP smoke run telemetry (django__django-13964) once available.

#mtp #foundation-models #benchmark