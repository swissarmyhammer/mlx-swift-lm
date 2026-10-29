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