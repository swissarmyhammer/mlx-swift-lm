---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g9fm6ys21ykq5k09fefdzh
  text: |-
    ### Requirement (foundationmodelsacpagent-a1, 2026-11-27)
    This affects single-Mac bench runs too: every SelectionTier call re-prefills its ~7.8k prefix on one Mac, which is 66 × ~3 s in an Inkling run. Test both paths, local and clustered: a second guided call with the same prefix has tokens.cached ≥ the prefix length, and its output is unchanged.
  timestamp: 2026-11-28T05:19:10.046813+00:00
- actor: claude-code
  id: 01m8g9rcxse0exfq3rzrqk3wvz
  text: |-
    ### Research (implement)
    - Why the bypass exists: 08a120e9 (^z1j7f67) — `GuidedGenerationLoop.run` built its own KV cache (GuidedTarget.init → model.newCache) and accepted none from a caller, so a guided pass could not reuse a prefix. d841f85c (^1wncmxq) then made `carriesNoCache()` also DROP the carried entry: before that, the guided pass left the slot holding the OLD checked-out cache and the response checked that stale cache back in, whose ledger did not record the guided pass. Handling the reason = the guided loop must take the plan's caches/state/drafter continuation AND hand back what it fed (emitted tokens + final caches + prefill state), so the slot commits a ledger that matches the caches.
    - Latent bug found: local GuidedSpeculation.speculate() returns before `target.keepVerified` when the generation stops inside a round (decide/accept returns nil). The cache then still holds the round's rejected drafts and the timeline is not committed. Harmless while the cache is thrown away; with check-in it would label wrong tokens. Must keep only the kept positions at the end. Cluster path already trims (trimSpeculated, ^9mvg23k).
    - GuidedTarget.prefill passes `state: nil` to prepare / the final forward; a carried model state must seed them (TokenIterator passes its state).
    - Plan machinery to reuse: ExecutorPromptCacheSlot.plan / ExecutorPromptCachePlan (make → extend/splice/rewind/restore/rebuild, prefillingToCheckpoint for transcriptBoundary < prompt end, prefillDidEnd for prompt-end checkpoint), ExecutorGenerationHandOff-style commit with generationCache (quantized KV replaces cache objects), usageInput.
    - Phase 1 of think-then-call (runToolCallReasoningPhase) is a plain unconstrained generation with its own cache; not on the guided loop.
  timestamp: 2026-11-28T05:23:57.497151+00:00
- actor: claude-code
  id: 01m8gaz6kzhg5h5xp1zkygk3n3
  text: |-
    ### Implementation landed (not committed)
    - MLXGuidedGeneration: new `GuidedPromptCache.swift` (package `GuidedCarriedCache`, `GuidedGeneratedCache`, `GuidedCacheHandOff`). `GuidedGenerationLoop.run` (package overload) takes `carried:` + `handOff:`; `GuidedTarget.init(carried:)` adopts the plan's caches + model state and prefill/first forward now pass that state (was `nil`); the local drafter feed starts at `processedTokenCount` from the plan's drafter continuation; the loop hands over the prefill state and then `(countedTokens, target.cache)` before `finish()` (so an out-of-budget pass checks in too). `GuidedDecoder.countedTokens` records every counted token.
    - Bug fixed (RED first in GuidedGenerationCacheHandOffTests, budget 6/threeActivities): `GuidedSpeculation.speculate` returned before `keepVerified` when the generation stopped inside a round, leaving rejected drafts / uncounted forced tokens in the main cache and the timeline uncommitted. It now keeps only the counted verified positions on every exit.
    - MLXFoundationModels: `runGuidedLoop` plans the slot inside the guided section (so checkpoint feeds of a ClusterLanguageModel stay under `withoutCancellingOnFailure`), runs the loop on the plan, and `ExecutorGuidedHandOff` commits `plan.holding(generationCache:)` with the counted tokens and the prefill state. No commit on a throw or a cancelled respond (same as plain). Schema and required-tool passes pass a transcriptBoundary (schema: `GuidedPrompt` from `schemaInput`), so hybrid models take the same checkpoints as plain passes. Usage now reports cached tokens (required turn subtracts opener + reasoning from the fed count).
    - Removed `carriesNoCache()`, `ExecutorPromptCacheReport.guidedLine` and their two tests; the three guided tests of ExecutorPromptCacheCheckInNothingTests (which asserted the bypass) are replaced by GuidedPassPromptCacheTests (scripted: plain→schema, schema check-in, schema→plain, schema→schema, plain→required, spilled→required, copy).
    - Real-model GuidedPromptCacheReuseTests (local + clustered): second schema turn cached ≥ first turn's prompt tokens, same text as cold; speculative (incl. plain→speculative schema, drafter state resumed); 8-bit KV; required after plain; plain after schema; copy; cluster MTP every rank offset == ledger; member loss checks nothing in.
    - Guided passes check in no drafter continuation (next turn cold-starts the drafter, output unchanged) → follow-up ^33h7w11. Think-then-call phase 1 still prefills cold (not on the guided loop) → follow-up ^8mvkb7t.
    - README (MLXFoundationModels) updated: guided turns now use the prompt cache.
  timestamp: 2026-11-28T05:45:08.991158+00:00
- actor: claude-code
  id: 01m8gberdj2rwmmfq8fy6pazrp
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests clean (0 warnings); swift-format run; xctest full bundles in one process each: MLXLMTests 2845 Swift Testing + 1261 XCTest passed (3 pre-existing known issues: AttentionKernelToleranceTests near-tie, MiniCPM5 CDATA), MLXFoundationModelsTests 701 passed, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11. New tests: GuidedPassPromptCacheTests (7, scripted), GuidedPromptCacheReuseTests (11, local+clustered), GuidedGenerationCacheHandOffTests (16 cases). Why the bypass existed: GuidedGenerationLoop built its own KV cache (08a120e9), and d841f85c dropped the carried entry so a stale ledger was not checked back in; now the loop takes the plan's cache and hands back counted tokens + caches, so the slot commits a matching ledger.
    - next: /review. Follow-ups ^8mvkb7t (think-then-call phase 1 cold prefill), ^33h7w11 (guided drafter continuation hand-over).
  timestamp: 2026-11-28T05:53:38.738850+00:00
- actor: claude-code
  id: 01m8gc03vpgb15x5cjqk04zr9r
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (no warnings from sources); xcrun xctest per bundle, all exit 0: MLXLMTests 2845 tests/281 suites passed; MLXGuidedGenerationTests 96/23 passed; MLXFoundationModelsTests 701/94 passed; CXGrammarTests 7/5 passed; MLXHuggingFaceMacrosTests XCTest 0 failures (7+3+11+11 executed, suites in the log). swift-format lint on changed files clean.
    - MLXLMTests reports 3 known issues, from withKnownIssue in AttentionKernelToleranceTests and ToolTests (MiniCPM5 CDATA). Tests/MLXLMTests has no working-tree changes, so these predate this task.
    - next: review
  timestamp: 2026-11-28T06:03:07.510383+00:00
- actor: claude-code
  id: 01m8gcdb89p66jre03pwrnh09w
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (9bba69fc): 0 findings (0 confirmed, 4 refuted, 28 attempted, 0 failed); 19 files reviewed, 9 unmatched (.kanban task files, Libraries/MLXFoundationModels/README.md)
    - next: none — task moved to done
  timestamp: 2026-11-28T06:10:21.065904+00:00
- actor: claude-code
  id: 01m8gcdp8gggfqxqch6y31kjfa
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — guided loop takes the carried cache and hands back tokens and caches; bypass removed; 3 new test files; follow-ups ^33h7w11 and ^8mvkb7t
    - test: green — all 5 bundles (MLXLMTests 2845, MLXFoundationModelsTests 701, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 0 failures)
    - commit: 9bba69fc
    - review: clean — 0 findings
  timestamp: 2026-11-28T06:10:32.336453+00:00
position_column: done
position_ordinal: ffffff9180
title: 'Prompt cache: guided passes take the carried cache for the common prefix and check their ledger in'
---
## What
Guided passes (`respond(to:generating:)` and required-tool turns on the guided loop) never use the prompt cache. ExecutorPromptCacheSlot.carriesNoCache() logs `rule=guided (the guided pass owns its cache and carries none)`. The guided loop takes no carried cache and checks nothing in, so a guided session never reuses a prefix. A guided session's cache also can't be copied, because the source is never in the store.

Bench cluster6i (Router 7bd4949f, mlx 35285ea9, Inkling on 2 Macs): all 36 guided selection calls of about 8.4k tokens had tokens.cached=0. They repeat an instructions prefix of about 7.8k tokens, and their ~3 s of prefill is most of each call. Reported by foundationmodelsrouter-22.

## Fix
- The guided pass checks out the carried cache like a plain pass does. It reuses the longest common prefix under the same plan rules (trim, checkpoints), prefills only the rest, and after the pass checks in its ledger: prompt plus generated tokens. A grammar mask changes sampling, not the KV.
- This must work with:
  - guided speculation on the cluster (^9mvg23k): every rank's offset equals the ledger;
  - local guided MTP, including the drafter state;
  - quantized KV (^2ry6yz7);
  - RemoteKVCache entries and the worker budget (^xz9zxyp, ^xqgcs5s);
  - a lost member (^axrb0cw).
- Output must be the same as the current guided pass with no cache.
- Remove the `rule=guided` bypass, or keep it only where a cache truly can't be carried, with a precise reason.

## Tests
- A second guided turn whose prompt shares a long prefix with the first reports cached tokens > 0 and the same output as a cold guided turn. Cover this locally and on the cluster.
- A guided turn after a plain turn of the same session reuses the plain turn's cache, and the reverse also works.
- A guided session's cache can be copied with copyPromptCache, and the copy's next turn reuses the prefix.
- A guided cluster turn with MTP leaves every rank's offset equal to the ledger.
#cluster