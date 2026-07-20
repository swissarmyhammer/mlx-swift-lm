---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxy1aafb0j5r38yqpc1dm63a
  text: |-
    INVESTIGATION (traced by reading real code, verified statically):

    DISPATCH: The failing test uses makeTestModel(qwen36HybridModelID) with defaultTestCapabilities() = [.guidedGeneration, .toolCalling] — NO .reasoning. Request has no tools/schema. So dispatchGeneration routes to runTextGeneration(reasoningSetup: nil) -> runUnconstrained. The ACTUAL path is runUnconstrained -> generate(...) (decoded text) -> commitPromptCache(emittedText:tokenizer:). This is the emittedText/re-encode path, NOT runReasoning/runToolCallReasoningPhase as the task assumed. (Those reasoning paths have the identical latent bug but the test does not exercise them.)

    EXACT DRIFT MECHANISM (confirmed): TokenIterator.next() (Evaluate.swift) uses "return previousY, prefetch next" pipelining: each next() call fter prefill feeds previousY via step() (advancing cache.offset by 1) and returns previousY. After N next() calls returning [T0..T_{N-1}], cache.offset == P + N and observed count == N. So:
    - maxTokens cutoff: offset advance == observed count EXACTLY. NO DRIFT. (Contradicts the task's claim that round 1 hit maxTokens and still drifted — that claim is wrong. maxTokens stores fine.)
    - EOS/stop termination: in generateLoopTask's stop branch (includeStopToken==false, the default for both generate and generateTokens), iterator.next() already RETURNED the EOS token — meaning step() already FED it (offset advanced) — then the loop calls discardGeneratedToken() (a no-op for TokenIterator) and breaks WITHOUT emitting it. So cache.offset = P + K + 1 while emitted/observed content = K. Drift = +1. The extra token IS the EOS/stop token, whose identity is known at the discard site (the `token` local).

    WHY THE TEST FAILS: "Say hi in one word" with maxTokens=8 terminates via EOS (im_end) well under budget. emittedText path: actualGeneratedCount = K+1, reencoded(text)=K, reconcileGeneratedTokens hits the -1 case -> trustedTokens=K. Downstream commitPromptCache(generatedTokenIDs:) sees cacheAdvance=K+1 vs observed=K -> .trimCacheByOne -> trimCacheIfValid -> canTrimPromptCache(hybrid)==false (MambaCache.isTrimmable==false) -> DROP. Round 2 cachedTokenCount==0. Test fails. This IS the EOS-trim degradation, just reached through the emittedText path rather than a raw-token path.

    SOUND FIX (verified against resolveHybridCheckpoint semantics): resolveHybridCheckpoint does EXACT-PREFIX match (element-by-element) with NO trimming — it restores the whole checkpoint and feeds the remaining suffix. So storing the checkpoint keyed on the FULL true sequence [promptTokens + generatedTokenIDs + [EOS]] (matching the cache's real offset P+K+1, so snapshotHybridCheckpoint's offset==tokens.count check passes) is SOUND and REUSABLE precisely because the EOS is Qwen's <|im_end|>, which is exactly the token the chat template places after the assistant reply in round 2 — so the stored sequence is a genuine prefix of round 2's render. If EOS != im_end, resolve simply finds no match (same as today's drop) — never corruption. Plan: surface the discarded stop token via GenerateCompletionInfo (.info event reaches both generate and generateTokens consumers), thread it into commitPromptCache, and in the .trimCacheByOne case when the cache is NOT trimmable but the extra token is known, store keyed on the extended sequence instead of dropping.
  timestamp: 2026-07-19T20:33:09.611891+00:00
- actor: claude-code
  id: 01kxy8pbfxxmp4pdkgaevk2n6m
  text: |-
    REAL-MODEL VERIFICATION overturns the task's diagnosis. I ran the actual PromptCacheHybridReuseTests against the real mlx-community/Qwen3.6-27B-mxfp4 with instrumentation. Hard evidence:

    COMMIT SIDE IS FINE (no one-ahead drift on maxTokens). Round 1: finalOffset=27, promptTokens=19, generatedTokenIDs.count=8, cacheAdvance=8, reconciliation=.matches, plan=store. Round 2 likewise plan=store. Both rounds STORE a checkpoint successfully. first.outputTokenCount=8=maxTokens (budget cutoff, NOT EOS; stopTokenFedToCache=nil). So the task's premise -- "maxTokens round shows the same one-ahead drift" -- is WRONG. My static analysis was right: maxTokens advance == observed, .matches, stores.

    THE REAL FAILURE IS ON THE RESOLVE SIDE, and it is a chat-template issue, not a Mamba/attention or EOS issue. resolveHybridCheckpoint debug (round 2, newLength=46, stored candidateLength=27):
      stored (round1): [...74455('assistant'), 198('\n'), 248068('<think>'), 198('\n'), 8160('Here'), 579, 264, 7047, 1817, 25, 271, 16]
      round2 newTokens: [...74455('assistant'), 198('\n'), 8160('Here'), 579, 264, 7047, 1817, 25, 271, 16, 248046('<|im_end|>'), ...]
      commonPrefix = 17.
    Decoded token IDs (from tokenizer.json): 248045=<|im_start|>, 248046=<|im_end|>(eos), 248068=<think>, 248069=</think>, 74455='assistant', 8160='Here', 198='\n'.

    ROOT CAUSE: Qwen3.6's chat_template.jinja injects `<|im_start|>assistant\n<think>\n` on the GENERATION PROMPT (add_generation_prompt branch, lines 147-152), so round 1 physically feeds `<think>\n` (248068,198) into its cache at positions 17-18. But when round 2 renders that same assistant turn as HISTORY, the template strips reasoning for any assistant turn that is NOT after the last user query (lines 100-102: `if preserve_thinking or loop.index0 > ns.last_query_index ... else '<|im_start|>assistant\n' + content`). In round 2's [user1, assistant1, user2] transcript, assistant1 is before user2, so it renders WITHOUT `<think>`. Result: round 1's cache = [17 common] + [<think>,\n] + [8 reasoning]; round 2 = [17 common] + [8 reasoning as content] + [<|im_end|>...]. They diverge at position 17, and because KV positions are ABSOLUTE and resolveHybridCheckpoint requires an EXACT full-prefix match (Mamba state can't be partially reused), reuse = 0.

    CONSEQUENCE: the genuinely-reusable cross-turn prefix for Qwen3.6 is only the pre-`<think>` region (17 tokens here). The test asserts cachedTokenCount >= sharedPrefixTokens - 1 = (19 prompt + 8 output) - 1 = 26. 17 < 26, and the 8 reasoning tokens sit at DIFFERENT absolute positions in round 1 (19-26) vs round 2 (17-24) due to the missing `<think>\n`, so they can never be reused. THE ASSERTION IS UNSATISFIABLE FOR QWEN3.6 AT THE CACHE LAYER, by any sound implementation. This is a fundamental thinking-model chat-template / KV-cache interaction (attention's Qwen2.5-3B test passes only because Qwen2.5 has no `<think>` scaffold, so generation-prompt and history render identically).

    WHAT I IMPLEMENTED (sound, kept, but does NOT fix this test): recovered the fed-but-uncounted terminal stop token via GenerateCompletionInfo.stopTokenFedToCache (Evaluate.swift generateLoopTask), threaded it into commitPromptCache, and added PromptCache.planCacheStore so an EOS-terminated hybrid round stores the EXTENDED [prompt+generated+stopToken] sequence instead of dropping. This fixes the real EOS-terminated-hybrid latent bug the task DESCRIBED, is unit-tested (PromptCacheStorePlanTests, 5 cases), keeps full `swift test` green (610 tests), and keeps PromptCacheReuseTests (attention) green. But this test's failure is the resolve-side `<think>` divergence above, which no commit-side fix can address.

    BLOCKED / NEEDS DIRECTION: I cannot weaken the assertion (forbidden), cannot mark the model unsupported (forbidden), and the assertion is provably unsatisfiable for Qwen3.6 because its template strips `<think>` from history. Escalating to the user rather than making a unilateral scope decision.
  timestamp: 2026-07-19T22:42:03.901217+00:00
- actor: claude-code
  id: 01kxyaqc3gpvk5d9xx09pp1h5f
  text: |-
    STATE OF THE TREE (task left in doing; BLOCKED on user direction):

    Verified green:
    - Full `swift test`: 610 tests, 0 failures (includes new PromptCacheStorePlanTests, 5 cases).
    - Integration PromptCacheReuseTests (pure-attention Qwen2.5-3B regression): PASS.

    Still failing (the target):
    - Integration PromptCacheHybridReuseTests (real Qwen3.6-27B): FAILS, cachedTokenCount=0, for the fundamental resolve-side `<think>` chat-template reason documented in the prior comment. Unsatisfiable at the cache layer.

    Changes kept (sound, additive, do NOT resolve the target test):
    - Evaluate.swift: GenerateCompletionInfo.stopTokenFedToCache + generateLoopTask captures the discarded terminal stop token.
    - PromptCache.swift: PromptCache.planCacheStore + CacheStorePlan (pure, unit-tested) -> a non-trimmable hybrid round with a known fed stop token stores the EXTENDED [prompt+generated+stopToken] sequence instead of dropping (fixes the EOS-terminated-hybrid latent bug the task DESCRIBED).
    - MLXLanguageModel.swift: threaded stopTokenFedToCache through both commitPromptCache overloads and all three callers (runUnconstrained, runReasoning, runToolCallReasoningPhase); commitPromptCache now uses planCacheStore.
    - PromptCacheChunks.swift: cacheAdvanceOffset doc rewritten from "KNOWN, ACCEPTABLE DEGRADATION (dropped)" to describe the new store-extended behavior.
    - PromptCacheTests.swift: added PromptCacheStorePlanTests.
    - Reverted my edits to PromptCacheHybridReuseTests.swift back to ^1fjmy9f's original (they had optimistically claimed the fix worked; it doesn't). Minor: that test's comment still references cacheAdvanceOffset's old "KNOWN, ACCEPTABLE DEGRADATION" phrase which my doc rewrite removed.

    WHY I DID NOT force the test green: the ONLY ways to reach cachedTokenCount>=26 are (a) weaken the assertion or (b) make round 2's render reproduce round 1's fed `<think>\n`+reasoning (i.e., preserve reasoning in transcript history / preserve_thinking). Both are forbidden or out of scope, and no cache-layer change can help because KV positions are absolute and the reusable common prefix is genuinely only 17 tokens.

    OPTIONS FOR THE USER (need direction):
    1. Accept that thinking-model (Qwen3.6) cross-turn KV reuse is fundamentally limited to the pre-reasoning prefix, and revise the test's expectation to the genuinely-reusable prefix (would change the >=26 bound -- currently forbidden).
    2. Change the test scenario so reuse is achievable (e.g., disable thinking so no `<think>` scaffold divergence; or replay history that reproduces the fed sequence).
    3. Pursue a deeper product change: preserve reasoning in replayed assistant history (preserve_thinking) so round 2's render matches round 1's cache -- larger than this task's scope and changes transcript semantics.
    4. Implement partial-prefix hybrid reuse (reuse the 17-token common prefix; currently 0) -- a genuine improvement, but still < 26, so the test still fails.

    I recommend (1)/(2): the test as written encodes an attention-model assumption that is false for a thinking model whose template strips reasoning from history.
  timestamp: 2026-07-19T23:17:34.448823+00:00
- actor: claude-code
  id: 01kxzmv7n21g5qj6p3v2dmm3vh
  text: |-
    FINAL RESOLUTION -- verified green against the real model.

    User direction received: "Lower the bound to match reality" -- i.e. correct `PromptCacheHybridReuseTests.swift`'s assertion rather than keep chasing an unsatisfiable one, with a doc comment explaining why.

    Re-verification (this session, empirical, not copied from the prior comment): re-ran the real test against `mlx-community/Qwen3.6-27B-mxfp4` with temporary debug instrumentation in `PromptCache.resolveHybridCheckpoint` (added, captured evidence, then fully removed -- `git diff` on `PromptCache.swift` is back to exactly the pre-existing 71-line landed fix). Confirmed:
    - Round 1 stores a clean 27-token checkpoint (19 prompt + 8 output), reconciling as `.matches` (it hits `maximumResponseTokens`, not EOS) -- the EOS-trim fix landed in this task isn't even exercised by this particular test.
    - Round 2's re-rendered prefix and round 1's stored checkpoint share their first 17 tokens, then diverge exactly at the `<think>` token: round 1's checkpoint has `<|im_start|>assistant\n<think>\n` (Qwen3.6's template always opens `<think>` for the turn being generated); round 2 re-renders that same turn as history WITHOUT `<think>` (the template strips reasoning from any assistant turn that precedes a newer user turn).
    - IMPORTANT CORRECTION to the prior investigation's framing: `resolveHybridCheckpoint` requires an EXACT WHOLE-checkpoint prefix match (no partial reuse -- Mamba state can't be sliced). Since the divergence lands INSIDE the stored checkpoint (position 17 of 27), the match fails completely and `cachedTokenCount` is empirically `0` -- NOT ~16-17 as a naive "lower the bound to the common-prefix length" reading of the prior comment would suggest. The 17-token "common prefix" is a diagnostic quantity (where the mismatch begins), not an amount that gets partially credited by the current all-or-nothing mechanism.

    Updated `PromptCacheHybridReuseTests.swift`: rewrote the file header and the final assertion. New assertion is `second.cachedTokenCount == 0`, with an extensive comment explaining the exact chat-template mechanism (live-generation `<think>` scaffold vs. history-replay stripping), why it's unrelated to the EOS-trim bug this task fixed (round 1 here reconciles as `.matches`, not `.trimCacheByOne`), and why hybrid's all-or-nothing matching means 0, not partial credit.

    Verified real-model green: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheHybridReuseTests` -- PASSED.

    Regressions verified green:
    - `-only-testing:IntegrationTestingTests/PromptCacheReuseTests` (pure-attention) -- PASSED.
    - `swift test` (full, unfiltered) -- 610 tests, 0 failures (265+80+258+7, matching the previously-recorded baseline).

    Task left in `doing` for `/review`; not moved to review/done by this session. `^1fjmy9f` updated in parallel to match (left in `doing`).
  timestamp: 2026-07-20T11:33:41.154690+00:00
depends_on:
- 01KXX2JDFWQ79CXZE1P1FJMY9F
position_column: doing
position_ordinal: '8180'
title: 'Fix root cause: hybrid prompt-cache always drops (TokenIterator lookahead advances cache one step beyond observed tokens)'
---
## What

Task `^1fjmy9f`'s new real-model integration test (`PromptCacheHybridReuseTests.swift`) proved that hybrid Mamba/attention prompt-cache reuse (task `^r9rf5g7`) never actually engages against a real `mlx-community/Qwen3.6-27B-mxfp4` session: `supportsPromptCacheReuse == true`, but round 2's `cachedTokenCount` is `0` when the test's ORIGINAL assertion expected `>= 26`. Investigation (this task) found TWO separate things bundled together under that one failure, and resolved them differently:

1. **A real, sound bug** in the hybrid checkpoint STORE path: an EOS-terminated hybrid round whose cache lands one token ahead of its observed generated-token count (`PromptCache.reconcileCacheAdvance`'s `.trimCacheByOne` case) used to be DROPPED entirely, because `MambaCache.isTrimmable == false` prevents the trim-and-store pure attention gets. **This is fixed** (see Resolution below).
2. **A separate, unrelated, structural limitation** on the RESOLVE side: Qwen3.6's chat template strips the `<think>` reasoning scaffold from an assistant turn once it becomes conversation history, which guarantees a hybrid checkpoint (all-or-nothing exact-prefix match, no partial reuse) can never match round 2's re-render for this test's two-round "replay-then-ask-again" shape. **This is not a bug and cannot be fixed at the cache layer** — the test's assertion has been corrected to reflect this reality instead.

## Root cause (as originally traced, before real-model verification)

`TokenIterator.next()` uses a \"return previous, prefetch next\" pipelining design (`Evaluate.swift` ~line 744-759): each call captures `previousY`, calls `step(previous: previousY)` (which FEEDS `previousY` through the model, physically advancing `cache.offset`/Mamba state by exactly 1), then returns `previousY`. In the normal per-call case this is exactly 1:1 (fed token == returned token, no drift).

For the **EOS-terminated case** (`generateLoopTask`, same file): when `iterator.next()` returns a stop/EOS token, `step()` has ALREADY fed that EOS token through the model (advancing the cache) BEFORE the stop-check runs — but the outer loop's own `tokenCount` (which feeds `generatedTokenIDs`) is only incremented in the non-stop branch, so it never counts the EOS token. Net effect: `cache.offset` ends up exactly 1 token ahead of `generatedTokenIDs.count`. The extra token's identity IS fully known — it's the literal stop-token id already in hand at the discard site.

**maxTokens-terminated rounds do NOT drift** (verified against the real model): `cache.offset` advance exactly equals the observed token count, reconciling as `.matches` and storing cleanly with no fix needed. The task's original premise that "round 1 hit maxTokens and still drifted" was based on an unverified assumption and turned out to be wrong once actually run against the real model — see Resolution.

## Resolution (what was actually implemented and verified)

- **Fix landed**: `GenerateCompletionInfo.stopTokenFedToCache` (`Libraries/MLXLMCommon/Evaluate.swift`) surfaces the discarded terminal stop token's identity from `generateLoopTask`. `PromptCache.planCacheStore` (`Libraries/MLXFoundationModels/PromptCache.swift`) is a new pure, unit-tested decision function: for a non-trimmable (hybrid) round whose reconciliation is `.trimCacheByOne`, if the fed stop token is known it now returns `.storeExtended(fedStopToken:)` — storing the TRUE, extended `[prompt + generated + stopToken]` sequence (matching the cache's real physical offset) instead of dropping. Threaded through both `commitPromptCache` overloads and all three callers (`runUnconstrained`, `runReasoning`, `runToolCallReasoningPhase`) in `Libraries/MLXFoundationModels/MLXLanguageModel.swift`. `PromptCache.cacheAdvanceOffset`'s doc comment in `Libraries/MLXFoundationModels/PromptCacheChunks.swift` rewritten from \"KNOWN, ACCEPTABLE DEGRADATION (dropped)\" to describe the new store-extended behavior. Unit-tested in `Tests/MLXFoundationModelsTests/PromptCacheTests.swift` (`PromptCacheStorePlanTests`, 5 cases).
- **Real-model finding, NOT fixable at the cache layer**: `IntegrationTesting/.../PromptCacheHybridReuseTests.swift`'s round 1 terminates by hitting `maximumResponseTokens` (a budget cutoff, not EOS) and reconciles as `.matches`, storing a clean 27-token checkpoint (19 prompt + 8 output) regardless of the fix above — so the EOS-trim fix is not even exercised by this test. Empirically verified (temporary debug instrumentation in `resolveHybridCheckpoint`, added and removed) that round 2's re-rendered prefix diverges from that stored checkpoint at token 17: round 1's checkpoint has `<|im_start|>assistant\n<think>\n` (the template unconditionally opens `<think>` for the turn being generated), round 2 re-renders that same turn as history WITHOUT `<think>` (Qwen3.6's template strips reasoning from any assistant turn that isn't the response to the most recent user query). Because `resolveHybridCheckpoint` only matches on an EXACT WHOLE-checkpoint prefix (Mamba state admits no partial/truncated reuse), that divergence drops the entire checkpoint — `cachedTokenCount` is deterministically and structurally `0` for this scenario, not close to the 17-token common prefix. The test's assertion has been corrected accordingly (`== 0`, with a full explanation of the mechanism) rather than weakened to a meaningless `> 0` — see `^1fjmy9f` for the corresponding test-file update.

## Why \"just trim\" or \"just store anyway\" were NOT valid fixes (still true)

- Hybrid stacks can't trim (`MambaCache`/`ArraysCache.isTrimmable == false`) because Mamba's recurrent state is a collapsed running summary with no per-position history.
- Storing a checkpoint keyed on a SHORTER observed-token sequence than the cache's true physical state would silently misalign RoPE positions and Mamba state on later restore — a correctness bug, worse than not caching. The landed fix avoids this by keying on the TRUE extended sequence instead.

## Acceptance Criteria

- [x] Read and document the EXACT generation loop `respond()` uses for a reasoning-capable model like Qwen3.6 and confirm precisely where/why a round's cache offset can land ahead of `generatedTokenIDs.count`. (Confirmed: only the EOS-terminated case drifts; maxTokens-terminated rounds reconcile as `.matches` with no drift — the original premise that maxTokens also drifted was wrong.)
- [x] Implement a sound fix that lets hybrid checkpoints actually get stored on realistic EOS-terminated rounds, without corrupting the checkpoint's claimed token sequence vs. the cache's true physical state. (`PromptCache.planCacheStore` + `GenerateCompletionInfo.stopTokenFedToCache`.)
- [x] `IntegrationTesting/.../PromptCacheHybridReuseTests.swift` passes against the REAL `mlx-community/Qwen3.6-27B-mxfp4` model — run for real, confirmed via `xcodebuild`. (Passes with the corrected, empirically-verified assertion; the original `>= sharedPrefixTokens - 1` assertion was provably unsatisfiable for this model/scenario due to an unrelated chat-template mechanism, not the bug this task fixed.)
- [x] No regression: `xcodebuild ... -only-testing:IntegrationTestingTests/PromptCacheReuseTests` (pure-attention path) still passes; `swift test` (full, unfiltered) stays green at 610 tests.
- [x] `PromptCache.cacheAdvanceOffset`'s doc comment rewritten to reflect the actual fixed behavior, not the old \"KNOWN, ACCEPTABLE DEGRADATION\" language.

## Scope

Primarily `Libraries/MLXFoundationModels/MLXLanguageModel.swift` and `Libraries/MLXFoundationModels/PromptCache.swift`/`PromptCacheChunks.swift`, plus `Libraries/MLXLMCommon/Evaluate.swift` for `stopTokenFedToCache`. Also touched `IntegrationTesting/.../PromptCacheHybridReuseTests.swift` (test-file-only correction, per explicit user direction) once real-model verification proved the original assertion unsatisfiable for reasons unrelated to this task's fix.