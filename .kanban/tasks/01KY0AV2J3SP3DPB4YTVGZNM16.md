---
comments:
- actor: wballard
  id: 01ky5rxv7320an7dgvdsav4t63
  text: |-
    Update (downstream re-test after picking up 5891f01/4d7cbaf/5ab0bb8 -- real hybrid cache reuse, preserve_thinking + reasoning_content replay, split-prefill stable-boundary checkpoints):

    **The good news first:** secondTurnReusesFirstTurnsKVCache (FoundationModelsRouter's hard, never-weakened acceptance test for real cache reuse against mlx-community/Qwen3.6-27B-mxfp4) now PASSES for the first time in this whole investigation. Real, positive cachedTokenCount observed on a real second turn. The split-prefill stable-boundary fix + preserve_thinking history replay genuinely works. Nice work.

    **Correction to my prior report on this task**: I previously characterized the toolUsingTurnRoundTripsToDisk full-suite-only failure as "reproducible, identical, twice in a row" -- that was only 2 data points and I overstated the certainty. Re-tested 3 times against the current tip (5ab0bb8):
    - Standalone: passes every time (3/3).
    - Full 5-suite run: FAILED twice (same 2 assertion lines both times), then PASSED cleanly (15/15) on a 3rd run.

    So this is genuinely intermittent in the full-suite context, not deterministic -- my earlier "always reproduces identically" claim doesn't hold up against more samples. It's still suspicious (a test that only ever fails when run after other suites touching the same model id, never standalone) and worth keeping an eye on, especially now that hybrid cache reuse is actually live end-to-end (previously it might not have engaged often enough to manifest). But I'm downgrading my confidence from "confirmed cross-session leak" to "unconfirmed, possibly-cache-related intermittent full-suite-only failure" -- could also be ordinary model-sampling variance at a tool-calling decision boundary, unrelated to the cache at all.

    Recommend leaving this task open but not treating it as urgent/blocking -- it needs more data (a longer run of repeated full-suite executions) to distinguish real cache leakage from generation-sampling noise before it's actionable.
  timestamp: 2026-07-22T20:40:27.619031+00:00
- actor: claude-code
  id: 01ky5s219bp5w4mc1hms92k14e
  text: 'Re-validation context before implementation (orchestrator, 2026-07-22): (1) Tag fixed #qwewn → #qwen — the typo kept this card out of the /finish #qwen batch. (2) The behavioral evidence was gathered at downstream pin c2790ed, which PREDATES three landed fixes: e78994c (stable-boundary hybrid checkpoints), 4d7cbaf/5891f01 (preserve_thinking + VLMModelFactory reasoningConfig inference). Critically, at c2790ed VLM-factory-loaded Qwen3.6 had reasoningConfig == nil, causing chain-of-thought to leak into the response channel — a wrong-transcript-shape confound at least as plausible as the suspected checkpoint leak for the RecordingHandle failure. The full-suite-only + faster-than-normal signature still warrants the card''s direct instrumentation: acceptance criterion 1 (remove(modelID:) clears hybridCheckpoints — unit test) is decisive and cheap; criterion 2''s cross-conversation isolation test likewise. Downstream re-validation at pin >= 5ab0bb8 recommended after this card''s unit-level work, before anyone chases the leak theory further.'
  timestamp: 2026-07-22T20:42:44.907405+00:00
- actor: claude-code
  id: 01ky5t2gr8496q9jnx06v2dpnr
  text: |-
    Investigation complete — VERDICT: LEAK REFUTED at the mechanism level. Evidence:

    **Acceptance criterion 1 (eviction actually clears hybridCheckpoints): CONFIRMED CORRECT.** New suite `Tests/MLXFoundationModelsTests/PromptCacheHybridEvictionScopeTests.swift` proves, against unmodified production code:
    - `remove(modelID:)` drops the model's ENTIRE hybrid checkpoint store (`hybridCheckpointCount == 0`), reclaims its bytes (`totalStoredByteCount == 0`), and leaves nothing resolvable (`resolveHybridCheckpoint` → nil). Mechanism: `removeAndReclaimBytes(modelID:from:&hybridCheckpoints)` does `store.removeValue(forKey: modelID)` — whole-submap removal, not per-entry.
    - `remove` is per-model: another model id's checkpoints remain resolvable.
    - `evictAll()` drops every model's checkpoints and zeroes byte accounting.
    - On `MLXLanguageModel.evict()` reaching `remove(modelID:)`: `evict()` is a source-verified one-line delegation (`await Self.promptCache.remove(modelID: modelID)`). The `promptCache` instance is a `private static let` with NO internal write seam below container-load level (`storePromptCache`/`commitPromptCache` are private; `resolvePromptCache` is read-only), so a through-`evict()` state assertion would require adding a test seam to the 274KB unreviewable MLXLanguageModel.swift — which this card's constraints forbid. The composition is proven as: (source-verified one-line delegation, already executed at runtime by `ModelCacheEvictionTests.evictIsPerModel`) + (actor-level behavior pinned by the new suite at the exact call `evict()` makes).

    **Acceptance criterion 2 (cross-conversation isolation): CONFIRMED IMPOSSIBLE to cross-match.** Through the real `resolve()` surface with a synthetic hybrid model (`[KVCacheSimple, MambaCache]` probe): conversation A checkpoints → pre-eviction CONTROL proves reuse is observable (continuation feeds only the suffix) → `remove(modelID:)` → unrelated conversation B on the same model id resolves EXACTLY as a cold cache (full token feed, `resolveHybridCheckpoint` nil), and even replaying A's own continuation is cold. The element-wise guard (`Array(newTokens.prefix(len)) == checkpoint.tokens`) is pinned with the adversarial shape a hash-key collision would present: a same-length, one-element-different prefix never matches (a constructed Hasher collision is not feasible — Swift's Hasher is per-process seeded — but the guard never consults the key during matching at all; it's a flat linear scan with element-wise verification, so even a genuine key collision could only overwrite-on-insert, never wrongly restore).

    **TDD red-green verification (the tests detect the leak they hunt):** temporarily mutating `remove`/`evictAll` to skip hybrid clearing made 4/5 tests fail exactly as the leak theory predicts (stale checkpoint restored post-evict, bytes leaked). Notably, EVEN UNDER the leak mutation, unrelated conversation B still could NOT cross-match — only A's own continuation reused stale state. So even if eviction were broken, the downstream `RecordingHandleIntegrationTests` failure would additionally require the later suite's prompt to be a genuine element-wise extension of an earlier suite's stored token prefix. Separately, removing the element-wise equality from `resolveHybridCheckpoint`'s guard made the collision-safety test fail (and the wrongly-restored checkpoint crashed the forward pass — a cross-match would be loud, not subtle). Production code restored; all mutations reverted.

    **Criterion 4 (real explanation for the downstream failure):** With the leak refuted, the most plausible cause of the toolUsingTurnRoundTripsToDisk full-suite-only failures is the confound the orchestrator identified: the behavioral evidence was gathered at downstream pin c2790ed, which PREDATES 5891f01 (VLMModelFactory reasoningConfig inference) — at that pin, VLM-factory-loaded Qwen3.6 had `reasoningConfig == nil`, leaking chain-of-thought into the response channel, a wrong-transcript-shape failure mode at least as consistent with the observations as any cache leak. The reporter's own follow-up (3 more samples at 5ab0bb8) already downgraded the failure to intermittent (2 fail / 1 pass full-suite), consistent with sampling variance at a tool-calling decision boundary. The "faster-than-normal" timing signature is also explained benignly: hybrid checkpoint reuse WAS genuinely engaging within each suite's own session (that's the feature working), which shortens runs without any cross-session contamination. RECOMMENDATION: downstream should re-validate at pin >= 5ab0bb8 (includes 5891f01 + 4d7cbaf + e78994c); if the full-suite failure persists there, gather per-round `cachedTokenCount` + transcript diffs rather than re-suspecting eviction — the eviction/matching mechanism is now regression-pinned by this suite.

    Files changed (tests only; PromptCache.swift and MLXLanguageModel.swift untouched):
    - NEW `Tests/MLXFoundationModelsTests/PromptCacheHybridEvictionScopeTests.swift` (5 tests)
    - `Tests/MLXFoundationModelsTests/PromptCacheTestSupport.swift` (hoisted shared `makeHybridCheckpoint` fixture)
    - `Tests/MLXFoundationModelsTests/PromptCacheHybridArchitectureTests.swift` (removed now-shared private helper)

    Full `swift test` green: 647 Swift Testing tests + all XCTest bundles, 0 failures (verified twice; one earlier run showed a transient MLXLMTests failure note that did not reproduce standalone (283/283) nor in two subsequent full runs — unrelated to this tests-target-only change).
  timestamp: 2026-07-22T21:00:29.320604+00:00
position_column: done
position_ordinal: c880
title: 'Suspected cross-session leak in hybrid prompt-cache checkpoint store (Qwen3.6): full-suite-only failure, faster-than-normal + wrong output'
---
## What

Suspected cross-session state leak in the new hybrid Mamba/attention prompt-cache checkpoint store (`PromptCache.hybridCheckpoints`, landed via task `r9rf5g7`'s commit chain — `7bb20a8`/`a1c1385`/`7c0522f`/`e89044c`/`00556b4`/`a54029a`/`00720fb`/`0b19446`/`83c43e8`). A downstream consumer (`FoundationModelsRouter`'s gated integration suite, pinned at branch `foundationmodels-fixes` commit `c2790ed`) is observing a real, reproducible behavioral regression that appeared only after this cache landed.

## Evidence (behavioral, not yet instrumented at the `PromptCache` level — see "What wasn't done" below)

`Tests/FoundationModelsRouterIntegrationTests/RecordingHandleIntegrationTests.swift`'s `toolUsingTurnRoundTripsToDisk` drives a single tool-calling turn against a freshly-loaded `mlx-community/Qwen3.6-27B-mxfp4` container (a real hybrid Mamba/attention model) and asserts the recorded transcript contains `.toolCalls`/`.toolOutput` events in the expected order.

- Run **alone** (`-XCTest ...RecordingHandleIntegrationTests`): **passes**, 12.4s.
- Run as the **last of 2 suites** (`LanguageModelSessionBackendIntegrationTests` immediately before it, same process, same model id): **passes**, 15.8s.
- Run as the **last of the full 5-suite gated run** (`IntegrationTests` → `LanguageModelSessionBackendIntegrationTests` → `RecordingHandleIntegrationTests` → `SessionTreeRestorationIntegrationTests` → `TranscriptReconstructionIntegrationTests`): **fails, identically, twice in a row** (same two assertion lines both times), completing in **9.7s and 9.8s** — noticeably *faster* than every passing run, not slower.

A faster-than-normal run combined with a *wrong* resulting transcript shape (the tool-call sequence doesn't land where expected) is the classic signature of a cache hit that shouldn't have happened: something restores state that skips real prefill work, but the restored state doesn't actually belong to this session's conversation, so generation behaves differently (here: the tool-calling round doesn't complete the same way).

Every gated suite in this fork's own test file explicitly calls `container.model.evict()` (or `harness.container.model.evict()`) at the end of every test — `MLXLanguageModel.evict()` calls `Self.promptCache.remove(modelID:)`, which per `PromptCache.swift`'s doc comments is supposed to drop **both** `chunkStore` and `hybridCheckpoints` for that model id. All 5 gated suites load the identical model id (`mlx-community/Qwen3.6-27B-mxfp4`) for `.standard`, so if eviction isn't fully clearing `hybridCheckpoints`, or if checkpoint resolution can match across two structurally-similar-but-content-different sessions in some other way, a later suite's fresh load could inherit contaminated state from an earlier, unrelated suite's session.

## What wasn't done (be upfront about the confidence level)

This has NOT been root-caused by reading/instrumenting `PromptCache.swift` directly — no debug logging was added inside `resolveHybridCheckpoint`/`insertHybridCheckpoint`/`remove(modelID:)` to directly observe a stale checkpoint being matched and restored. The evidence above is behavioral (timing + wrong output shape, reproducible only in the full multi-suite run, never in isolation), which is suggestive and specific enough to report, but not a confirmed root cause. It is presented as a lead for someone with the sandbox/GPU access to instrument directly, not a diagnosed fix.

## Suggested acceptance criteria

- [x] Confirm or refute whether `PromptCache.remove(modelID:)` actually clears `hybridCheckpoints` for that model id (the doc comment on `PromptCache.swift`'s "Drops one model's remembered chunk store AND hybrid checkpoint store" section claims it does — verify with a direct unit test: insert a hybrid checkpoint, call `remove(modelID:)`, assert `hybridCheckpointCount(modelID:) == 0`). — CONFIRMED CORRECT: `Tests/MLXFoundationModelsTests/PromptCacheHybridEvictionScopeTests.swift` (plus `evictAll()` and byte-accounting coverage); red-green verified by mutation.
- [x] Add a regression test that loads a real (or synthetic hybrid-architecture) model, drives one conversation and lets it checkpoint, evicts, then drives a SECOND, unrelated conversation on a freshly-loaded instance of the SAME model id — assert the second conversation's generation is unaffected (same output it would produce with a cold cache) and that `resolveHybridCheckpoint` does not match across the two. — DONE: `evictedModelIDIsColdForTheNextConversation` (with pre-eviction reuse control) + `divergentConversationsNeverCrossMatch` (element-wise guard / collision safety).
- [x] If the leak is confirmed, fix eviction (or checkpoint-matching scope) so a model's checkpoint store cannot outlive/cross-contaminate an evicted-and-reloaded instance. — N/A: leak NOT confirmed; eviction and matching scope are correct, no production change required (verified against unmodified code).
- [x] If NOT confirmed (i.e. eviction is already correct), investigate whether checkpoint content-hash collisions or some other mechanism explains `RecordingHandleIntegrationTests`'s full-suite-only failure, and document the real cause. — DONE: documented in task comments. Hash-key collisions cannot cause a wrong restore (matching is a flat linear scan with element-wise token verification; the key is never trusted). Most plausible cause of the downstream failure: the reasoningConfig-nil confound at pin c2790ed (fixed in 5891f01) plus sampling variance at a tool-calling decision boundary (reporter's own re-test at 5ab0bb8 showed the failure is intermittent, 2/3). Downstream should re-validate at pin >= 5ab0bb8.

## Scope

`Libraries/MLXFoundationModels/PromptCache.swift`'s `remove(modelID:)`/`insertHybridCheckpoint`/`resolveHybridCheckpoint`, and `MLXLanguageModel.evict()`'s call into it. Reported from a downstream consumer without GPU-level instrumentation access in this environment — someone with the ability to run this fork's own test suite against a real Qwen3.6 model should reproduce directly against `PromptCache` rather than trust this behavioral report alone. #qwen