---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx5cg3fd6f58dj46c515as1q
  text: |
    ## Investigation result: CONFIRMED BUG, fixed.

    ### Trace

    Read `run()`, `applyBiasAndSample()`, `processFastForwardTokens()`, `advanceSingleSampledToken()` in `GuidedGenerationLoop.swift`, and `commitToken`/`CommitResult`/`emitFastForwardLocked` in `XGrammarBridge.swift` end to end.

    - At the top of each loop iteration, `state.logits` holds logits produced by the **previous** iteration's forward pass (or prefill). `tokenId` is sampled from these already-computed logits in `applyBiasAndSample` — it has NOT itself been fed through the model at that point. Standard autoregressive decoding requires `tokenId` to get its own forward pass afterward to populate its KV-cache slot and produce logits for the next position.
    - Confirmed via `GrammarConstraint.commitToken`'s doc comment and code: `CommitResult.tokens` is populated purely from `FindJumpForwardString`'s suffix, walked in `emitFastForwardLocked` — it **never** includes the token passed into `commitToken` itself. `commitToken` calls `xg_matcher_accept_token(matcher, tokenId)` first (advancing grammar state for `tokenId`), and only *then*, if not terminated, queries the FF suffix. So `tokenId` is grammar-accepted but structurally absent from `ffTokens`.
    - In `run()`'s dispatch: non-empty `ffTokens` → `processFastForwardTokens(ffTokens, ...)` (before my fix, no `tokenId` parameter at all); empty `ffTokens` → `advanceSingleSampledToken(tokenId, ...)`, which explicitly does `model(MLXArray([Int32(tokenId)]), cache:, state:)`.
    - `processFastForwardTokens` (before fix) looped `model(...)` calls only over `ffTokens`, never over `tokenId`. So whenever a commit produced a non-empty FF batch, `tokenId`'s own KV-cache entry was silently never created — every subsequent token's attention could not see it, even though it was emitted to the caller and accepted into the grammar's parse state.
    - Independent confirmation: `onTokenCommitted`'s doc comment (on `run()`) explicitly promises it fires for "every token actually fed through the model this call (sampled tokens **and** fast-forward tokens alike) -- i.e. exactly the tokens `cache`'s `offset` advances over," with two enumerated, deliberate exceptions (the terminal token on `commitResult.isTerminated`, and an FF token abandoned by a mid-batch stop). The FF-triggering sampled token was NOT among those documented exceptions, yet was omitted — proving this was an oversight, not intended behavior.

    ### Ground truth via diagnostic test (RED before fix, GREEN after)

    No existing unit test in `Tests/MLXGuidedGenerationTests/` exercised `GuidedGenerationLoop.run()` end-to-end (existing FF-related tests are integration tests requiring real model weights). Built a fully synthetic regression test, `Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift`:
    - Byte-fallback `GrammarTokenizer` vocab (256 entries, same shape as `ConstraintCachingTests.makeByteFallbackTokenizer()`), literal EBNF grammar `root ::= "ABCD"` with `fastForward: true`.
    - A `RecordingProbeModel` (`Module` + `LanguageModel` + `KVCacheDimensionProvider`) that records every token id actually fed to `callAsFunction`, returning deterministic all-zero logits (irrelevant, since the literal grammar's mask forces exactly one bit at every step).
    - After committing 'A' (65), `FindJumpForwardString` forces "BCD"; `emitFastForwardLocked`'s boundary-safety trim drops the last token ('D'), yielding a genuine non-empty FF batch `['B'=66, 'C'=67]` and leaving 'D' for a separate ordinary commit.
    - Verified via `git stash` on the production fix: **RED** — before the fix, `committedTokenIDs == [66, 67, 68]` (65 missing) and `model.fedTokenIDs == [0, 66, 67, 68]` (0 = prompt token; 65 never fed). Failure message: `Expectation failed: committedTokenIDs == [65, 66, 67, 68]`.
    - **GREEN** after restoring the fix: `committedTokenIDs == [65, 66, 67, 68]`, `model.fedTokenIDs == [0, 65, 66, 67, 68]`, emitted text `"ABCD"`, `tokenCount == 4`.
    - (Debugging note for future readers: an early version of this test crashed with "Can't take a suffix of negative length" inside `NaiveStreamingDetokenizer.next()` — a test-harness bug, not a production one. Root cause: the test's host tokenizer hadn't registered the grammar's own EOS id (255) as its `eosToken`, so `run()`'s premature-EOS stop-check never recognized it, and the loop tried to commit/emit the raw invalid-as-a-lone-UTF8-byte 0xFF token as ordinary text. Fixed by registering `eosToken` properly so the existing EOS pre-commit check (unrelated to this bug) short-circuits cleanly.)

    ### Fix

    `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift`:
    - `processFastForwardTokens` now takes the sampled `tokenId` as an explicit first parameter.
    - Added a single-token forward pass for `tokenId` (mirroring `advanceSingleSampledToken`'s shape) immediately before the existing FF-tokens forward-pass loop, updating `state.modelState` and calling `onTokenCommitted?(tokenId)`. Its logits are discarded (irrelevant — the grammar, not these logits, forces every subsequent FF token; only the last FF token's logits are kept, unchanged from before).
    - Updated the call site in `run()` to pass `tokenId` through.
    - Updated doc comments on `processFastForwardTokens` (parameter docs, KV-cache mutation description) to describe the new behavior precisely.

    ### Verification

    - `swift build` (MLXGuidedGeneration target): clean.
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' ...` then unfiltered `xcrun xctest <bundle>.xctest` (no `-XCTest` filter, no `tail` piping):
      - `MLXGuidedGenerationTests`: **63/63 passed** (62 pre-existing + 1 new regression test).
      - `MLXFoundationModelsTests`: **148/148 passed** (unchanged from last known-good state).

    Task left in `doing` for review per instructions — no commit made.
  timestamp: 2026-07-10T06:47:32.589637+00:00
- actor: claude-code
  id: 01kx5ec58fx5k1zhk2yw5v8m00
  text: |-
    ## Review findings addressed

    ### Finding 1: Deduped "feed token through model" pattern

    Extracted a single shared helper in `GuidedGenerationLoop.swift`:

    ```swift
    private static func feedTokenThroughModel(
        _ tokenId: Int,
        state: inout LoopState,
        model: any LanguageModel,
        onTokenCommitted: ((Int) -> Void)?
    ) -> MLXArray {
        let input = LMInput.Text(tokens: MLXArray([Int32(tokenId)]))
        let result = model(
            input[text: .newAxis],
            cache: cacheOrNil(state.cache),
            state: state.modelState
        )
        state.modelState = result.state
        onTokenCommitted?(tokenId)
        return result.logits
    }
    ```

    Placed right after `cacheOrNil` (which it calls). Rather than a `captureLogits: Bool` param (the sketch in the finding), the helper always returns the freshly-computed logits unconditionally — the model always computes them as part of the forward pass regardless, so there's no actual cost difference to gate with a flag. Each call site instead just decides for itself whether to keep or discard the returned value, which is the genuine per-site difference:

    - **Sampled-token feed in `processFastForwardTokens`** (the fix's own new code): `_ = feedTokenThroughModel(tokenId, ...)` — discards the logits, preserving the documented behavior that only the last FF token's logits matter.
    - **Per-FF-token loop in `processFastForwardTokens`**: assigns the per-iteration return value to a local `logits`, and only stores it into `state.logits` on the last iteration (`if i == ffTokens.count - 1`) — unchanged behavior, now routed through the shared helper.
    - **`advanceSingleSampledToken`**: `state.logits = feedTokenThroughModel(tokenId, ...)` — always keeps the logits, as before.

    All 3 call sites now share one implementation; the only per-site variation is what each does with the returned `MLXArray`. No behavior change at any site — verified via the full test suite (below), including re-running the new regression test that specifically targets this code path.

    ### Finding 2: Casing fix in the new test file

    Renamed `fedTokenIDs` → `fedTokenIds` and `committedTokenIDs` → `committedTokenIds` throughout `Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift` (declaration, mutation call sites, doc-comment prose, and the `#expect` assertions/messages) to match the project's `stopTokenIds`/`eosTokenIds` convention.

    ### Verification

    - `swift build`: clean (only pre-existing, unrelated deprecation warning in `MLXFoundationModels/MLXLanguageModel.swift`).
    - `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation`: **TEST BUILD SUCCEEDED**.
    - `xcrun xctest MLXGuidedGenerationTests.xctest` (unfiltered, full output captured to a file rather than piped through `tail`, to avoid truncating early lines): **63/63 passed**, confirmed line-by-line — `Suite FastForwardSampledTokenKVCacheTests passed` with its one test ("The sampled token that triggers a non-empty FF batch gets its own model forward pass...") passing in 0.004s, and the overall `Test run with 63 tests in 14 suites passed`.
    - `xcrun xctest MLXFoundationModelsTests.xctest` (unfiltered): **148/148 passed**.
    - `mcp__sah__diagnostics check working`: 0 errors, 0 warnings.

    Both findings resolved. Task left in `review` per instructions — no commit made, no column move.
  timestamp: 2026-07-10T07:20:20.495728+00:00
- actor: claude-code
  id: 01kx5fawc6vvvwtcqg51jz65th
  text: |-
    Review of commit 3506e85 (2026-07-10 02:25) returned 16 findings, all missing-doc-section findings (`- Parameters:`/`- Returns:`/`- Throws:`) across the file. Confirmed via `git diff HEAD~1..HEAD` that the ONLY function this commit added/changed is `feedTokenThroughModel` (which already has a substantive doc comment, just not the exact formal-block format) — every cited finding is on OTHER, pre-existing, untouched functions (`prefillAndGetLogits`, `applyBiasAndSample`, `hardZoneBias`, `computeMaskAndArray`, `buildStopTokenIds`, `buildMaskArray`, etc.). This is the same whole-file-rescan pattern already seen with `MLXLanguageModel.swift` (`9jtbtkd`) and `Chat.swift` (`2yyn7f7`) — any touch to a file triggers a full doc-debt rescan, not a diff-scoped one.

    Deferred to new tracking task `zfnef85` ("Add full Parameters/Returns/Throws doc blocks across GuidedGenerationLoop.swift's pre-existing functions"). w7nvvg8's own actual deliverable — the fast-forward KV-cache correctness fix — is complete, independently verified (including a reproduced RED/GREEN), and committed across 2d01861/3506e85. Moving to done.
  timestamp: 2026-07-10T07:37:07.206072+00:00
position_column: done
position_ordinal: '9280'
title: 'Investigate: fast-forward path may skip the sampled token''s own forward pass, leaving a KV-cache gap'
---
## What\nSurfaced independently twice while reviewing `tba2jnb`'s refactor of `GuidedGenerationLoop.run()` (by the implementer agent and, separately, an independent verification agent) — both flagged the same pre-existing behavior, confirmed present in the code before that refactor too (the refactor only relocated it, didn't introduce it).\n\nIn `GuidedGenerationLoop.swift`'s `processFastForwardTokens` (previously inline in `run()`): `GrammarConstraint.commitToken`'s `CommitResult.tokens` carries only the **fast-forward** token ids from `FindJumpForwardString` (confirmed via `XGrammarBridge.swift`'s doc comment) — it does NOT include the just-committed originally-sampled token. `processFastForwardTokens` iterates only over these FF tokens, feeding each through `model(...)` with the cache/state to populate the KV cache and get fresh logits.\n\nThe originally-sampled `tokenID` that triggered the FF batch is emitted (detokenized/accumulated) and committed to the grammar constraint, but is apparently **never independently fed through the model** in the FF-batch branch — contrast with the non-FF `else` branch (`advanceSingleSampledToken`), which explicitly does `model(MLXArray([Int32(tokenID)]), cache:, state:)` for exactly this purpose.\n\nStandard autoregressive KV-cache decoding requires every emitted token to pass through the model once to populate its cache entry (key/value at its position). If `tokenID` is skipped whenever a commit produces a non-empty FF batch, the KV cache and position tracking would silently omit one token's worth of cache state — the first FF token's attention/position would then be computed as if it directly followed the *previous* committed token, with the sampled token invisible to the model's cache.\n\n## Confirmed and fixed (commit 2d01861)\nConfirmed a genuine bug via careful investigation, a synthetic diagnostic test, and an independently-reproduced RED/GREEN (verifier ran `git stash`/rebuild/rerun themselves, not just trusting the implementer). Fixed by feeding the sampled token through the model (mirroring `advanceSingleSampledToken`'s existing pattern) before processing the FF batch in `processFastForwardTokens`. New test: `Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift`. Repo-wide search confirmed `GuidedGenerationLoop.run()` is the only production caller of `commitToken`/`CommitResult.tokens` — fix is complete, no other instance of this bug class elsewhere.\n\n## Review Findings (2026-07-10 01:54)\n\n- [x] `Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift:316,333,399` — the \"feed token through model\" pattern (build `LMInput.Text`, call model, update `state.modelState`, call `onTokenCommitted`) is now duplicated across 3 sites: the new sampled-token feed in `processFastForwardTokens`, the existing per-FF-token loop in the same function, and `advanceSingleSampledToken`. This commit's own fix introduced the third near-duplicate. Extract a shared helper (e.g. `feedTokenThroughModel(_:state:model:onTokenCommitted:captureLogits:)`) and use it at all 3 sites.\n- [x] `Tests/MLXGuidedGenerationTests/FastForwardSampledTokenKVCacheTests.swift:76,92,152` — new test file uses `fedTokenIDs`/`committedTokenIDs` (capital IDs) instead of this project's established `fedTokenIds`/`committedTokenIds` convention (matches `stopTokenIds`/`eosTokenIds` elsewhere in the same package). Rename both identifiers throughout the new file.