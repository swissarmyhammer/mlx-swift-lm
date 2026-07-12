---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kx9832r98pfzx5tpafeth2wb
  text: |-
    Fix implemented and fully verified.

    ## Fix
    `Libraries/MLXFoundationModels/PromptCache.swift`, `reconcileGeneratedTokens(reencoded:actualGeneratedCount:)`: added a third switch case (one-line, exactly as prescribed) plus an expanded doc comment explaining both the existing `+1` (next()-ahead-prefetch-discards-EOS) case and the new `-1` (EOS-decodes-to-empty-text) case:

    ```swift
    switch reencoded.count {
    case actualGeneratedCount:
        return reencoded
    case actualGeneratedCount + 1:
        return Array(reencoded.dropLast())
    case actualGeneratedCount - 1:
        return reencoded
    default:
        return nil
    }
    ```

    `Libraries/MLXFoundationModels/MLXLanguageModel.swift`'s `commitPromptCache(modelID:slot:emittedText:tokenizer:)` doc comment extended to document that it now also accepts one fewer re-encoded token than the cache's offset advance, and that this composes with the `generatedTokenIDs` overload's own `reconcileCacheAdvance`/`trimCacheByOne`/`trimAndVerify` handling (no new logic needed in that overload).

    ## Tests added (TDD, RED confirmed before the fix, GREEN after)
    In `Tests/MLXFoundationModelsTests/PromptCacheTests.swift`:
    - `oneFewerReencodedTokenIsTrustedAsIs` -- hand-picked-integers version of the bug reproduction; confirmed `nil` against old code, `[10,11,12]` against fixed code.
    - `realisticTextTerminalEOSOffByOneTrustsReencodingAsIs` -- realistic-text (`ByteTokenizer`) mirror of the existing `realisticTextNaturalStopOffByOneDropsTrailingToken`, using `actualGeneratedCount = reencoded.count + 1`.
    - `terminalEOSCaseComposesWithCacheAdvanceTrimAndVerify` -- end-to-end-style unit test chaining `reconcileGeneratedTokens` -> `reconcileCacheAdvance` (asserts `.trimCacheByOne`) -> `PromptCache.trimAndVerify` with a real `KVCacheSimple`, proving the RESULT is a correctly trimmed cache (`cache.offset` lands exactly at `promptTokenCount + reencoded.count`), not just that reconciliation returns non-nil. Reused the existing `KVCacheSimple`/`trimAndVerify` test infrastructure from `PromptCacheTrimAndVerifyTests` rather than inventing new fixtures.

    Also had to update one PRE-EXISTING test, `fewerReencodedTokensIsUntrustworthy` (renamed `twoOrMoreFewerReencodedTokensIsUntrustworthy`): its original data (`reencoded: [10,11], actualGeneratedCount: 3`) was itself the exact N-1 bug case and asserted the old (buggy) `nil` result as correct. Bumped its `actualGeneratedCount` from 3 to 4 (a genuine two-token gap, still correctly untrustworthy) so it continues to test real over-mismatch rejection rather than contradicting the fix. This was a required consequence of the fix, not a discretionary refactor.

    ## Verification
    - RED: ran `swift test --filter 'PromptCache'` against the unfixed code -- the 3 new tests failed exactly as expected (one via `#expect` failure, one via a force-unwrap fatal error consistent with `reconcileGeneratedTokens` returning `nil`).
    - GREEN: `swift test --filter 'PromptCache'` -- 95/95 tests pass (includes `PromptCacheReconciliationTests`, `PromptCacheAdvanceReconciliationTests`, `PromptCacheTrimAndVerifyTests`, and the rest of the PromptCache suite).
    - Full 4-bundle mandated verification: `xcodebuild build-for-testing -scheme mlx-swift-lm-Package -destination 'platform=macOS' -clonedSourcePackagesDirPath .build -disableAutomaticPackageResolution -skipPackagePluginValidation` succeeded, then unfiltered `xcrun xctest <bundle>` per bundle, all green:
      - CXGrammarTests: 7/7
      - MLXFoundationModelsTests: 202/202
      - MLXGuidedGenerationTests: 63/63
      - MLXLMTests: 245/245
    - Real-model verification: built IntegrationTesting for testing (`xcodebuild build-for-testing ... -project IntegrationTesting.xcodeproj -scheme IntegrationTesting`), then ran `xcodebuild test-without-building ... -only-testing:IntegrationTestingTests/PromptCacheReuseTests` against the real Qwen2.5-3B-Instruct-4bit model. Result: **PASS** -- `Test "Second respond() round prefills only the appended suffix, not the whole transcript" passed after 4.262 seconds`, confirming `cachedTokenCount > 0` on round 2 and that the fix resolves the real-model bug this task was opened for.

    All acceptance criteria met. Did not commit per constraints; task left in `doing` for review.
  timestamp: 2026-07-11T18:47:29.289309+00:00
- actor: claude-code
  id: 01kx98n7ntkvavvg2snwm02a22
  text: |-
    Adversarial double-check (via the double-check agent) ran against the diff before final sign-off. Verdict: REVISE, with two findings:

    1. (High) The new `actualGeneratedCount - 1` case trusts a one-token-shorter re-encoding by COUNT alone, not by verifying the shortfall is actually the trailing EOS position -- a false positive (some other cause of a one-shorter re-encoding) would store a `trustedTokens` array whose content doesn't match the real token IDs the trimmed cache tensors were produced from.

    2. (Medium) `actualGeneratedCount == 1` (a round that generated a single, EOS-only token) was untested.

    Response:
    - Finding 2: added `pureEOSSingleTokenRoundReencodesToEmptyAndIsTrustedAsIs` to `PromptCacheTests.swift`, covering `reencoded: [], actualGeneratedCount: 1`. Verified it traces through `commitPromptCache(...generatedTokenIDs:)`'s `guard !generatedTokenIDs.isEmpty else { return }` to a safe no-op (leaves any prior valid cache entry untouched, rather than the OLD code's destructive `removePromptCache` wipe) -- an improvement, not a regression.
    - Finding 1: investigated rather than dismissed. Read `lookupLongestPrefix` and `lookupTailMatch` in `PromptCache.swift`: both re-verify STORED token content against a fresh, independent re-tokenization of the full transcript before ever reusing a chunk (`chunk.tokens == window` exact compare for full chunks; longest-common-prefix for the tail) -- explicitly documented as "COLLISION SAFETY" protecting against exactly this class of wrong-content-under-a-matching-key scenario. A false-positive `-1` case would store a wrong token-ID list, but the NEXT round's resolve() would simply fail to prefix-match that chunk's content against the freshly re-tokenized transcript and treat it as a miss (degrade to rebuilding from that point), never silently serving mismatched KV state. This is the identical safety net the pre-existing (already-shipped) `actualGeneratedCount + 1` case already relies on per its own doc comment ("gets caught by the next round's real re-tokenization comparison... falls back to trim/rebuild"), so this fix introduces no new class of risk beyond what the existing reconciliation design already accepts. Added this reasoning explicitly to `reconcileGeneratedTokens`'s doc comment so it isn't tribal knowledge. Per the task's own explicit instruction ("implement this exact fix, don't re-investigate from scratch"), building exact-terminal-token verification via MLXLMCommon's stream `.info` event was out of scope for this task; flagged as a possible future hardening if this reasoning is ever found insufficient in practice, but not required here.

    Re-verified after these additions: `swift test --filter 'PromptCache'` 96/96 green (was 95, +1 new test); rebuilt for testing and re-ran `xcrun xctest MLXFoundationModelsTests.xctest` unfiltered: 203/203 green (was 202, +1). The other three bundles (CXGrammarTests 7/7, MLXGuidedGenerationTests 63/63, MLXLMTests 245/245) were unaffected by this round's changes (doc-comment + one new pure-unit-test only, no production logic changed) and were already verified fresh earlier in this session. The real-model `PromptCacheReuseTests` result (PASS) also stands unchanged, since no production logic changed in this round, only comments/tests.

    Proceeding to sign-off per really-done's advisory-gate contract (fix + logged justification, one round, not looping further).
  timestamp: 2026-07-11T18:57:24.154777+00:00
- actor: claude-code
  id: 01kx98nay79zae3nnyerbzkyxm
  text: |-
    Committed at `4518120293603ca47ad9fbf5bb2cdd00e1a239d8` (3 files, 122 insertions / 12 deletions — `PromptCache.swift`'s new switch case, `MLXLanguageModel.swift`'s doc-comment-only update, `PromptCacheTests.swift`'s 3 new tests + 1 adjusted pre-existing test).

    Review (`review sha HEAD~1..HEAD`) returned 1 finding: `MLXLanguageModel.swift:2230` — `runReasoning`'s 4-level nesting. REJECTED as stale/pre-existing: verified via `git diff HEAD~1..HEAD -- Libraries/MLXFoundationModels/MLXLanguageModel.swift`, which shows this commit touched ONLY a 14-line doc-comment addition at line ~2266-2286 (the `commitPromptCache(...emittedText:tokenizer:)` doc comment) — nowhere near `runReasoning` (a distinct, untouched function). The finding cites code this commit never modified.

    Task's own acceptance criteria are all met (see prior comment): fix verified via TDD RED/GREEN, full 4-bundle xcodebuild+xctest green, and real-model `PromptCacheReuseTests.secondRoundPrefillsOnlyAppendedSuffix()` PASSED against Qwen2.5-3B-Instruct-4bit. Moving to done.
  timestamp: 2026-07-11T18:57:27.495967+00:00
- actor: claude-code
  id: 01kx98zksfts943cpw9mkc5nq6
  text: 'Follow-up committed at `2eca4312e244df1c26aa936630fefc0d8cb752c0`: the adversarial double-check round''s remaining output (a collision-safety rationale doc paragraph + the `pureEOSSingleTokenRoundReencodesToEmptyAndIsTrustedAsIs` edge-case test for `actualGeneratedCount == 1`) had landed in the working tree after the first commit but before I checked — reconciled and committed separately. Re-review clean (zero findings). Both commits (`4518120`, `2eca431`) pushed to origin.'
  timestamp: 2026-07-11T19:03:04.239441+00:00
position_column: done
position_ordinal: a880
title: commitPromptCache's emittedText reconciliation gate rejects legit terminal-EOS-only generations (reencoded.count == actualGeneratedCount - 1)
---
## What\nReal-model evidence (Qwen2.5-3B-Instruct-4bit, via `PromptCacheReuseTests` in `IntegrationTesting`) shows `MLXLanguageModel.swift`'s `commitPromptCache(modelID:slot:emittedText:tokenizer:)` (~line 2273) calling `PromptCache.reconcileGeneratedTokens(reencoded:actualGeneratedCount:)` and getting `nil` back on BOTH rounds of a real two-turn conversation, wiping the model's entire prompt cache (`removePromptCache`) each time -- zero reuse ever survives a round, independent of the tail-chunk floor fix (kanban `5ra1wzm`).\n\nConfirmed via a temporary debug probe at the `nil` branch (reverted, not committed) logging `reencoded.count` vs `actualGeneratedCount`:\n```\nreconcileGeneratedTokens nil: reencoded.count=1 actualGeneratedCount=2\nreconcileGeneratedTokens nil: reencoded.count=2 actualGeneratedCount=3\n```\nBoth times `reencoded.count == actualGeneratedCount - 1` -- the OPPOSITE direction from the one mismatch `reconcileGeneratedTokens` currently tolerates (`actualGeneratedCount + 1`, documented as `TokenIterator`'s next()-ahead prefetch discarding a terminal EOS/stop token that already advanced the cache but was never handed to the stream).\n\nMost likely root cause: `runUnconstrained`'s `emittedText` (`Libraries/MLXFoundationModels/MLXLanguageModel.swift` `handleGenerationEvent`, `case .chunk(let text): emittedText += text`) only accumulates DECODED TEXT chunks from MLX-LM's `Generation` stream. When the model's actual final generated token is itself the EOS/stop token, that token advances `cache.offset` (contributing 1 to `actualGeneratedCount`) but the streaming detokenizer emits no corresponding text chunk for it (special tokens decode to empty content), so re-encoding `emittedText` recovers one FEWER token than the cache's real advance. `reconcileGeneratedTokens` has no tolerance for this direction and returns `nil`, and the caller responds by dropping the ENTIRE cache entry rather than just failing to store this round's marginal tokens.\n\nThis reproduced on EVERY round of a real conversation in this run (not an edge case) -- with `GenerationOptions(maximumResponseTokens: 8)` on short prompts, natural EOS-terminated generation is the common case, not the exception, so this gate is currently rejecting the common case rather than the rare one.\n\n## Why this matters\nThis is the reason `PromptCacheReuseTests.secondRoundPrefillsOnlyAppendedSuffix()` fails against the real model even after the tail-chunk floor fix landed (kanban `5ra1wzm`): `second.cachedTokenCount == 0` on round 2 because round 1's entry was already wiped by this gate, not because of a floor/tail-matching regression. `PromptCachePrewarmTests` passes 3/3 in the same run because prewarm stores tokens directly and never goes through this `emittedText:tokenizer:` reconciliation path at all.\n\n## Proposed direction (needs its own investigation/TDD, not prescribed as final)\nExtend `PromptCache.reconcileGeneratedTokens` (or add a sibling reconciliation entry point) to also handle `reencoded.count == actualGeneratedCount - 1`, but note appending an assumed placeholder is unsafe without knowing the REAL terminal token ID -- `trustedTokens` must end up with an array whose length matches the cache's real offset advance for chunk storage/verification (`sliceChunks`'s `cache.first?.offset` check) to succeed. Likely needs the real EOS/stop token ID from the tokenizer or from MLXLMCommon's generation stream/`GenerateCompletionInfo`, not just padding with a guess. Investigate whether MLXLMCommon's `Generation`/`.info` event already exposes the actual stopping token ID or reason, which would let this be reconstructed exactly rather than assumed.\n\n## Acceptance Criteria\n- [x] Root cause confirmed with a proper (non-debug-probe) test: a unit test reproducing \"final generated token is EOS, decodes to empty text, cache offset advances by N, re-encoded text yields N-1 tokens\" against `reconcileGeneratedTokens` or the `commitPromptCache(...emittedText:tokenizer:)` call site\n- [x] Gate correctly trusts (and correctly stores, offset-aligned) this legitimate case instead of dropping the whole cache entry\n- [x] `PromptCacheReuseTests.secondRoundPrefillsOnlyAppendedSuffix()` passes against the real model (`cachedTokenCount > 0` on round 2)\n- [x] `swift test --filter 'PromptCache'` green; full xcodebuild+xctest run green\n\n## Context\nDiscovered while working kanban `5ra1wzm` (tail-chunk PromptCache floor fix) -- that task's own tail/floor fix is complete, correct, and verified independently (unit tests + `PromptCachePrewarmTests` passing 3/3 against the real model); this is a distinct, pre-existing gap in the re-encode-reconciliation gate, orthogonal to the floor fix, and should NOT block that task.