---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m632q3kw56apv7z7p2knqp5r
  text: |-
    Research + implementation notes:
    - The completion-info path from ^jjpt9a2/^h91ttb2 (`generateLoopTask` sets `stopTokenFedToCache` only when `iterator.hasCachedLastToken`) is reused unchanged; the only production change is the iterator property.
    - `SpeculativeTokenIterator.hasCachedLastToken` is a computed property: `pendingIndex > 0 && pendingIndex - 1 < mainCommittedPendingTokenCount`. A computed property needs no extra state to keep in sync. It reads `false` for the round-final token, for the first token sampled from the main prefill logits (`startDecoding(mainLogits:)` leaves `mainCommittedPendingTokenCount` at 0), and for `plainRound` tokens (count reset to 0). The MTP iterator stores its flag in `nextPendingToken()`; the draft iterator has two inline drain sites in `next()`, so computing it from the index covers both without touching them.
    - RED: with the protocol default, `.roundFinal` failed both tests (reported 35 as fed; main offset 4 vs ledger 5). `.acceptedDraft` already passed. GREEN once the property was added.
    - Fixture: `CacheTrackingTransitionModel` in SpeculativeDecodingTests.swift went from `private` to internal so the new file can reuse it instead of a copy. Prompt [7], 3 drafts: the first round returns [24, 51, 88, 35] with 3 committed.
    - Tests are in a new file (user rule: new tests go in new files) rather than in SpeculativeDecodingTests.swift as the card says.
    - Validator dump was 754 KB, mostly swiftlint/periphery tool-runner bodies for other languages. I read the general hygiene, reuse, duplication, Swift guideline and test-integrity rules in full and skimmed the per-language tool bodies.
  timestamp: 2026-10-29T05:39:23.900654+00:00
- actor: claude-code
  id: 01m632q7yfydz7xscbr236x14s
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/Evaluate.swift (`SpeculativeTokenIterator.hasCachedLastToken`), Tests/MLXLMTests/SpeculativeStopTokenCacheTests.swift (new, 5 KB, 2 tests x 2 slots), Tests/MLXLMTests/SpeculativeDecodingTests.swift (`CacheTrackingTransitionModel` private -> internal). `swift build --build-tests`: 0 compiler warnings. The only build warning is SwiftPM's "missing creator for mutated node" on the Cmlx bundle, which is not a compiler warning. xctest: CXGrammarTests 7 passed, MLXFoundationModelsTests 404 passed, MLXGuidedGenerationTests 70 passed, MLXHuggingFaceMacrosTests 6 passed, MLXLMTests 1990 swift-testing + 920 XCTest passed, 0 failures. `benchmarkVerification` was already skipped before this change. Release integration: qwen38FourBitMTPGeneration MTP 92.5/83.1/80.8 tok/s (plain 59.6); qwen38FourBitMTPSampledGeneration passed; qwen38FourBitMTPSystemPromptCache passed with no known issues, edit turn prefilled 123 tok with MTP on.
    - next: /review
  timestamp: 2026-10-29T05:39:28.335538+00:00
- actor: claude-code
  id: 01m632t5wv22ssdjzvczyqh11q
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (610a7974) — findings 0, confirmed 0, refuted 1, attempted 7, failed 0; 3 files reviewed, 2 .kanban files skipped (no validator matched)
    - next: none — task moved to done
  timestamp: 2026-10-29T05:41:04.539528+00:00
- actor: claude-code
  id: 01m632tkre621jscnmyn5vp3g3
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 3 files (hasCachedLastToken from mainCommittedPendingTokenCount, new stop-token cache tests)
    - test: green — 5 bundles; Release MTP 92.5/83.1/80.8 vs plain 59.6; edit turn 123 tokens
    - commit: 610a7974
    - review: clean — task moved to done
  timestamp: 2026-10-29T05:41:18.734432+00:00
position_column: done
position_ordinal: fffb80
title: 'Speculative decoding: report the true cache state of a stop token from SpeculativeTokenIterator (hasCachedLastToken)'
---
## What
^jjpt9a2 added `TokenIteratorProtocol.hasCachedLastToken` (default `true`) and `generateLoopTask` now sets `GenerateCompletionInfo.stopTokenFedToCache` only when it is `true`. `MTPSpeculativeTokenIterator` implements it. The draft-model `SpeculativeTokenIterator` (Libraries/MLXLMCommon/Evaluate.swift) has the same shape and still takes the default: the final token of each round (`pendingTokens` index `>= mainCommittedPendingTokenCount`) and every `plainRound` token are returned before any forward pass feeds them. A stop on such a token is reported as fed, so a ledger that adds `stopTokenFedToCache` (the FoundationModels executor, `MLXLanguageModel.generatedTokenCount(of:)`) is one ahead of the main cache.

- [x] Implement `hasCachedLastToken` on `SpeculativeTokenIterator`: `true` only when the returned pending token's index is below `mainCommittedPendingTokenCount`.

## Acceptance Criteria
- [x] A draft-model stream that stops on the round's final token reports `stopTokenFedToCache == nil`, one that stops on an accepted draft reports the token, and in both the main cache offset after `finalizeGeneration` equals prompt + `generationTokenCount` + (fed ? 1 : 0).

## Tests
- [x] Tests/MLXLMTests/SpeculativeDecodingTests.swift: the two stop cases above. (Placed in the new file Tests/MLXLMTests/SpeculativeStopTokenCacheTests.swift per the user's new-tests-in-new-files rule; it reuses `CacheTrackingTransitionModel` from SpeculativeDecodingTests.swift.)
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd`. #speculative-decoding