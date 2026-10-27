---
assignees:
- claude-code
position_column: todo
position_ordinal: '9680'
title: 'Speculative decoding: report the true cache state of a stop token from SpeculativeTokenIterator (hasCachedLastToken)'
---
## What
^jjpt9a2 added `TokenIteratorProtocol.hasCachedLastToken` (default `true`) and `generateLoopTask` now sets `GenerateCompletionInfo.stopTokenFedToCache` only when it is `true`. `MTPSpeculativeTokenIterator` implements it. The draft-model `SpeculativeTokenIterator` (Libraries/MLXLMCommon/Evaluate.swift) has the same shape and still takes the default: the final token of each round (`pendingTokens` index `>= mainCommittedPendingTokenCount`) and every `plainRound` token are returned before any forward pass feeds them. A stop on such a token is reported as fed, so a ledger that adds `stopTokenFedToCache` (the FoundationModels executor, `MLXLanguageModel.generatedTokenCount(of:)`) is one ahead of the main cache.

- [ ] Implement `hasCachedLastToken` on `SpeculativeTokenIterator`: `true` only when the returned pending token's index is below `mainCommittedPendingTokenCount`.

## Acceptance Criteria
- [ ] A draft-model stream that stops on the round's final token reports `stopTokenFedToCache == nil`, one that stops on an accepted draft reports the token, and in both the main cache offset after `finalizeGeneration` equals prompt + `generationTokenCount` + (fed ? 1 : 0).

## Tests
- [ ] Tests/MLXLMTests/SpeculativeDecodingTests.swift: the two stop cases above.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd`. #speculative-decoding