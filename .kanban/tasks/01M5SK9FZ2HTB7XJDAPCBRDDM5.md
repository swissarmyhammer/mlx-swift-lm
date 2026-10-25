---
assignees:
- claude-code
depends_on:
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5SJNWMNA5D8N2VWEJJPT9A2
position_column: todo
position_ordinal: '9380'
title: 'MTP: public generate overloads take and give MTPDrafterContinuation (rest done by PR 581)'
---
## What
Upstream PR 581 (ported in commit 843854ec) already does most of this task for ChatSession: `MTPSpeculativeTokenIterator` takes `mtpContinuation:`, `finalizeDrafterContinuation` runs in the `defer` of `finalizeGeneration`, the tail `passthroughStep(maintainingDrafterState: true)` keeps the drafter in step, the continuation is nil after a sticky passthrough, and `RecordedGenerationResult` returns the continuation (Libraries/MLXLMCommon/Evaluate.swift). Tests: Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift and ChatSessionMTPEarlyTerminationTests.swift.

What remains:
- [ ] The public `generate` / `generateTokens(...mtpDrafter:)` overloads (Evaluate.swift ~2690-2790) take an optional `MTPDrafterContinuation` and give the final one back (completion callback or return value), so that callers other than ChatSession (the FoundationModels executor) can continue a drafter.
- [ ] Check the zero-round path (`finalizeGeneration` with no round) and the cancellation path export the correct continuation or nil; add the tests that PR 581 does not have.

## Acceptance Criteria
- [ ] Two `generate(...mtpDrafter:)` calls, where the second continues the main cache and the continuation of the first, give the same greedy tokens as one cold call and as plain decode.
- [ ] Release benchmark on the M5 Max (`qwen38FourBitMTPGeneration`): MTP tok/s within ±3% of the value measured after the PR 581/459 port.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPTests.swift: continuation through the public overload; zero rounds; cancellation.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38