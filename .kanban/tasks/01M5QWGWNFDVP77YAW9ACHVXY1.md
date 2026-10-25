---
assignees:
- claude-code
depends_on:
- 01M5SK9FZ2HTB7XJDAPCBRDDM5
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5SJPFM01WFRYWMQXVGMA4MQ
position_column: todo
position_ordinal: '8280'
title: 'MTP: ChatSession support with drafter-cache reuse across turns'
---
## What
`ChatSession` (Libraries/MLXLMCommon/ChatSession.swift) supports only `SpeculativeDecodingConfig(draftModel: ModelContainer)`. Add an MTP source, and keep the MTP drafter state (`MTPDrafterCacheState` from the drafter-state task: drafter cache + pending hidden row + drafter identity) next to the main cache, so that a multi-turn chat reuses the drafter prefix the same way it reuses the main prefix.

- [ ] Add an MTP case to `SpeculativeDecodingConfig.DraftModelSource` with a public initializer `SpeculativeDecodingConfig(mtpDrafter: any MTPDrafterModel, blockSize:memoryPolicy:)`. Check `drafter.isCompatible(with:)` and fall back to plain decode when it is not compatible. (Deferred loading through the factory is a separate task.)
- [ ] Store the `MTPDrafterCacheState` in `RealizedCache` (~365-411) next to the `draft` slot, NOT in `lmState` and NOT in `main`. Use `isAligned(withMainTokenCount:drafterID:)` in `draftCacheIsAligned` (~1387) and the `PromptCacheState` input, so that the append-suffix decision keeps speculation on. When the state is not aligned (or the drafter changed), use the cold start, not passthrough.
- [ ] Checkpoints: today ChatSession takes checkpoints only when `speculativeDecoding == nil` (~1365-1371). Allow them for the MTP source. `prefillToCheckpoint` (~1562-1642) fills the main cache before any iterator runs: run it with the per-chunk hidden-row hook of the chunked-prefill task and feed the drafter, so that the drafter stays aligned across that boundary.
- [ ] In the generation path (~1660-1760), when the source is MTP, build `MTPSpeculativeTokenIterator` with the session main cache, `lmState` (through the `state:` input), and the stored drafter state. After the turn, store the exported drafter state from the completion callback (nil after passthrough → cold start on the next turn).

## Acceptance Criteria
- [ ] A two-turn `ChatSession` with a system prompt at temperature 0 gives the same text with MTP as without speculation, and the checkpoints are taken with MTP on.
- [ ] On turn 2, `cachedPromptTokenCount > 0`, and the drafter is fed only over the new suffix (assert on the drafter `processedTokenCount` before and after turn 2).
- [ ] When the MTP drafter is not compatible with the target, the session falls back to plain decode and does not throw.

## Tests
- [ ] Tests/MLXLMTests/ChatSessionMTPTests.swift on the shared Qwen35 + MTP fixture: two turns with a system-prompt checkpoint, and the incompatible-drafter fallback.
- [ ] IntegrationTesting/IntegrationTestingTests: a real-model two-turn equality test that skips when `mlx-community/Qwen3.8-27B-MTP-4bit` is not local.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session