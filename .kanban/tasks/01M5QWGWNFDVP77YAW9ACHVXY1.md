---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5zxy5h66kbwhkp785nwcwr4
  text: 'Note from the ^vgma4mq review (2026-10-27), for checkpoint byte accounting: `MTPDrafterContinuation.byteCount` counts like the main caches, and inherits two inaccuracies of `KVCacheSimple.copy()` / `residentByteCount`. (1) Undercount: a copied drafter cache counts only its filled rows (offset); once the live cache grows into a new buffer, the old one is kept alive by the copy at full size (up to 255 extra rows per layer; the Qwen35 drafter has 1 layer). (2) Double count: while a copy and its source both live, each counts the shared boundary hidden row and seed. Decide whether the checkpoint policy needs exact numbers (e.g. compact the copy with a contiguous slice when it is stored) and test it.'
  timestamp: 2026-10-28T00:18:09.062099+00:00
depends_on:
- 01M5SJNWMNA5D8N2VWEJJPT9A2
- 01M5QWG18ZZAMS370KT33CTVTY
- 01M5QWGEWP4506CZYXCPV3C64M
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8280'
title: 'MTP: ChatSession MTP gaps after PR 581 (checkpoints, incompatible drafter, cold start, memory policy)'
---
## What
Upstream PR 581 (commit 843854ec) added `SpeculativeDecodingConfig(mtpDrafter: MTPDrafterContainer, blockSize:) throws` with a `Strategy` enum, `MTPDrafterContinuation` in `RealizedCache`, the append-suffix reuse, fork copy, and typed `SpeculativeDecodingFallbackReason`. Tests: Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift.

What remains in Libraries/MLXLMCommon/ChatSession.swift:
- [ ] Checkpoints: the fork takes checkpoints only when `speculativeDecoding == nil` (~1365-1371). Allow them for the MTP strategy. `prefillToCheckpoint` (~1562-1642) must feed the drafter (per-chunk hidden-row hook of the chunked-prefill task) so that the continuation stays aligned across that boundary.
- [ ] An incompatible drafter: PR 581 throws from `SpeculativeDecodingConfig(mtpDrafter:)`. Make the session fall back to plain decode with a fallback reason (keep a throwing validation API for callers that want it).
- [ ] `unavailableContinuation`: use the cold start from the drafter task instead of target-only.
- [ ] `memoryPolicy` for the MTP strategy (PR 581 sets it to nil): report the drafter bytes to `SpeculativeDecodingMemoryPolicy`.

## Acceptance Criteria
- [ ] A two-turn `ChatSession` with a system prompt at temperature 0 gives the same text with MTP as without speculation, the checkpoints are taken with MTP on, and turn 2 feeds the drafter only over the new suffix.
- [ ] An incompatible drafter gives plain decode with a fallback reason, and does not throw.
- [ ] A memory policy that does not admit the drafter gives plain decode (fallback action) or throws (fail action).

## Tests
- [ ] Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift: checkpoints with MTP, incompatible drafter, cold start, memory policy.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session