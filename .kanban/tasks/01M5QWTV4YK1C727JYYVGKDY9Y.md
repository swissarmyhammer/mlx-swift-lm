---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzv3pbye859tdw9xeanw4
  text: 'Change after the PR 581 port (commit 843854ec): the MTP source is `SpeculativeDecodingConfig.Strategy.mtp(MTPDrafterContainer, blockSize)`. Add the deferred form to this enum.'
  timestamp: 2026-10-25T13:46:20.150521+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: 8a80
title: 'MTP: deferred drafter loading in ChatSession through MTPDrafterModelFactory and the memory policy'
---
## What
After the ChatSession MTP task, a caller must load the MTP drafter itself (`MTPDrafterModelFactory.shared.load`, after `Qwen35TextMTPRegistration.register()`) and pass the model. The separate-draft-model path already supports deferred loading (`DraftModelSource.deferred(bytes:loader)`) and `SpeculativeDecodingMemoryPolicy` (ChatSession.swift ~43-140, ~1336-1345, ~1673-1711). Give MTP the same support.

- [ ] Add a deferred MTP source, for example `SpeculativeDecodingConfig(mtpDrafterID: String, blockSize:memoryPolicy:)` or `(mtpConfiguration: ModelConfiguration, ...)`. It loads through `MTPDrafterModelFactory` (Libraries/MLXLMCommon/MTPDrafterModelFactory.swift) with the registrations of MLXLLM and MLXVLM Qwen35 MTP, only when speculation is admitted.
- [ ] Report the drafter bytes (`estimatedDraftModelBytes`) to `SpeculativeDecodingMemoryPolicy`, so that `.recommendedWorkingSet` falls back to plain decode, and `.fail` throws `SpeculativeDecodingMemoryError`, as for draft models.
- [ ] Keep the loaded drafter in `loadedDraftModel` (or a matching MTP slot), so that it loads one time for each session and its forks.

## Acceptance Criteria
- [ ] A session configured with the MTP drafter ID loads the drafter on the first speculative turn and not before, and reuses it on later turns.
- [ ] With a memory policy that does not admit the drafter, the session generates with plain decode (fallback action), or throws (fail action).

## Tests
- [ ] Tests/MLXLMTests: ChatSession tests with a scripted loader closure that counts loads (one load across two turns and a fork), and memory-policy fallback and fail tests.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session