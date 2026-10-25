---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzr0ttt5993g35vame3b6
  text: 'Change after the PR 581/459 port (commits 843854ec, 82f273fd): the fake models and drafters of Tests/MLXLMTests/ChatSessionQwenTextMTPTests.swift, ChatSessionMTPEarlyTerminationTests.swift and MTPSpeculativeTokenIteratorTests.swift (MockDrafter, StatefulMockDrafter) are private to their files. Move them into the shared fixture target.'
  timestamp: 2026-10-25T13:46:16.986966+00:00
position_column: todo
position_ordinal: '9280'
title: 'MTP: shared Qwen35 + MTP test fixture for MLXLMTests and MLXFoundationModelsTests'
---
## What
Many MTP tasks need a small Qwen35 target with an MTP head that runs on random weights in unit tests. Today `makeFixture` is private in Tests/MLXLMTests/Qwen35MTPTests.swift (~929). The shared scripted ChatSession builder (Tests/MLXLMTests/ScriptedChatSession.swift, commit 478abaa1) takes a `ScriptedTokenModel`, which fails `Qwen35MTPDraftModel.isCompatible(with:)` (Libraries/MLXLLM/Models/Qwen35MTP.swift ~117-119). `MLXFoundationModelsTests` cannot see helpers in `MLXLMTests`.

- [ ] Add a test-support target (for example `MLXLMTestSupport` in Package.swift, linked by `MLXLMTests` and `MLXFoundationModelsTests`) with `Qwen35MTPFixture`: a tiny Qwen35 text config (GDN + attention layers, small vocabulary), a matching `Qwen35MTPDraftModel`, a seeded weight init, a tokenizer stub, and a `ModelContainer` builder.
- [ ] Give the fixture a mode where the drafter copies the target (high acceptance) and a mode with independent random weights (low acceptance), so that tests can force accept and reject rounds.
- [ ] Move the private `makeFixture` users in Qwen35MTPTests.swift to the shared fixture.
- [ ] Add a FoundationModels builder that makes an `MLXLanguageModel` on the fixture container, so that executor tests can run `LanguageModelSession` on it.

## Acceptance Criteria
- [ ] The existing Qwen35MTPTests pass on the shared fixture with no change in results.
- [ ] A smoke test in each of `MLXLMTests` and `MLXFoundationModelsTests` builds the fixture and runs one greedy MTP stream that equals plain decode.

## Tests
- [ ] Tests/MLXLMTests/Qwen35MTPFixtureSmokeTests.swift and Tests/MLXFoundationModelsTests/Qwen35MTPFixtureSmokeTests.swift.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest && xcrun xctest .build/out/Products/Debug/MLXFoundationModelsTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #testing