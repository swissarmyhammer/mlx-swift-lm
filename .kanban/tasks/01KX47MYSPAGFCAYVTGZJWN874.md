---
assignees:
- claude-code
position_column: done
position_ordinal: '9080'
title: Review of f94aefa..HEAD (other agent's image/tool-output + doc work)
---
Scope: f94aefa..HEAD (13 commits: tool-output image preservation fb7e9a3, image-failure error mapping f402e64, Chat.Message.tool images param 321f136, access-level/doc cleanups, plus PromptCache test/slot-limit commits)

## Review Findings (2026-07-09 14:42)

- [x] `Libraries/MLXLMCommon/Chat.swift:3` — Public enum `Chat` lacks documentation. Fixed via ^2yyn7f7 (commit 7236bbd).
- [x] `Libraries/MLXLMCommon/Chat.swift:4` — Public struct `Message` lacks documentation. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:23` — Public struct `Tool` lacks documentation. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:46` — Public initializer lacks documentation. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:67` — MessageGenerator.generate(message:) silently drops images/videos/audios on the tool-calling render path. Confirmed real via code investigation (runToolCalling hand-built retokenization via DefaultMessageGenerator + raw applyChatTemplate, bypassing UserInputProcessor entirely). Fixed via ^9p675gx (commit 23abb2c) — root-caused and fixed at the actual architectural level (routing through context.processor.prepare(input:)) rather than just adding dict keys, verified end-to-end with a real RED/GREEN test plus independent re-verification including a stash/rebuild reproduction of the original bug.
- [x] `Libraries/MLXLMCommon/Chat.swift:67` — assistant(...) factory doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:78` — user(...) factory doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:99` — Role enum doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:155` — generate(message:) doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:173` — generate(messages:) doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:183` — generate(from:) doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:203` — DefaultMessageGenerator.init doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:211` — NoSystemMessageGenerator.init doc. Fixed via ^2yyn7f7.
- [x] `Libraries/MLXLMCommon/Chat.swift:215` — NoSystemMessageGenerator.generate(messages:) doc. Fixed via ^2yyn7f7.
- [ ] `Tests/MLXFoundationModelsTests/UnsupportedTranscriptContentMappingTests.swift:84,111,186` — test function names missing `test` prefix. Confirmed pre-existing (functions predate this review's scope, created during an earlier context-size-validation task) — renaming existing test functions falls under the review skill's blanket "never ask to refactor existing tests" exception. Dropped, not fixed.

## Review Findings (2026-07-09 20:57) — re-review after ^2yyn7f7/^9p675gx landed

- [ ] `Tests/MLXFoundationModelsTests/PromptCacheMultiSessionTests.swift:81` — nested-loop complexity in a pre-existing test file. Dropped under the test-refactor exception.
- [ ] `Tests/MLXFoundationModelsTests/ToolCallingImagePreservationTests.swift:138` + 4 occurrences in `UnsupportedTranscriptContentMappingTests.swift` — channel/drainer boilerplate duplicated across test files. Satisfying this fully requires editing `UnsupportedTranscriptContentMappingTests.swift`, a pre-existing test file — dropped under the test-refactor exception (the new file's own instance isn't self-duplicated; the duplication is only across files, one of which is pre-existing).

All findings are now either fixed (docs, and the real tool-calling image-drop bug — fixed at the correct architectural root, not just patched), deferred to a dedicated tracking task (^50rqt15, for the public-API first-parameter-label rename — too large/breaking a change for either source task), or dropped under the test-refactor exception (test-only naming/duplication on pre-existing test files). Moving to done.