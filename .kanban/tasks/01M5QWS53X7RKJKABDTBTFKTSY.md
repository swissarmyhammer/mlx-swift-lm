---
assignees:
- claude-code
depends_on:
- 01M5QWHA34WP7DQMWQ4AHNB0J9
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: todo
position_ordinal: '8980'
title: 'MTP: save and load the drafter state with PromptCacheSnapshot'
---
## What
A session snapshot (`PromptCacheSnapshot`, `savePromptCache`, `loadPromptCache` in Libraries/MLXLMCommon/KVCache.swift ~2340-2400, ~2915-3026) holds only the main cache and the model state; `saveCache` in ChatSession saves `main` and `state` only (ChatSession.swift ~2026-2035). A session restored from it with MTP has no drafter to reuse.

- [ ] Add an optional `MTPDrafterCacheState` to `PromptCacheSnapshot`. Write it with the `arrays()` / `metadata()` of the drafter-state task under its reserved key prefix, and read it back in `loadPromptCache` with `init(arrays:metadata:drafter:)`. The main-cache layer-count check (~2496) must ignore the drafter keys.
- [ ] Make the ChatSession snapshot export and the `init(_:promptCache:speculativeDecoding:...)` initializers (ChatSession.swift ~839-885) carry the drafter state. A session restored with an MTP config must reuse the drafter without a new prefill.
- [ ] Make sure that old snapshot files, which have no drafter keys, still load. Then the drafter uses the cold start.

## Acceptance Criteria
- [ ] Save a session snapshot, load it, and continue with MTP: the text is the same as with no save/load, and the drafter is reused (no drafter feed over the cached prefix).
- [ ] A snapshot written before this change loads, and generation is correct.

## Tests
- [ ] Tests/MLXLMTests: a `savePromptCache` / `loadPromptCache` round-trip unit test with the drafter keys, and a test with an old-format file.
- [ ] Tests/MLXLMTests: a ChatSession snapshot round-trip test on the scripted Qwen35 + MTP fixture.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session