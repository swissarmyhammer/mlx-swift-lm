---
assignees:
- claude-code
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
position_column: todo
position_ordinal: '8380'
title: 'MTP: drafter state through trim, hybrid checkpoint restore and fork in ChatSession'
---
## What
After the ChatSession MTP task, the append-suffix case works. These in-session cache operations must also keep the MTP drafter state (`MTPDrafterCacheState`: drafter cache + pending hidden row) in step with the main cache. Snapshot save/load is a separate task. A drafter state cannot be trimmed by itself (entry j needs target hidden j, and no cache keeps old hidden rows), so it is restored from a checkpoint at the same position or dropped for a cold start.

- [ ] **Checkpoint creation** (ChatSession.swift ~1562-1642, and the `prefillToCheckpoint` end-of-instructions checkpoint): add a `copy()` of the drafter state at the checkpoint position to `PromptCacheCheckpoint` (Libraries/MLXLMCommon/PromptCacheCheckpoint.swift ~33-65), and add its bytes to the checkpoint byte count.
- [ ] **Trim to a common prefix P / checkpoint restore** (ChatSession.swift ~1429-1450 and ~1508-1528): Qwen3.8 is hybrid, so a rewind restores a checkpoint. Restore the drafter state of that checkpoint together with the main caches. If no checkpoint gives an aligned drafter state, use the cold start. Never use a misaligned drafter.
- [ ] Make sure that the MTP drafter state does not set `carriesModelState` in `PromptCacheTurn`. Update `PromptCacheReusePolicy` (Libraries/MLXLMCommon/PromptCacheReusePolicy.swift) so that an MTP session with a checkpoint restore keeps speculation on and does not use `requiresMainOnlyContinuation`. `maxTrimCount` takes the minimum over main and draft caches (~1421-1422); do not include the MTP drafter in it.
- [ ] **Fork / copy** (`RealizedCache.copy()` ~407-410, fork ~1954-1960): copy the MTP drafter state with the session. A fork with a new history sets `transcriptBuiltCache = false`; the drafter state follows the same rule as the main cache.

## Acceptance Criteria
- [ ] Edit-and-regenerate (a turn that rewinds to an earlier message end) with MTP gives the same greedy text as plain decode, and does not feed the drafter again over the kept prefix.
- [ ] A forked session and its source continue independently with MTP, and each gives the same text as plain decode.
- [ ] A rewind with no aligned drafter checkpoint uses the cold start and still gives the same text as plain decode.

## Tests
- [ ] Tests/MLXLMTests: ChatSession tests on the scripted Qwen35 + MTP fixture for checkpoint restore, cold-start fallback, and fork.
- [ ] Tests/MLXLMTests: a `PromptCacheReusePolicy` unit test where an MTP drafter state keeps rewind allowed.
- [ ] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #chat-session