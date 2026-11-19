---
assignees:
- claude-code
position_column: done
position_ordinal: ffffcb80
title: Copy the prompt cache of one session to a new session (D205 f)
---
Request from TestudoApp (D205 (f)). Brief: /Users/wballard/github/swissarmyhammer/.briefs/fork-kv/mlx-brief.md.

## What

Add `MLXLanguageModel.copyPromptCache(fromSession:toSession:) async -> PromptCacheCopyResult`.

- Copy each key of `modelID` and the source session (each `kvCache` configuration) to the same key with the destination session.
- Memory and spilling tiers: make a new entry with new `KVCache` objects (`copy()`). The arrays share their buffers until a session writes (copy-on-write in MLX). Copy `drafterState` with `MTPDrafterContinuation.copy()`. The checkpoint is read-only, thus the two entries share it.
- Disk tier: clone the file (APFS `copyItem`) to a new file of the destination. The handle records the key of the file metadata, thus the read checks the source key.
- A source that a pass holds: return `.sourceInUse` at once (no wait). The store counts the check-outs of each key.
- The destination has a cache, or a pass holds it, or it is the source: `.refused(reason)`.
- Memory accounting: the copy counts its full `byteCount` (conservative), it is the most recently used entry, and `evictToBudget` applies.

## Acceptance criteria

- [ ] The public API and `PromptCacheCopyResult` with doc comments.
- [ ] The source does not change; release of one session does not change the other.
- [ ] Tests from the brief (see the list in the brief) in `PromptCacheCopyTests.swift` and `PromptCachePublicAPITests.swift`.
- [ ] README (prompt cache section) and the doc of `PromptCacheScope` updated.
- [ ] All five test bundles green.
- [ ] Push to `stable`; send the sha to foundationmodelsrouter-f1 and testudoapp-9a.