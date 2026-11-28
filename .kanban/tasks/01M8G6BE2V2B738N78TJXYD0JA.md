---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g6tbtz49pdgb00qmhx77wd
  text: |-
    Research + implementation notes:
    - RemoteModelSession.send recorded a `.failed` reply as the session failure, so a copy the worker refused poisoned the model ("source stays usable" was impossible). RemoteModelExecutor.copy is atomic (makes the target handle only on success), so a `.failed` copy leaves the worker unchanged.
    - Added RemoteModelSession.performAtomic(_:) (a `.failed` reply throws workerFailed but is not kept; transport failures/lost members still poison). RemoteKVCache.copyOnWorker() throws and makes no mirror on failure (so no release of a never-made handle); RemoteKVCache.copy() wraps it and records the failure as before (KVCache.copy() behaviour unchanged). KVCache.checkedCopy() = copy() for local caches, copyOnWorker() for remote.
    - ExecutorPromptCacheEntry.copy() now throws; the store's copyEntry routes memory/spilling copies through keepCopy(of:of:under:), which logs and returns nil on a failed copy -> `.refused(.copyFailed)` with no destination entry, source entry untouched.
    - InProcessRemoteModelWorker: new FailureKind `.copies` (fail(.copies, after:with:)).
    - Card wording "budget set to fit one entry" cannot hold literally: the source and its copy must both stay, so the test budget fits two remote entries (source + copy), and the copy releases the older third entry, not the source.
    - Coverage tests that passed before the fix (existing behaviour): budget release of the LRU entry / source kept, refused copy keeps source, refused copy releases no handle, member loss surfaces at next beginRequest, and the 4 in-process copy tests (ledger, second handle, same continuation, cached prefix). RED before fix: refused -> copyFailed, refused -> no destination, refused -> model usable, lost member -> copyFailed, in-process fail(.copies) -> copyFailed and no destination.
    - `swift build` prints "warning: missing creator for mutated node: .../mlx-swift_Cmlx.bundle/Contents/MacOS" — a build-graph warning from the swiftbuild system, not from source.
  timestamp: 2026-11-28T04:32:36.191377+00:00
- actor: claude-code
  id: 01m8g7a7rarsz1j9bc25s97tz9
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests clean (only the pre-existing swiftbuild "missing creator for mutated node ... mlx-swift_Cmlx.bundle" graph warning). xcrun xctest MLXFoundationModelsTests: 687 Swift Testing tests passed, 0 failures. xcrun xctest MLXLMTests (one process): XCTest 1261/0 failures; Swift Testing 2845 passed, 3 known issues (pre-existing withKnownIssue in AttentionKernelToleranceTests and MiniCPM5ToolCallParserTests, files I did not touch). swift-format applied + lint clean.
    - files: Libraries/MLXLMCommon/Remote/RemoteModelTransport.swift (performAtomic, send keepsWorkerFailure), Libraries/MLXLMCommon/Remote/RemoteKVCache.swift (copyOnWorker, KVCache.checkedCopy, copy() wraps), Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift (copy() throws), Libraries/MLXFoundationModels/ExecutorPromptCacheStore.swift (keepCopy; docs), Libraries/MLXFoundationModels/MLXLanguageModel+PromptCacheCopy.swift (docs), Tests/MLXLMTestSupport/InProcessRemoteModelWorker.swift (FailureKind.copies), Tests/MLXFoundationModelsTests/ExecutorPromptCacheRemoteEntryTests.swift (new suite ExecutorPromptCacheRemoteCopyTests, 9 tests; stub transport CopyAnswer), Tests/MLXFoundationModelsTests/ClusterPromptCacheCopyTests.swift (new, 6 in-process worker tests).
    - next: /review
  timestamp: 2026-11-28T04:41:16.298882+00:00
- actor: claude-code
  id: 01m8g7vntta2b3jttbz0e03gjm
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (0 compiler warnings); xcrun xctest per bundle: MLXLMTests 2845 Swift Testing (3 pre-existing known issues in AttentionKernelToleranceTests and ToolTests MiniCPM5, untouched files) + 1261 XCTest, 0 failures; MLXGuidedGenerationTests 96; MLXFoundationModelsTests 687; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 XCTest. swift-format lint clean on all 8 changed files. 0 skipped.
    - next: review/commit
  timestamp: 2026-11-28T04:50:47.770282+00:00
position_column: doing
position_ordinal: '80'
title: 'Prompt cache: copying a session that holds a RemoteKVCache is untested, and a failed worker copy reports .copied instead of copyFailed'
---
## What
FoundationModelsRouter plans to seed each new session's cache from a template session, so selection calls stop re-prefilling a ~7.8k-token prefix. It will do this through `copyPromptCache(fromSession:toSession:)`, which calls `ExecutorPromptCacheStore.copy` and then `ExecutorPromptCacheEntry.copy()`. For a clustered model the path already works in code:
- `RemoteKVCache.copy()` (Libraries/MLXLMCommon/Remote/RemoteKVCache.swift:110) sends `RemoteModelOp.copy(id, into:)`.
- The worker (`RemoteModelExecutor.copy`, RemoteModelExecutor.swift:320) copies every layer and the drafter state into a new handle.
- The copy goes through `keepInMemory`, so `releaseToWorkerBudget()` counts the copy's worker bytes against `workerBudgetBytes`.

Two gaps:
1. Nothing tests a copy of a remote entry. PromptCacheCopyTests covers local caches only.
2. When the worker's copy op fails, `RemoteKVCache.copy()` records the failure on the session and returns a mirror under a handle the worker never made. The store then returns `.copied(tokenCount:)`, the destination session holds a dead entry, and the failure only surfaces at the next `beginRequest`.

## Fix
- A failed remote copy makes `ExecutorPromptCacheStore.copy` return `.refused(.copyFailed)` and keep no entry for the destination. For example, check `ClusterLanguageModel.checkFailure()`, or have the entry copy throw for a remote cache.
- The source entry stays usable if the failure was the copy op only. A lost member still surfaces as `memberLost` at the next request.
- No protocol change.

## Tests
- Copying a session that holds a remote entry gives the destination an entry whose next checkOut is `.memory` with the full ledger. The worker holds a second handle, and source and destination generate the same continuation independently, so a write to one does not change the other.
- The copy counts against `workerBudgetBytes`: with the budget set to fit one entry, copying releases the least recently used remote entry, which is not the source.
- A worker that fails the copy op makes `copy` return `.refused(.copyFailed)` with no destination entry.
#cluster