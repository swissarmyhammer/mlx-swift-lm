---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8fe4anj5px0bsrwr0n1dfvy
  text: |-
    ### Requirement change (from foundationmodelsacpagent-a1, 2026-11-27)
    Every checked-in remote entry is evictable by LRU, including a live session's only entry, and including the entry just checked in. In a bench run every session is live, so exempting those would mean the budget never fires. Only an entry that is checked out (in use by a generation) is exempt, because it isn't in the store. Dropping an entry costs a cold prefill on that session's next turn, which is correct, just slower.
    Replaces the description's "never evict the entry being checked in if it alone fits; keep it only if it's the only remote entry" and the test "a remote entry over budget by itself, alone in the store, stays". The new tests:
    - N live sessions, one entry each, over the budget: the oldest is released on the host.
    - A single remote entry over the budget by itself is released at check-in.
  timestamp: 2026-11-27T21:21:08.274610+00:00
- actor: claude-code
  id: 01m8fepn3gxq375mk59m5x80z6
  text: |-
    ### Research / budget source
    - No per-worker KV reservation exists in this repo. Searched Libraries/MLXLMCommon/Remote (ClusterLanguageModel, ClusterModelLoading.loadClusterModelContext, RemoteModelProtocol RemoteWorkerCapabilities/RemoteModelCapabilities, RemoteModelExecutor) and the docc distributed-inference article: the capabilities carry no memory/reservation field, and there is no cluster plan type; the planner lives outside this package. The old doc on ExecutorPromptCacheEntry.byteCount ("the cluster plan reserves them there") was aspirational and is now replaced.
    - Budget source used: configurable store setting. `ExecutorPromptCacheStore.workerBudgetBytes`, default `ExecutorPromptCacheStore.defaultWorkerBudgetBytes` = 8 GiB (`8 << 30`), set by `ExecutorPromptCacheStore.configure(workerBudgetBytes:)` and the public `MLXLanguageModel.configurePromptCache(workerBudgetBytes:)` (next to the memory/disk budget setters). A host that knows its cluster plan's reservation passes it there.
    - Code path: checkIn -> keepInMemory -> evictToBudget (local, unchanged) -> releaseToWorkerBudget: while `workerByteCount` (sum of entry.workerByteCount) > workerBudgetBytes, `entries.removeLeastRecentlyUsed(where: \.holdsRemoteCache)` (new unconditional LRU-tier method; the whileOver form now calls it) and log `workerReleaseLine`; dropping the last reference makes each RemoteKVCache.deinit send `.release(id)` to the worker. Never spilled. Applies to the entry just checked in too (requirement change in the comment above).
    - entry.workerByteCount = sum of `KVCache.remoteResidentByteCount` (new, residentByteCount for RemoteKVCache, 0 local) + `PromptCacheCheckpoint.workerByteCount` (remote copies in the checkpoint are separate worker caches). ExecutorPromptCacheEntry.copy() shares the checkpoint, so a copy counts the checkpoint's worker bytes again: same conservative over-count as the local byteCount's minimumByteCount.
    - Test gotcha: `Mutex` needs macOS 15 and @Suite types cannot be @available, so the recording transport uses OSAllocatedUnfairLock.
  timestamp: 2026-11-27T21:31:08.784923+00:00
- actor: claude-code
  id: 01m8ff58gqgfctndm39fky97p0
  text: |-
    ### implement — changed
    - evidence: 10 files. Libraries: MLXFoundationModels/ExecutorPromptCacheStore.swift (workerBudgetBytes, defaultWorkerBudgetBytes = 8 GiB, workerByteCount, configure(workerBudgetBytes:), releaseToWorkerBudget after evictToBudget in keepInMemory), ExecutorPromptCacheEntry.swift (workerByteCount), ExecutorPromptCacheTier.swift (removeLeastRecentlyUsed(where:)), ExecutorPromptCache.swift (workerReleaseLine), MLXLanguageModel.swift (public configurePromptCache(workerBudgetBytes:)), MLXLMCommon/PromptCacheCheckpoint.swift (workerByteCount), MLXLMCommon/Remote/RemoteKVCache.swift (remoteResidentByteCount). Tests: ExecutorPromptCacheRemoteEntryTests.swift (new suite ExecutorPromptCacheWorkerBudgetTests, 12 tests, and a shared RecordingStatusTransport that records `.release`), ExecutorPromptCacheTierTests.swift (+2), PromptCachePublicAPITests.swift (+1). The local budget, eviction and spill paths are unchanged.
    - Budget source: a configurable store setting, because the repo has no cluster-plan KV reservation (see the research comment).
    - RED seen: 7 runtime failures against the stubs, plus compile-RED for the tier and log-line APIs. GREEN: swift build --build-tests is clean with no warnings. xcrun xctest: MLXLMTests 2788 ST + 1261 XCT passed (3 known issues are existing withKnownIssue tests), MLXGuidedGenerationTests 96, MLXFoundationModelsTests 643, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11. swift-format ran. Not committed.
    - next: /review
  timestamp: 2026-11-27T21:39:07.415879+00:00
- actor: claude-code
  id: 01m8ffmst6yvbkqmmpfwq16x3p
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (only the pre-existing SwiftPM "missing creator for mutated node" notice); swift-format lint clean on all changed Swift files. xcrun xctest per bundle: MLXLMTests 1261 XCTest + 2788 Swift Testing, 0 failures; MLXGuidedGenerationTests 96 passed; MLXFoundationModelsTests 643 passed; CXGrammarTests 7 passed; MLXHuggingFaceMacrosTests 22 XCTest passed (7+3+11 suites counted by xctest, 0 failures). No skips.
    - note: MLXLMTests reports 3 known issues (withKnownIssue in AttentionKernelToleranceTests.swift and ToolTests.swift MiniCPM5 CDATA); neither file is in the working-tree diff, so they pre-exist and are not caused by this change.
    - next: review
  timestamp: 2026-11-27T21:47:36.646312+00:00
position_column: doing
position_ordinal: '80'
title: 'Clustered models: bound the worker memory that kept RemoteKVCache entries hold in the prompt cache store'
---
## What
Since ^xz9zxyp (b15b6209), ExecutorPromptCacheStore keeps a RemoteKVCache entry across check-ins and never budget-evicts or spills it; a remote entry leaves only at check-out, a newer check-in of the same session, or host release. Nothing bounds the worker-side bytes. The SWE-bench bench runs a root plus several child agent sessions at once; 5–10 long sessions at 50k+ tokens each keep their own remote entry, so the workers hold 5–10 large KV caches — the worker OOM that killed cluster1i. (Raised by foundationmodelsacpagent-a1, 2026-11-27.)

## Fix
- Give the store a worker-side byte budget for remote entries, measured by `RemoteKVCache.residentByteCount` (worker bytes), separate from the head-process budget (which still counts only local bytes). Source it from the cluster plan's per-worker KV reservation if one exists in this repo (find the code path); otherwise a configurable store setting with a safe default, and say on the card which it is.
- When a check-in puts the remote entries over that budget, release the least-recently-used remote entries on the host (dropping frees the worker caches) until under budget. Never spill a remote entry to disk. Every checked-in remote entry is evictable by LRU, including a live session's only entry and the entry just checked in (requirement change, 2026-11-27, see comments). Only a checked-out entry is exempt, because it is not in the store.
- Local-cache budget, eviction and spill behavior unchanged.

## Tests
- N live sessions, one remote entry each, together over the worker budget: after the Nth check-in the oldest remote entry is released on the host (its worker caches freed, the stub transport sees the release) and the newest N-1 that fit stay; checkOut of the oldest is cold, of the others `.memory`.
- A single remote entry over the worker budget by itself is released at check-in.
- Existing ExecutorPromptCacheRemoteEntryTests still pass; local spill behavior unchanged.
#cluster