---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g9fjpp7pthev0270daqjzj
  text: |-
    ### Preference (foundationmodelsacpagent-a1, 2026-11-27)
    If a remote trim to the divergence point isn't cheap, implement `copyPromptCache(fromSession:toSession:keepingTokens: n)` so the Router can make an instructions-only template.
  timestamp: 2026-11-28T05:19:08.502370+00:00
- actor: claude-code
  id: 01m8gcr4kh97w543t7rqfhtz3w
  text: |-
    ### Root cause (reproduced before any code change)
    Repro: Tests/MLXFoundationModelsTests/PromptCacheDivergentCopyTests.swift — tiny Inkling 2 target (Inkling2TinyModels.draftJSON, vocab 64, scalar tokenizer of Qwen35MTPFixture), local AND behind ClusterLanguageModel over InProcessRemoteModelWorker. Template pass instructions(12)+prompt(8) → 16 generated; copyPromptCache → .copied; seeded pass instructions(12)+different prompt(7). Both placements: cachedPromptTokenCount 0. Executor log (log stream, subsystem com.apple.FoundationModels-MLX):
    ```
    plan  session=divergent-copy-template source=none rendered=20 reused=0 fed=20 rule=cold
    commit session=divergent-copy-template ledger=36 checkpoint=20
    plan  session=divergent-copy-seeded source=memory rendered=19 reused=0 fed=19 rule=rebuild divergence=12
    ```
    What this rules in/out:
    - **Different key: NO.** The seeded check-out is `source=memory` (the copy, same model id, same kvCache config).
    - **Checkpoint dropped by the copy: NO.** ExecutorPromptCacheEntry.copy() carries the checkpoint (commit shows checkpoint=20 and the restore path sees it). Remote checkpoints restore fine (they are worker copies; existing cluster checkpoint tests cover that).
    - **Can't trim: YES**, and it is not remote-specific — the local run is identical. Inkling's layers are CacheList(rotating window / attention, SpeculativeConvolutionCache): the convolution state rolls back only a few verify positions and the window wraps, so neither a local nor a RemoteKVCache (status.rewindLimit/maxTrimCount = min over the worker layers) can rewind 24 tokens from 36 to 12. Same for any hybrid (the Qwen 3.5 fixture's linear layer too).
    - **No checkpoint at the instructions boundary: YES — the real cause.** The FM executor takes exactly ONE checkpoint per pass, at the end of the last message (ThinkingRender.transcriptBoundary → PromptCacheCheckpoint.lastMessageEnd), i.e. after instructions+prompt (20). It never takes the leading-instructions checkpoint that ChatSession takes (PromptCacheCheckpoints.leading via PromptCacheCheckpoint.leadingSystemEnd). A prompt that diverges right after the instructions is served by neither the rewind rule nor the only checkpoint → rule=rebuild → cold.
    - **^nztjxdh:** does not explain it. The copy returned .copied, so the template had an entry; and even after ^nztjxdh a guided pass's checkpoint is at the end of the transcript (instructions+prompt), which does not serve a prompt diverging after the instructions. The repro uses plain passes on current HEAD and still goes cold.

    ### Approach
    1. Executor takes a second checkpoint at the end of the leading instructions (PromptCacheCheckpoint.leadingSystemEnd, as ChatSession does), the entry carries it beside the transcript checkpoint (copy() shares it), a later pass that keeps the instructions carries it forward, and the plan restores the longest checkpoint that serves the render.
    2. Trimming is impossible for this cache, so also add `copyPromptCache(fromSession:toSession:keepingTokens:)`: the copy holds the first n tokens (trimmed when the caches can rewind there) or the nearest checkpoint ≤ n, and refuses n > ledger.
  timestamp: 2026-11-28T06:16:14.705491+00:00
- actor: claude-code
  id: 01m8ge8eacreqkzqzzqsw7cvrj
  text: |-
    ### implement — changed
    - Approach taken (and why): trimming the remote (or local) Inkling cache back to where the prompt diverges is impossible: the convolution state and the sliding window cannot rewind 24 tokens. So I used BOTH fixes the card allows.
      (1) The executor now takes a second checkpoint per hybrid pass, at the end of the instructions. `ThinkingRender.checkpointBoundaries` uses `PromptCacheCheckpoint.leadingSystemAndLastMessageEnds`, which reuses one sample render, so it adds no render beyond what lastMessageEnd already did. The new `ExecutorCheckpointBoundaries` replaces the `transcriptBoundary: Int?` that every pass threaded through. `ExecutorPromptCacheEntry.instructionsCheckpoint` is carried through extend, splice, rewind and restore whenever the render keeps the instructions. `copy()` shares it, and the byte and worker-byte counts include it. A restore picks the longest checkpoint that serves the render. The commit log line appends ` instructions=N`.
      (2) Added `copyPromptCache(fromSession:toSession:keepingTokens:)`. It keeps exactly n tokens when the caches can rewind and there is no model state. Otherwise it keeps the longest checkpoint ≤ n, restored from copies (`PromptCacheCheckpoint.restoredCopy(of:)`). A ledger of n tokens or fewer is copied whole, and a disk file is cloned only whole. New refusals: `.tokenCountBeyondLedger` and `.noPrefixToKeep`.
    - No remote protocol change.
    - Files: Libraries/MLXFoundationModels/{ExecutorPromptCache,ExecutorPromptCacheEntry,ExecutorPromptCacheSlot,ExecutorPromptCacheStore,MLXLanguageModel+PromptCacheCopy,MLXLanguageModel,MLXLanguageModelGeneration,ThinkingRender}.swift, README.md; Libraries/MLXLMCommon/PromptCacheCheckpoint.swift.
    - Tests added: PromptCacheDivergentCopyTests (12: Inkling, local and cluster; whole copy and keepingTokens; cached == 12 and output equals the cold run). ExecutorPromptCacheInstructionsCheckpointTests (15). PromptCacheCopyKeepingTokensTests (13). MLXLMTests/PromptCacheCheckpointEndsTests (4).
    - Existing tests updated for the API change: ExecutorCommittedGeneration, ExecutorPromptCacheCheckpoint, KVCacheConfigurationOption, PrefillStepSize, ExecutorPromptCacheMTP (its helpers now carry instructionsCheckpoint).
    - Full runs, one xctest process per bundle, all exit 0, 0 warnings: MLXFoundationModelsTests 734 passed; MLXLMTests 2849 Swift Testing + 1261 XCTest passed (3 known issues from existing withKnownIssue tests); MLXGuidedGenerationTests 96; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11. swift-format applied.
    - Public API added: `MLXLanguageModel.copyPromptCache(fromSession:toSession:keepingTokens:)`, `PromptCacheCopyResult.Refusal.tokenCountBeyondLedger`, `.noPrefixToKeep`. The new enum cases break exhaustive switches in clients such as the Router.
    - next: /review
  timestamp: 2026-11-28T06:42:37.516054+00:00
- actor: claude-code
  id: 01m8get0qq82ndsnrfnmxg2fmq
  text: |-
    ### test — green
    - evidence: swift build --build-tests (exit 0, binaries newer than every changed source, new tests present); xcrun xctest per bundle, all exit 0: MLXLMTests 2849 Swift Testing + 1261 XCTest, 0 failures; MLXGuidedGenerationTests 96, 0 failures; MLXFoundationModelsTests 734, 0 failures; CXGrammarTests 7, 0 failures; MLXHuggingFaceMacrosTests 11 XCTest, 0 failures. 0 skipped. swift-format lint on changed files clean. The only build message is a SwiftPM "missing creator for mutated node" notice, not a source warning.
    - note: MLXLMTests reports 3 known issues, all intentional withKnownIssue markers in unmodified files (AttentionKernelToleranceTests, ToolTests MiniCPM5 CDATA); not caused by these changes.
    - next: review/commit
  timestamp: 2026-11-28T06:52:13.431711+00:00
- actor: claude-code
  id: 01m8gfdw2ze1yx3a2qbxwtatht
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (1fd70743): 4 findings (4 confirmed, 8 refuted) — Libraries/MLXFoundationModels/ExecutorPromptCache.swift:265, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift:487, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift:530, Tests/MLXFoundationModelsTests/ExecutorPromptCacheInstructionsCheckpointTests.swift:38
    - next: implement the 4 reuse findings in the 2026-11-28 00:52 Review Findings section; task stays in review
  timestamp: 2026-11-28T07:03:04.031318+00:00
- actor: claude-code
  id: 01m8gfe9pexbvqktqenp87xxbb
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — instructions-boundary checkpoint + copyPromptCache keepingTokens; 4 new test files (44 tests)
    - test: green — all 5 bundles (MLXLMTests 2849+1261, MLXFoundationModelsTests 734, MLXGuidedGenerationTests 96, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11)
    - commit: 1fd70743
    - review: findings — ExecutorPromptCache.swift:265, PromptCacheCheckpoint.swift:487, PromptCacheCheckpoint.swift:530, ExecutorPromptCacheInstructionsCheckpointTests.swift:38
  timestamp: 2026-11-28T07:03:17.966544+00:00
- actor: claude-code
  id: 01m8gg1d0kek4t36b3ythrhvcz
  text: |-
    ### implement — changed
    - evidence: all 4 review findings (2026-11-28 00:52) fixed and checked, behaviour unchanged.
      1. `ExecutorPromptCacheEntry.checkpoints(longestWhere:)` returns the matching checkpoints sorted longest first. `ExecutorPromptCachePlan.restoring` and `restoredCopy(keepingTokens:)` both call it.
      2. New private `PromptCacheCheckpoint.leadingSystemEnd(of:messages:sampleTokens:)` holds the only leading-system computation. The public `leadingSystemEnd(of:messages:render:)` calls it with `renderedSample(...)`, after the same guard so it still renders nothing when there are no leading instructions. `leadingSystemAndLastMessageEnds` calls it with the shared sample.
      3. New private `leadingSystemMessages(of:) -> ArraySlice<Chat.Message>`. `startsWithSystemBeforeAnotherMessage` and `sampleConversation(of:)` both use it.
      4. New `hybridScriptedModel()` in Tests/MLXFoundationModelsTests/PromptCachePlanTestSupport.swift. ExecutorPromptCacheCheckpointTests and ExecutorPromptCacheInstructionsCheckpointTests no longer have their private `hybridModel()`. PromptCacheCopyKeepingTokensTests also had the same literal; it was new in 1fd70743 and now calls the shared builder too.
    - Files: Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Libraries/MLXFoundationModels/ExecutorPromptCacheEntry.swift, Libraries/MLXLMCommon/PromptCacheCheckpoint.swift, Tests/MLXFoundationModelsTests/{PromptCachePlanTestSupport,ExecutorPromptCacheCheckpointTests,ExecutorPromptCacheInstructionsCheckpointTests,PromptCacheCopyKeepingTokensTests}.swift
    - Tests: `swift build --build-tests` passed; its only warning is the SwiftPM "missing creator for mutated node" notice. `xcrun xctest` MLXFoundationModelsTests: 734 passed, exit 0. MLXLMTests in one process: 1261 XCTest, 0 failures, and 2849 Swift Testing passed (the 3 known issues already reported), exit 0. swift-format applied.
    - next: /review
  timestamp: 2026-11-28T07:13:43.955911+00:00
- actor: claude-code
  id: 01m8ggjqhxdx16hzreqdwb4fyr
  text: |-
    ### test — green
    - evidence: swift build --build-tests clean (no source warnings; one SwiftPM "missing creator for mutated node" tooling notice for mlx-swift_Cmlx.bundle). xcrun xctest per bundle, all exit 0: MLXLMTests 1261 XCTest (0 failures) + Swift Testing 2849 tests/282 suites passed with 3 known issues; MLXGuidedGenerationTests 96 tests/23 suites passed; MLXFoundationModelsTests 734 tests/97 suites passed; CXGrammarTests 7 tests/5 suites passed; MLXHuggingFaceMacrosTests 11 XCTest, 0 failures. swift-format lint on all changed Swift files: clean. 0 skipped.
    - pre-existing: the 3 known issues are in ToolTests.swift "Test MiniCPM5 Format via ToolCallProcessor keeps a function close tag inside CDATA" (withKnownIssue, committed in 021b9341; ToolTests.swift is not modified by this task).
    - next: review
  timestamp: 2026-11-28T07:23:11.805502+00:00
position_column: doing
position_ordinal: '80'
title: 'Prompt cache: a session seeded by copying a remote template gets no partial hit when its prompt diverges after the shared prefix'
---
## What
Bench cluster6i (Router 7bd4949f, mlx 35285ea9, Inkling on 2 Macs). The Router seeded a standard-slot session with copyPromptCache from a template, and the copy returned `.copied`. The template held a full first-session ledger: instructions, prompt and answer. The seeded session's first call had 4924 tokens in and tokens.cached=0. Its prompt shares only the instructions with the template, so the checkout should reuse the instructions and drop the rest. It went cold instead. Reported by foundationmodelsrouter-22.
^jxyd0ja's tests only extended the template, so divergence was never tested.

## Investigate first
Find the actual cause before changing code. Possible causes:
- **Can't trim:** a RemoteKVCache checkout can't trim back to the divergence point, because the Inkling cache's rewindLimit / maxTrimCount on the worker doesn't allow it (sliding window or recurrent layers).
- **Checkpoint dropped:** the plan's prompt-cache checkpoint at the instructions boundary, which a local Inkling cache would use, isn't carried by the copy, or doesn't work for remote checkpoints.
- **Different key:** a cache key or configuration mismatch between the template and the seeded session.
Compare with the same flow on a local Inkling model.

## Fix
Make the seeded session get a partial hit on the shared prefix, by trimming, by a checkpoint, or by both. If trimming is impossible for this cache, add `copyPromptCache(fromSession:toSession:keepingTokens:)`. It copies only the first n ledger tokens, or the nearest checkpoint at or below n, so the Router can make an instructions-only template. Record on the card which approach you took and why.

## Tests
- A session seeded from a remote template, whose prompt diverges after the shared instructions prefix, reports cached tokens > 0 (about the prefix length). Its output equals a cold run.
- The same test locally.
- If `keepingTokens:` is added: it copies exactly n tokens, or the nearest checkpoint at or below n, and refuses n larger than the ledger.
#cluster

## Review Findings (2026-11-28 00:52)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 19 file(s) reviewed, 3 not reviewed.

> 3 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8G9FF1CS1H0HWZ9H3WJDPA4.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8G9FF1CS1H0HWZ9H3WJDPA4.md` — no validator matches this file
> - `Libraries/MLXFoundationModels/README.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:265` `reuse/reuse` — `restoring` re-implements the 'longest checkpoint at or below a token count' selection that `ExecutorPromptCacheEntry.restoredCopy(keepingTokens:)` also implements. It filters `entry.checkpoints` by a predicate, sorts by `tokens.count` descending, and takes the first checkpoint that restores. Two copies of this ordering rule can drift apart, and a fix to one would not reach the other. Add one helper on `ExecutorPromptCacheEntry`, for example `checkpoints(longestWhere: (PromptCacheCheckpoint) -> Bool) -> [PromptCacheCheckpoint]` that returns the matching checkpoints sorted longest first. Both `restoring` and `restoredCopy(keepingTokens:)` then call it and keep only the restore step local to each.
- [x] `Libraries/MLXLMCommon/PromptCacheCheckpoint.swift:487` `reuse/reuse` — `leadingSystemAndLastMessageEnds` re-implements the body of `leadingSystemEnd` (the system-prefix check, the sample-token common-prefix computation) instead of sharing it. The private `lastMessageEnd(of:messages:sampleTokens:render:)` overload already shows the pattern for taking a precomputed render, but `leadingSystemEnd` has no equivalent, so the two copies can drift apart. Extract a private `leadingSystemEnd(of promptTokens: [Int], messages:, sampleTokens: [Int]?)` overload, mirroring the `lastMessageEnd` overload. Have the public `leadingSystemEnd(of:messages:render:)` call it with `renderedSample(...)`, and have `leadingSystemAndLastMessageEnds` call it with the shared `sampleTokens`. That leaves one implementation of the leading-system computation.
- [x] `Libraries/MLXLMCommon/PromptCacheCheckpoint.swift:530` `reuse/reuse` — `sampleConversation(of:)` recomputes the leading-system slice `messages.prefix { $0.role == .system }` that `startsWithSystemBeforeAnotherMessage` already computes. Both new helpers carry their own copy of the same predicate, so a change to what counts as leading instructions would have to be made in two places. Extract a private `leadingSystemMessages(of messages: [Chat.Message]) -> ArraySlice<Chat.Message>` (or `[Chat.Message]`) helper. Have `startsWithSystemBeforeAnotherMessage` and `sampleConversation(of:)` both call it, so the definition of leading instructions lives in one place.
- [x] `Tests/MLXFoundationModelsTests/ExecutorPromptCacheInstructionsCheckpointTests.swift:38` `reuse/reuse` — `hybridModel()` re-declares the same one-attention-layer, one-recurrent-layer `ScriptedLanguageModel` fixture that `ExecutorPromptCacheCheckpointTests.hybridModel` already provides. The copy is a parallel definition, not a call to the shared one. Keep one hybrid-model builder in the shared test support (beside `PromptCachePlanFixtures`) and call it from both suites. Do not declare the `ScriptedLanguageModel` literal again in each file.
