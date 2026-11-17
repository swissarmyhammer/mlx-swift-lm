---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7mqtcjnxebxpan26xbxk9nc
  text: "Research (implement): \n- `Executor.makeParameters` already takes `prefillStepSize` with a default from the task-local `Executor.prefillStepSize`, bound in `respond` from `model.prefillStepSize`. KV config will follow the same pattern: `@TaskLocal static var kvCache`, bound from `model.kvCache`, defaulted into `makeParameters(kvCache:)`. All four generation sites in MLXLanguageModelGeneration.swift call `makeParameters` with defaults, and the plan/restore/new-cache paths take those parameters, so they pick it up. The guided loop (`runGuidedLoop` → `GuidedGenerationLoop.run`) builds its own `GenerateParameters` from `kvCache:` — it gets `Self.kvCache` too, like it gets `Self.prefillStepSize`.\n- Warm-up: plain `GenerateParameters(maxTokens: 1, prefill:)` and speculative `GenerateParameters(maxTokens:temperature: 0)` both need `kvCache:`.\n- Prompt cache: `ExecutorPromptCacheKey(modelID, sessionID)`; the plan reuses `entry.caches` without looking at parameters, and generation then applies the request's `KVCachePlan` dynamically, so a full-precision cache would be silently reused under affine 8 (and an already-quantized cache under nil). Spilled files are restored by the store's key → handle mapping. Decision: put `kvCache: KVCacheConfiguration?` into `ExecutorPromptCacheKey` (Hashable). Spill/disk tiers are keyed the same way, so no file metadata change is needed (file names carry a store-wide generation, so two configs of one session never collide). `releasePromptCache(sessionID:)` will release the session under every KV configuration of the model, so a session that switched configuration leaves nothing behind.\n- Deprecated `init(..., load: ContainerLoader)` has no callers in Tests/IntegrationTesting that I could find; it is left as is (optional on the card) and gets no `kvCache:`.\n- Unit-test caches have head dim 1, so affine (group 64) skips them as `unsupportedShape`; tests use `compatibility: .allowPartial`."
  timestamp: 2026-11-17T12:30:58.645396+00:00
- actor: claude-code
  id: 01m7mrjpeadgthjdd57jmw1388
  text: 'Discovery (blocker for KV8 reuse, fixed in scope): the executor hands `plan.caches` (a `[KVCache]` array) to generation, and generation wraps them in a `KVCacheStorage` whose `KVCachePlan.apply/applyAndValidate` REPLACES each `KVCacheSimple` leaf with a new `QuantizedKVCache` object after the prefill (`maybeAffineQuantizeKVCache` via `KVCacheTree.rewrite`). The executor then commits its own stale array: full-precision `KVCacheSimple` objects frozen at the prompt end. Effects: (a) a non-hybrid model checks in a full-precision cache under an 8-bit key and re-feeds the whole answer next turn; (b) a hybrid model (Qwen3.8: recurrent layers advance, attention stale) is refused with `positionsDisagree`, so turn 2 never reuses. Two unit tests reproduce both (RED confirmed): `an 8-bit pass checks in the caches its generation compressed`, `a second 8-bit turn of a hybrid model reuses the prefix of the first`. Fix: MLXLMCommon package-only — iterators expose the caches their generation last wrote (`GenerationCacheProviding`, TokenIterator + MTPSpeculativeTokenIterator), `RecordingGeneratedTokenList` hands them over at the end beside the drafter continuation, `generateProtocolTokensTask` / `generateTaskRecordingTokens` take a defaulted `generationCache:` callback, and the executor commits those caches in place of the ones it handed over. Pre-converting empty caches up front was rejected: an empty cache is never converted (offset must exceed compressionStart), and it would change prefill numerics vs. ChatSession.'
  timestamp: 2026-11-17T12:44:15.178747+00:00
- actor: claude-code
  id: 01m7mseh1bwna5qgprpekgyek3
  text: |-
    Real-model evidence (IntegrationTestingTests/KVCacheSessionPromptCacheTests, affine 8-bit gs64, LanguageModelSession, greedy; bench.lock held; 4/4 passed in 19.6 s; unified log `KV8 SESSION:`):
    - Qwen3.8-27B-mxfp4 two turns: turn 1 rendered 73, turn 2 cached 97 of 122, 16 QuantizedKVCache layers in the checked-in entry; turn 1 "OK", turn 2 "teal".
    - Flint oQ4e two turns: turn 1 rendered 23, turn 2 cached 63 of 78, 14 compressed layers; "OK" / "teal".
    - Qwen KV8 PromptCacheCheckpoint (dropped generated turn): cached 69 of 92 == checkpoint 69; "teal".
    - Qwen KV8 disk spool round trip (memory budget 0): 157,344,423 disk bytes after turn 1, turn 2 cached 97 of 122; "teal".
    IntegrationTesting did not build on stable before this card. I fixed three call sites only so this run could build: `Qwen38FourBitMTPModel.containerLoader` was still typed as the deprecated `ContainerLoader`, and two DeepSeek `switch` statements had no case for `Generation.steering`.
  timestamp: 2026-11-17T12:59:27.147875+00:00
- actor: claude-code
  id: 01m7msepzj1a12ybqps91jqycr
  text: |-
    ### implement — changed
    - evidence: Libraries: MLXFoundationModels/{MLXLanguageModel,MLXLanguageModelGeneration,ExecutorPromptCache}.swift; MLXLMCommon/{TokenIterators,MTPSpeculativeTokenIterator,GenerationLoop,GenerateTokens,Evaluate}.swift (package-only `GenerationCacheProviding` / `generationCache:` hand-off). Tests: new MLXFoundationModelsTests/KVCacheConfigurationOptionTests.swift (17 tests), new MLXLMTests/GenerationCacheHandOffTests.swift (3), updated ScriptedModelTestSupport, PrefillStepRecordingModel, ExecutorPromptCacheTests. IntegrationTesting: new KVCacheSessionPromptCacheTests.swift, FMTestHelpers `kvCache:`, sessionCacheKey callers, plus the build fixes above. xcrun xctest: MLXLMTests 2356 Swift Testing + 1098 XCTest (4 skips are the LFM2 ones tracked in ^gcsvy3f; 1 profile suite skip and 3 known issues were already there), MLXGuidedGenerationTests 78, MLXFoundationModelsTests 514, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 — 0 failures. No compiler warnings; the SwiftPM "missing creator for mutated node" build-system notice was already in earlier build logs. swift-format run. Real model: 4/4 KV8 integration tests on Qwen3.8 mxfp4 + Flint oQ4e.
    - Deprecated `init(..., load: ContainerLoader)` kept, sets `kvCache = nil`.
    - next: /review; after push, reply to FoundationModelsRouter with the stable SHA (acceptance item 4).
  timestamp: 2026-11-17T12:59:33.234527+00:00
- actor: claude-code
  id: 01m7mt02nk9386yvx35vjmnepw
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (811e4da7); 9 findings (9 confirmed, 11 refuted) — Libraries/MLXFoundationModels/ExecutorPromptCache.swift:31, Libraries/MLXFoundationModels/ExecutorPromptCache.swift:2208, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:857, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1230, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1385, Libraries/MLXLMCommon/Evaluate.swift:581, Libraries/MLXLMCommon/Evaluate.swift:594, Libraries/MLXLMCommon/GenerateTokens.swift:331, Tests/MLXLMTests/GenerationCacheHandOffTests.swift:13
    - next: implement the Review Findings (2026-11-17 07:00) checklist; task stays in review
  timestamp: 2026-11-17T13:09:02.259376+00:00
- actor: claude-code
  id: 01m7mt0r0fcxyb2hbfcjafddy4
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 8 lib files, 2 new test files, IntegrationTesting KVCacheSessionPromptCacheTests + 3 build fixes
    - test: green — run in the implement step: 5 bundles 0 failures (MLXLMTests 2356+1098, Guided 78, FM 514, CXGrammar 7, Macros 11); real-model KV8 4/4
    - commit: 811e4da7 (kv-config)
    - review: findings — ExecutorPromptCache.swift:31, ExecutorPromptCache.swift:2208, MLXLanguageModelGeneration.swift:857, :1230, :1385, Evaluate.swift:581, :594, GenerateTokens.swift:331, GenerationCacheHandOffTests.swift:13
  timestamp: 2026-11-17T13:09:24.111527+00:00
- actor: claude-code
  id: 01m7mt8nvcm4dhvbc62v5sw6ew
  text: "Implement (review findings 2026-11-17 07:00): \n- Finding 1: `ExecutorPromptCacheKey.kvCache` is read only by the synthesized `Hashable`; the dead-code-swift rule text says not to delete such a property and to mark it `// periphery:ignore` with the reason on the line above. Done that way (deleting it would collapse two KV configurations into one key). Adding it to the log line was rejected: the plan/commit/spill log lines are asserted verbatim by existing tests, so that would change behavior.\n- Findings 2-5: `ExecutorDrafterStateReceiver` + `ExecutorGenerationCacheReceiver` replaced by one generic `ExecutorHandOffReceiver<Value: Sendable>` (Mutex, `receive`, consuming `take()`), and `ExecutorGenerationHandOff` bundles the two receivers, exposes `generationHandOff` for the generation call and `commit(_:generatedTokens:state:to:)` for `withPromptCacheCommit`. Each of the three passes now makes one `ExecutorGenerationHandOff`.\n- Findings 6-8: new `package struct GenerationHandOff` (MLXLMCommon, GenerationLoop.swift) carries both callbacks; `generateTaskRecordingTokens` and `generateProtocolTokensTask` take `handOff: GenerationHandOff = GenerationHandOff()` and both build `RecordingGeneratedTokenList(handOff:)`. MTPSpeculationTests caller updated.\n- Finding 9: the test recorder holds an `OSAllocatedUnfairLock` and is plain `Sendable`. `Mutex` does not compile in MLXLMTests (package floor macOS 14; Mutex is macOS 15), OSAllocatedUnfairLock is the prevailing lock in that bundle.\n- IntegrationTesting references none of the changed symbols (grep)."
  timestamp: 2026-11-17T13:13:44.044912+00:00
- actor: claude-code
  id: 01m7mtq1t5sw7cxevq9nsspkdc
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXFoundationModels/{ExecutorPromptCache,MLXLanguageModelGeneration}.swift, Libraries/MLXLMCommon/{GenerationLoop,GenerateTokens,Evaluate}.swift, Tests/MLXLMTests/{GenerationCacheHandOffTests,MTPSpeculationTests}.swift, new Tests/MLXFoundationModelsTests/ExecutorHandOffReceiverTests.swift (3 tests, RED = missing type, then green). 9/9 findings checked. swift build --build-tests: 0 compiler warnings (only the pre-existing SwiftPM "missing creator for mutated node" notice). xcrun xctest: MLXFoundationModelsTests 517 passed, MLXLMTests 2356 Swift Testing + 1098 XCTest passed (4 LFM2 skips ^gcsvy3f, 3 known issues already there), MLXGuidedGenerationTests 78, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 — 0 failures. IntegrationTesting xcodebuild build-for-testing: TEST BUILD SUCCEEDED. swift-format run. Hand-off semantics unchanged (same commit order, same values), so no real-model rerun.
    - next: /review
  timestamp: 2026-11-17T13:21:35.045545+00:00
- actor: claude-code
  id: 01m7mtvr7y7gt2xv6wfp1fgtxr
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (a0d5bff2) — 2 findings (2 confirmed, 1 refuted): Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:850, Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1372. All 9 items of the 07:00 section are addressed (generic ExecutorHandOffReceiver<Value> with take(), ExecutorGenerationHandOff bundle, package GenerationHandOff + RecordingGeneratedTokenList(handOff:), periphery:ignore on kvCache key, no @unchecked Sendable in GenerationCacheHandOffTests).
    - next: implement the shared helper that owns prefill recorder, ExecutorGenerationHandOff, startTrackedGeneration and withPromptCacheCommit across the three pass functions; re-review.
  timestamp: 2026-11-17T13:24:09.086527+00:00
- actor: claude-code
  id: 01m7mtw2fccc85s74snf87nxx8
  text: |-
    ### finish iteration 2 — findings
    - implement: changed — 8 files, 9/9 findings checked
    - test: green — run in the implement step: FM 517, MLXLMTests 2356+1098, Guided 78, CXGrammar 7, Macros 11, 0 failures; IntegrationTesting builds
    - commit: a0d5bff2 (kv-config)
    - review: findings — MLXLanguageModelGeneration.swift:850, MLXLanguageModelGeneration.swift:1372
  timestamp: 2026-11-17T13:24:19.564419+00:00
- actor: claude-code
  id: 01m7mv8rjp1qyxjtmrm0wajgw4
  text: 'Implement (review findings 2026-11-17 07:22): `withPromptCacheCommit` (sole callers: the three passes) replaced by one instance helper `withCommittedGeneration(plan:promptCache:generate:consume:)` in MLXLanguageModelGeneration.swift. It owns the `ExecutorPrefillRecorder`, the `ExecutorGenerationHandOff`, `startTrackedGeneration`, the error-exit `cancelAndDrain`, and the commit with `prefill.state`. `generate` receives `(preparedState: PreparedGenerationState, handOff: GenerationHandOff)` and returns the stream + task; `consume` receives `(stream, task)` (task needed by the allowed-tool and reasoning passes for `task.cancel()` on early stop). runAllowedToolGeneration / runUnconstrained / runReasoning now supply only their generator call and response loop. Order of operations unchanged: recorder + hand-off created, generation started in the drain section, stream read, error exit cancels and drains, good exit awaits tokens then commits. New test Tests/MLXFoundationModelsTests/ExecutorCommittedGenerationTests.swift (RED = missing member compile error, then green): the prefill a generation reports through the helper becomes the checkpoint of the committed cache.'
  timestamp: 2026-11-17T13:31:15.414897+00:00
- actor: claude-code
  id: 01m7mvcp0st261mgmjv9s5xwk2
  text: |-
    ### implement — changed
    - evidence: Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift (`withPromptCacheCommit` → `withCommittedGeneration(plan:promptCache:generate:consume:)`, three passes reduced to generator call + response loop), new Tests/MLXFoundationModelsTests/ExecutorCommittedGenerationTests.swift (1 test, RED = missing member, then green). 2/2 findings of the 07:22 section checked. swift build --build-tests: 0 errors, 0 compiler warnings. xcrun xctest: MLXFoundationModelsTests 518 passed; MLXLMTests 2356 Swift Testing passed (3 known issues already there) + 1098 XCTest, 0 failures (4 LFM2 skips ^gcsvy3f). Only MLXFoundationModelsTests links MLXFoundationModels, so the Guided/CXGrammar/Macros bundles are unaffected; IntegrationTesting references none of the changed symbols. swift-format run.
    - next: /review
  timestamp: 2026-11-17T13:33:23.865015+00:00
position_column: doing
position_ordinal: '80'
title: 'MLXLanguageModel: KV-cache configuration option (kvCache:) applied to generation, warm-up and the prompt-cache key'
---
## Why
FoundationModelsRouter (Router card ^2fet0gs) asked for KV-cache quantization to be settable per model from the ACPAgent config (`profile.kvBits`). The chain is MLXLanguageModel (here) → Extras MLXModelLoader → Router LiveModelLoader → ACPAgent. Router maps `.bits8` / `.bits4` to `KVCacheConfiguration(strategy: .affine(try AffineKVCacheConfiguration(bits: N, groupSize: 64)))` (Libraries/MLXLMCommon/KVCacheConfiguration.swift). Write it TDD: failing tests first.

## What
- `MLXLanguageModel.init(..., kvCache: KVCacheConfiguration? = nil)` on the progress initializer (Libraries/MLXFoundationModels/MLXLanguageModel.swift ~513). The deprecated initializer (~485) may be removed; no backward compatibility is needed for it.
- Store it, and apply it as `parameters.kvCache` in `Executor.makeParameters` (~1033) and in the warm-up `GenerateParameters` (~620, including the speculative warm-up), so warm-up runs the quantized path.
- The executor prompt cache must never restore a full-precision cache under an 8-bit configuration, or the reverse. Put the configuration (or its resolved strategy identity) in the cache key / compatibility check.
- Real-model check: on Qwen3.8-27B (hybrid, 16 full-attention layers; mxfp4 if present locally, else the 4-bit build) affine 8-bit works with PromptCacheCheckpoint and multi-turn prefix reuse; also Flint oQ4e. Real-model runs take the scratchpad bench.lock.

## Acceptance
- [x] Unit tests (written first, failing first): the option reaches makeParameters and warm-up; a prompt cache built under one KV configuration is not reused under another; a same-configuration second turn reuses the prefix.
- [x] Real-model KV8 multi-turn reuse verified on Qwen3.8 and Flint oQ4e (prefix hit reported, coherent output).
- [x] All 5 test bundles green via xcrun xctest; zero warnings; swift-format run.
- [ ] Reply to FoundationModelsRouter with the stable SHA once pushed. #kv-config

## Review Findings (2026-11-17 07:00)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 20 file(s) reviewed, 4 not reviewed.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tags/01M7MSEBZQ22VAYERV66GVC6FS.jsonl` — no validator matches this file
> - `.kanban/tags/01M7MSEBZQ22VAYERV66GVC6FS.yaml` — no validator matches this file
> - `.kanban/tasks/01M7MN11N2WF3WAGFG84ZV2RDX.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7MN11N2WF3WAGFG84ZV2RDX.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:31` `code-hygiene/dead-code-swift` — var.instance `kvCache` is assignOnlyProperty.
- [x] `Libraries/MLXFoundationModels/ExecutorPromptCache.swift:2208` `reuse/reuse` — ExecutorGenerationCacheReceiver copies the shape of the existing ExecutorDrafterStateReceiver: a Mutex-held optional value, a receive callback that stores it, and a read accessor. Two near-identical receiver types now have to be maintained, and a fix to one will not reach the other. Generalize ExecutorDrafterStateReceiver into a generic receiver (for example `ExecutorHandOffReceiver<Value>` with a consuming `take()`), then specialize it for the drafter continuation and the generation caches, or add a consuming read to the existing class. If the two cannot share a type, at least keep the consuming read as a method on the shared shape so the locking logic lives in one place.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:857` `duplication/duplication` — The receiver wiring for the generation cache is copy-pasted across three pass functions. Each one declares `let finalCache = ExecutorGenerationCacheReceiver()`, passes `generationCache: finalCache.receive` into the generation call, and passes `generationCache: finalCache` into `withPromptCacheCommit`, next to the matching `finalDrafterState` lines. A fix to one copy, such as changing the hand-off order or the commit argument, can drift from the others. Bundle the two receivers (drafter state and generation cache) into one small value type that owns both, with a single method that builds the `withPromptCacheCommit` arguments. Each pass then creates one receiver object and passes it through, so the wiring lives in one place.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1230` `duplication/duplication` — The `finalCache` receiver declaration added here is the same line as the ones in the other two pass functions, so the three copies have to stay in sync by hand. Replace the three hand-written receiver blocks with the shared receiver type proposed for line 857, so this pass only creates one receiver object.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1385` `duplication/duplication` — The `finalCache` receiver declaration added here repeats the block in the other two pass functions, and the copies can drift apart. Use the shared receiver type from the fix for line 857 so the three passes stop repeating this wiring.
- [x] `Libraries/MLXLMCommon/Evaluate.swift:581` `duplication/duplication` — The hand-off parameter pair is declared again here, with the same defaults as in `generateProtocolTokensTask`. Take the shared hand-off type proposed for GenerateTokens.swift:318 in this signature as well.
- [x] `Libraries/MLXLMCommon/Evaluate.swift:594` `duplication/duplication` — This collector construction copies the one in `generateProtocolTokensTask` in GenerateTokens.swift, so the two entry points can diverge. Use the shared factory proposed for GenerateTokens.swift:331 so both entry points call one construction.
- [x] `Libraries/MLXLMCommon/GenerateTokens.swift:331` `duplication/duplication` — The `RecordingGeneratedTokenList(drafterContinuation:generationCache:)` construction is copied verbatim from `generateTaskRecordingTokens` in Evaluate.swift, so two entry points build the same collector the same way. Add one factory, for example `RecordingGeneratedTokenList.forHandOff(drafterContinuation:generationCache:)`, or make the collector's default parameters do the work. Both entry points then call the same code.
- [x] `Tests/MLXLMTests/GenerationCacheHandOffTests.swift:13` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.

## Review Findings (2026-11-17 07:22)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 8 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7MN11N2WF3WAGFG84ZV2RDX.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7MN11N2WF3WAGFG84ZV2RDX.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:850` `duplication/duplication` — The hand-off plumbing is copied verbatim into all three generation pass functions: `let handOff = ExecutorGenerationHandOff()`, passing `handOff.generationHandOff` to the generator, and passing `handOff: handOff` to `withPromptCacheCommit`. This change added the same lines to each copy, so the three copies can drift apart. Extract one helper that owns the prefill recorder, the hand-off, `startTrackedGeneration`, and `withPromptCacheCommit`, and take the generator as a closure that receives `handOff.generationHandOff` and `prefill.record`. Each pass then supplies only its generator call and its response loop. Keep the three call sites as the only per-pass differences.
- [x] `Libraries/MLXFoundationModels/MLXLanguageModelGeneration.swift:1372` `duplication/duplication` — Same added hand-off scaffolding in the reasoning pass, the third copy of the block. Fold into the shared helper proposed for line 850, so this copy becomes a call with its own generator closure.
