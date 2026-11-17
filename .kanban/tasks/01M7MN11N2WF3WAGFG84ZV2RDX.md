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