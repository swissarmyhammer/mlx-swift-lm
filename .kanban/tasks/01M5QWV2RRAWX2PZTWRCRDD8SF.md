---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5smzxkzg6vv47mbarf1fz2c
  text: 'Change after the PR 581 port: a cache that cannot stage a verify block now throws `MTPInitializationError.unsupportedSpeculativeCache` before prefill, and ChatSession falls back with `SpeculativeDecodingFallbackReason.unsupportedCache`. Use this path for unsupported rotating/quantized combinations.'
  timestamp: 2026-10-25T13:46:22.719346+00:00
- actor: claude-code
  id: 01m62jqhbaath3ff25ya2rmdpy
  text: |-
    Research + implementation notes (implement step):
    - Card line refs were stale; `MTPDrafterCacheState` does not exist — the serialized form is `MTPDrafterContinuation.arrays()/metadata()` (MTPDrafterContinuationSerialization.swift). Its prompt-cache layout ALREADY records each drafter cache's class + metaState (window, keep, reserve); `restorePromptCacheSnapshot(into:)` refuses another class/window. The gap was that the restore always built `makeState(parameters: nil)` caches, so a rotating continuation could never restore. Added `parameters:` to `MTPDrafterContinuation.init(arrays:metadata:drafter:parameters:)` / `restored(by:parameters:)` and ChatSession passes its generateParameters.
    - Drafter cache: `ShiftedMTPDrafter.makeCache(layerCount:parameters:maximumBlockSize:)` (QwenMTP.swift) builds `RotatingKVCache` of the main capacity (same window + kept prefix) with rewind reserve `2 * GatedDeltaDecodeKernel.mostTokens` (= max finalize rewind: proposal + discarded lookahead + boundary), else `KVCacheSimple` (quantized main => bf16 drafter, doc comment on Qwen35MTPPredictor.newCache). Both LLM + VLM predictors use it.
    - FOUND A TRAP in main path: with maxKVSize and blockSize 6, `rewindSpeculativeEntries` classified a wrapped RotatingKVCache (isTrimmable == false) as recurrent -> returned 0 -> iterator precondition "advertised native rewind ... rewound 0 of 5". Fix: classify by `MambaCache`, check attention entries with counted `maxTrimCount` (a multi-row write keeps window-1+S rows so S-1 drafts always rewind exactly); new `canRewindSpeculativeVerify` gate replaces the 3 copies of `isTrimmable || Mamba` (iterator helper `verifiesInPlace`). Non-rotating caches keep the old `isTrimmable` gate (VarianceNormalized `isTrimmable(after:)` would otherwise turn rounds off).
    - FOUND: a multi-token verify past the window of a ring that keeps a prefix (legacy `maxKVSize` => keep 4) masks a plain window and drops the kept prefix, while plain decode keeps it -> MTP tokens differed from plain. Ring with keep 0 (typed capacity) is exact and now speculates past the wrap. For keep > 0 the round is shortened to the window and then the stream passes through with `MTPSpeculativeTokenIterator.keptPrefixWindowReason`. A keep-aware multi-row mask would let maxKVSize streams keep speculating (follow-up task).
    - Quantized (kvBits 8, group 32; fixture gained a `headDimension` knob since head_dim 8 cannot quantize): MTP == plain, no passthrough, drafter stays KVCacheSimple.
  timestamp: 2026-10-29T01:00:00.746613+00:00
- actor: claude-code
  id: 01m62krafsm4tt9r582a9xfvsa
  text: |-
    ### implement — changed
    - evidence: `swift build --build-tests` clean (only the pre-existing "missing creator" build-system note, 0 compiler warnings); new suite `xcrun xctest -XCTest MLXLMTests.Qwen35MTPCacheKindTests` 9 tests / 21 cases pass (watched RED first: maxKVSize+blockSize 6 trapped in the iterator precondition, keep-prefix tokens != plain, rotating restore cold-started at 55); full bundles: MLXLMTests 1981 Swift Testing + 920 XCTest pass, MLXFoundationModelsTests 376, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5 — 0 failures. Release guards (xcodebuild, load ~6.5 with mediaanalysisd busy): `qwen38FourBitMTPGeneration` blockSize 2 MTP 92.0/82.7/80.2 tok/s (baseline 91/80/79), plain 59.5; `qwen38FourBitMTPSystemPromptCache` passes, no known issues, edit turn prefills 123 tokens with MTP at 89.4 tok/s; `qwen38FourBitMTPSampledGeneration` passes.
    - files: Libraries/MLXLMCommon/{QwenMTP,KVCache,RotatingKVCache,MTPSpeculativeTokenIterator,MTPDrafterContinuationSerialization,MTPDrafterContinuation,ChatSession}.swift, Libraries/MLXLLM/Models/Qwen35MTP.swift, Libraries/MLXVLM/Models/Qwen35MTP.swift, Tests/MLXLMTestSupport/{Qwen35FixtureJSON,Qwen35MTPFixture}.swift, Tests/MLXLMTests/Qwen35MTPCacheKindTests.swift (new, 12 KB), skills/mlx-swift-lm/references/kv-cache.md.
    - deviation noted: tests live in the new file Qwen35MTPCacheKindTests.swift instead of Qwen35MTPTests.swift (user constraint: no growing the 104 KB file). The drafter ring uses `Capacity.makeRotatingCache(rewindCapacity:)` (what `makeAttentionKVCache` calls) so it can carry its own rewind reserve.
    - follow-up filed: ^8f954cv (keep-aware multi-row mask so maxKVSize rings keep speculating past the window).
    - next: /review
  timestamp: 2026-10-29T01:17:55.065896+00:00
- actor: claude-code
  id: 01m62kzpef6whh69rfchdmtyn6
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (5724dbc3) — 0 findings, 0 confirmed, 1 refuted, 21 attempted, 0 failed; 12 files reviewed, 5 not matched by any validator (.kanban task files, skills/mlx-swift-lm/references/kv-cache.md)
    - next: task moved to done
  timestamp: 2026-10-29T01:21:56.687446+00:00
- actor: claude-code
  id: 01m62m01hekvrm421nnjmyjz6d
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 13 files (rotating rewind fix, drafter window, kept-prefix fallback, rotating continuation restore)
    - test: green — 5 bundles; Release MTP 92.0/82.7/80.2 vs plain 59.5; edit turn 123 tokens
    - commit: 5724dbc3
    - review: clean — task moved to done
  timestamp: 2026-10-29T01:22:08.046934+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5SJPFM01WFRYWMQXVGMA4MQ
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: done
position_ordinal: fff780
title: 'MTP: define drafter behaviour with maxKVSize (rotating) and quantized main KV caches'
---
## What
The Qwen MTP drafter cache is a plain `KVCacheSimple` (`Qwen35MTPPredictor.newCache`, Libraries/MLXLLM/Models/Qwen35MTP.swift ~43-45, ~113-115). It never rotates or quantizes. When the main full-attention caches use `maxKVSize` (rotating) or KV quantization (`kvBits`, `KVCachePlan`), the rule "drafter = main − 1" (by `processedTokenCount`) and the drafter memory use are not defined.

- [x] Rotating main cache: make the drafter cache rotate with the same window (`RotatingKVCache` from `makeAttentionKVCache(parameters:)`). The alignment checks compare logical token counts, not stored rows. `commitDrafterState` trims rejected entries (`trimPromptCache`, Qwen35MTP.swift ~231) and finalize trims `lookahead + 1`; after the ring wraps, `RotatingKVCache` can trim only a limited count (KVCache.swift ~1041-1054). Keep the keep-size large enough for `maximumBlockSize` entries, or change to passthrough with a reason when a trim is not possible.
- [x] Record the cache kind and window in `MTPDrafterCacheState.arrays()` / `metadata()`, and reject a restore into a different kind.
- [x] Quantized main cache: keep the drafter cache bf16 (it is one layer; about 2 × 4 heads × 256 × 2 B = 4 KB for each token, which is small). Record this rule in a doc comment in Qwen35MTP.swift.
- [x] If a combination is not supported, `MTPSpeculativeTokenIterator` must change to passthrough with a clear `passthroughReason`, and must not trap or produce wrong tokens.

## Acceptance Criteria
- [x] With `maxKVSize` smaller than the generated length, MTP greedy output equals plain greedy output with the same `maxKVSize`, the drafter cache stays bounded, and a reject after the ring wraps does not trap.
- [x] With `kvBits = 8`, MTP greedy output equals plain greedy output with the same KV setting, or the stream changes to passthrough with a reason. A test asserts which of the two happens.

## Tests
- [x] Tests/MLXLMTests/Qwen35MTPTests.swift on the shared fixture: rotating-cache equality, a reject after wrap, bounded drafter size, quantized-KV behaviour, and a restore of a rotating drafter state into a simple cache (rejected). (In the new file Tests/MLXLMTests/Qwen35MTPCacheKindTests.swift, per the user's rule not to grow Qwen35MTPTests.swift.)
- [x] `swift build --build-tests && xcrun xctest .build/out/Products/Debug/MLXLMTests.xctest` passes.

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #mtp #qwen38 #kv-cache