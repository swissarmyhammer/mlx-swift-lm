---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m63avc5pxxdh5crgff916tg0
  text: 'Research: `RotatingKVCache.makeMask` n > 1 builds `createCausalMask(n, offset: min(maxSize-1, live), windowSize: maxSize)` over a presentation laid out as [kept prefix | contiguous newest rows | new rows] (both `updateConcat` and the rewind-reserve `retainedView` path keep that layout). Query r >= 1 past the window therefore drops prefix columns first, while a one-row write drops the oldest non-prefix row. Keep-aware rule: column j visible to query column c iff j <= c and (j < keep or j > c - (maxSize - keep)). Needs an array mask under the same condition as today (cappedOffset + n > window). `RotatingStagedKVCache.makeMask` delegates to the live ring, so the staged (attention-only) path inherits the fix too. The drafter cache (`ShiftedMTPDrafter.makeCache`) is a ring with the same kept prefix + rewind reserve, so it inherits decode-consistent attention. `drafts(_:fittingWindowOf:)` is also used by `GuidedTarget.verifiesInPlace(tokens:)` in MLXGuidedGeneration. Existing `Qwen3VLContinuationTests.testWindowedPrefillBeyondCacheCapacityMatchesDecode` already asserts windowed prefill == decode for keep 0; Python mlx-lm uses the plain window for prefill.'
  timestamp: 2026-10-29T08:01:32.342626+00:00
- actor: claude-code
  id: 01m63be517k2t90t6fq3ktpjck
  text: 'Implementation landed (uncommitted). Decision on subtask 2: the keep-aware mask applies to EVERY multi-row write of a ring that keeps a prefix under its own window (`keep > 0 && windowSize ?? maxSize == maxSize`), chunked prefill included, not only the verify. Reasons: the cache cannot tell a verify from a prefill; one rule makes a stream''s tokens independent of how it batches them (prefill chunk size, verify vs decode), which is the same invariant `Qwen3VLContinuationTests.testWindowedPrefillBeyondCacheCapacityMatchesDecode` already pins for keep 0; plain and MTP prefill stay identical so prompt-cache reuse is unaffected. This diverges from Python mlx-lm (plain window over the presentation) for maxKVSize prompts longer than the window; documented in skills/mlx-swift-lm/references/kv-cache.md. Removed `RotatingKVCache.rowsAttendingAsSingleRowWrites`, `MTPSpeculativeTokenIterator.drafts(_:fittingWindowOf:)`, `keptPrefixWindowReason`, and the window check in `GuidedTarget.verifiesInPlace(tokens:)`. `RotatingStagedKVCache.makeMask` delegates to the live ring, so the staged (attention-only) path inherits the fix. TDD: new Tests/MLXLMTests/RotatingKVCachePrefixMaskTests.swift (4 scenarios incl. rewind reserve) failed on values before, passes after; `prefixKeepingRingSpeculatesPastTheWindow` failed on passthroughReason before, passes after. Full unit run: MLXLMTests 2001 swift-testing + 920 XCTest, MLXFoundationModelsTests 404, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6 — 0 failures, 0 skipped. Build: 0 compiler warnings (only SwiftPM''s pre-existing "missing creator for mutated node" note for the Cmlx bundle).'
  timestamp: 2026-10-29T08:11:47.623514+00:00
- actor: claude-code
  id: 01m63cdxsb6fhen1d2sb6gdckr
  text: 'Release guards (xcodebuild Release, -destination ''platform=macOS,arch=arm64'' ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES, load ~3-3.5): qwen38FourBitMTPGeneration pass, default blockSize 5 pMin 0.75 math/code/prose 100.4/87.6/83.6 tok/s (baseline 100.4/87.4/83.4), plain 59.5; qwen38FourBitMTPSampledGeneration pass; qwen38FourBitMTPSystemPromptCache pass, edit turn prefilled 123 tokens, no known issues, all turn-0/turn-1 tokens exact or bf16 near-tie.'
  timestamp: 2026-10-29T08:29:08.779658+00:00
- actor: claude-code
  id: 01m63ce1cfav4852wwp0k5jcn2
  text: |-
    ### implement — changed
    - evidence: 7 files — Libraries/MLXLMCommon/RotatingKVCache.swift, Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift, Libraries/MLXGuidedGeneration/GuidedTarget.swift, skills/mlx-swift-lm/references/kv-cache.md, Tests/MLXLMTests/Qwen35MTPCacheKindTests.swift, Tests/MLXLMTests/RotatingKVCachePrefixMaskTests.swift (new, 4.8 KB); `swift build --build-tests` 0 compiler warnings; xctest 5 bundles 0 failures 0 skipped (MLXLMTests 2001+920, MLXFoundationModelsTests 404, MLXGuidedGenerationTests 70, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6); release guards pass (MTP 100.4/87.6/83.6 vs plain 59.5, edit turn 123 tokens).
    - next: /review
  timestamp: 2026-10-29T08:29:12.463721+00:00
position_column: doing
position_ordinal: '8180'
title: 'MTP: keep speculating past the window of a maxKVSize ring that keeps a prefix'
---
## What
Found in ^crdd8sf. A `RotatingKVCache` built from legacy `maxKVSize` keeps a prefix (`keep = min(4, maxKVSize - 1)`). A multi-row write (an MTP verify) masks a plain window over the rows it presents (`RotatingKVCache.makeMask` n > 1), so a query past the window loses the kept prefix, while single-token decode keeps the prefix in the ring. MTP verify tokens then differ from plain decode. ^crdd8sf made the Qwen in-place round shrink to the window and then pass through with `MTPSpeculativeTokenIterator.keptPrefixWindowReason`, so `maxKVSize` streams stop speculating once the window fills (typed capacity with `preservedPrefixTokens: 0` keeps speculating).

- [x] Give the multi-row write of a ring that keeps a prefix a keep-aware mask: query row r sees rows `< keep` and the `maxSize - keep` rows up to r (as single-row decode does), at least for a speculative verify.
- [x] Decide whether chunked prefill past the window should use the same mask (plain prefill semantics change) or only the verify. Decision: every multi-row write under the ring's own window uses it, prefill included (see comments).
- [x] Remove the passthrough in `MTPSpeculativeTokenIterator.drafts(_:fittingWindowOf:)` once the verify attends as decode.

## Acceptance Criteria
- [x] With `maxKVSize` smaller than the generated length, Qwen MTP greedy output equals plain greedy output and `passthroughReason == nil`.

## Tests
- [x] Extend Tests/MLXLMTests/Qwen35MTPCacheKindTests.swift (`prefixKeepingRingStopsSpeculatingAtTheWindow` becomes a speculate-past-the-window test).
- [x] Release guards: `qwen38FourBitMTPGeneration` and `qwen38FourBitMTPSystemPromptCache` unchanged.

#mtp #kv-cache #qwen38