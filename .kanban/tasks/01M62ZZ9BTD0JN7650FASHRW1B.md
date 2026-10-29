---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m63k2gqsdpfrgs47qwkdqw2r
  text: |-
    Research + implementation notes:
    - One function, `GuidedGenerationLoop.modelFeed(sampled:fastForward:)`, feeds both paths (GuidedDecoder.emit for plain + target decisions, GuidedDraft.feed for MTP drafts). Changed it to `[sampled] + fastForward`.
    - MTP audit with the new feed (no code change needed): `decode` routes a 1-token feed to `speculate` and a multi-token feed to `feedForcedTokens`; a decision that forces tokens now always has feed.count >= 2, so it goes through `feedForcedTokens` (before, a decision that forced exactly ONE token was speculated after directly with that forced token as the bonus — that cheap path is gone, one extra verify-shaped forward per such decision). `verifyBatch` = [bonus, d1..d(n-1)] + last.feed = [.., dn, ff..]; acceptance reads rows 0..n-1 (bonus..d(n-1)), unchanged. `keptPositions`/`keepVerified` and `commitDrafter` pair hidden row i with fed[i+1] and the last hidden row with next.first; with feed[0] now the decided token, the drafter's pending token (previous next.first) is that decided token, so the pairing holds in `speculate`, the `!feedsLastDraft` branch (next = [dn]+ff) and `feedForcedTokens`. Depth: `verifyBatch` checks `verifiesInPlace(withLast.count)` (one longer now), `feedForcedTokens` chunks by `forcedChunkLength` (mostVerifiedTokens or drafts+1, both verified in place), so the in-place verify depth still bounds every batch.
    - Tests: Tests/MLXGuidedGenerationTests/GuidedModelFeedTests.swift (unit, modelFeed) and Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift (loop on the Qwen35 fixture through a recording LanguageModel wrapper; asserts the fed tokens after the prompt are a prefix of the emitted tokens, missing at most the terminating decision). RED showed the missing `"` before every forced key, e.g. fed `[0, 27,16,27,19,12, ...]` vs emitted `[0, 4, 27,16,27,19,12, ...]`. `GuidedFixtureRun.generate` got an optional `target:` override.
    - Running one Swift Testing suite: `swift test` and `xcrun xctest -XCTest` cannot filter Swift Testing; copying `swiftpm-testing-helper` into `.build/out/Products/Debug/` (so MLX finds the metallib colocated) and running it with DYLD_FRAMEWORK_PATH/DYLD_LIBRARY_PATH = Xcode MacOSX platform Developer Library/Frameworks + usr/lib and `--testing-library swift-testing --filter <Suite>` works.
    - End-of-generation note (not changed): MTP `speculate` returns nil before `keepVerified` when the generation stops inside a round, so the guided loop's cache timeline lags the physical cache at the end; harmless because the guided loop discards its cache, but worth knowing if it ever keeps one.
  timestamp: 2026-10-29T10:25:15.001009+00:00
- actor: claude-code
  id: 01m63khjyz42ew9kv0dd9a6x0h
  text: |-
    Implementation landed (uncommitted, task left in doing).
    - Fix: `modelFeed` returns `[sampled] + fastForward`; docs updated in GuidedGenerationLoop.swift and GuidedSpeculation.swift. No MLXLMCommon change, so the text MTP guards are unaffected.
    - TDD: RED — GuidedModelFeedTests (got [6,7,8], want [5,6,7,8]) and GuidedGenerationModelFeedTests (7 issues over 4 schemas: fed tokens skip the decided `"` before each forced key). GREEN after the fix. GuidedGenerationMTPTests (plain == MTP on the fixture, 8 tests incl. rejecting drafter and block sizes 2/4) pass unchanged.
    - Full suite: `swift build --build-tests` 0 compiler warnings (only the pre-existing build-system note "missing creator for mutated node ... mlx-swift_Cmlx.bundle"); xctest bundles — CXGrammarTests 7 passed, MLXGuidedGenerationTests 72 passed, MLXHuggingFaceMacrosTests 6 passed, MLXFoundationModelsTests 415 passed, MLXLMTests 920 XCTest + 2011 Swift Testing passed; 0 failures, 0 skipped.
    - Release benchmark `GuidedGenerationBenchmarkTests/qwen38GuidedJSONWithMTP()` (load ~3): before plain 54.05 / MTP 74.71 tok/s, 64 tokens, accepted 27/32; after plain 52.27 / MTP 67.59 (rerun 52.21 / 67.72), 58 tokens, accepted 31/40, divergence=none. The output changed by design (the cache now holds the decided token). MTP stays 1.29-1.30x plain (was 1.38x). The drop comes from the extra fed token per fast-forwarding decision, and on MTP also from losing the old shortcut where a decision that forced exactly one token was speculated after directly. Under the fix such a decision costs a 2-token verify forward plus a target decision before the next round. I checked "speculate after the last forced token" as an alternative: it gives fewer tokens per forward, not more. Forced tokens as chain drafts would be a redesign, not this card.
  timestamp: 2026-10-29T10:33:28.799400+00:00
- actor: claude-code
  id: 01m63khnmbzx53d1d8j2jjr9zc
  text: |-
    ### implement — changed
    - evidence: 5 files — Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift, Libraries/MLXGuidedGeneration/GuidedSpeculation.swift, Tests/MLXFoundationModelsTests/GuidedGenerationMTPTestSupport.swift, Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift (new), Tests/MLXGuidedGenerationTests/GuidedModelFeedTests.swift (new); swift build --build-tests 0 warnings; 5 xctest bundles 0 failures 0 skipped; release guided JSON MTP bench plain 52.27 / MTP 67.59 tok/s, divergence=none (was 54.05 / 74.71)
    - next: /review
  timestamp: 2026-10-29T10:33:31.531962+00:00
position_column: doing
position_ordinal: '8180'
title: 'Guided generation: feed the sampled token to the model when its commit fast-forwards'
---
## What
`GuidedGenerationLoop` (plain decode, since the upstream import f1573a93) feeds the model only the jump-forward tokens when a commit fast-forwards: the token the model decided right before them is emitted to the output and committed to the grammar, but never fed, so the KV cache holds `[.., ff1..ffk]` without it. Every later logit is conditioned on a sequence that skips one output token per fast-forward (JSON keys, enum completions...).

The MTP guided path (^9xe94tr) reproduces the same feed through one shared function, `GuidedGenerationLoop.modelFeed(sampled:fastForward:)` (Libraries/MLXGuidedGeneration/GuidedGenerationLoop.swift), so that its output stays byte-identical to plain decode. Fixing the feed there (`[sampled] + fastForward`) fixes both paths at once.

- [x] Write a failing test: after a fast-forwarding commit, the model cache holds the decided token before the forced tokens (cache length = prompt + every emitted token).
- [x] Change `modelFeed` to feed the decided token and then the forced tokens.
- [x] MTP: the verify batch then holds the draft and its forced tokens; check `GuidedSpeculation.verifyBatch` and the drafter commit still pair each fed token with the hidden row before it, and that the in-place verify depth still bounds the batch.
- [x] Re-run the guided benchmarks (output quality changes: guided outputs differ from before by design).

## Acceptance Criteria
- [x] Plain and MTP guided outputs stay byte-identical to each other on the fixture.
- [x] The cache holds every emitted token.

#guided-generation #bug