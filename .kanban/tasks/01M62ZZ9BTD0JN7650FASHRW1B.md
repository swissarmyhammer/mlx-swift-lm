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
- actor: claude-code
  id: 01m63kneyermw8fenszbnwn9tj
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (3c69b862) — 1 finding (1 confirmed, 0 refuted, 7 attempted) — Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift:39
    - next: implement the open finding (cover the MTP path in theTargetIsFedEveryEmittedToken), then re-review
  timestamp: 2026-10-29T10:35:35.758899+00:00
- actor: claude-code
  id: 01m63knwxmvt9m5tmx1xg56eh6
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 5 files (modelFeed feeds decided + forced tokens, feed tests)
    - test: green — 5 bundles 0 failures 0 skipped; Release guided JSON plain 52.27 / MTP 67.59, identical
    - commit: 3c69b862
    - review: findings — Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift:39
  timestamp: 2026-10-29T10:35:50.068716+00:00
- actor: claude-code
  id: 01m63makq7cmxv83ma1x9rap8n
  text: |-
    Finding (GuidedGenerationModelFeedTests.swift:39, MTP coverage) addressed, tests only.
    - `theTargetIsFedEveryEmittedToken` now takes `blockSize` from `[nil, 2, 4]` (plain, single draft, 3-draft chains, as GuidedGenerationMTPTests) x 4 schemas = 12 cases. Also asserts `speculationFallbackReason == nil` and `(draftedTokenCount > 0) == (blockSize != nil)`, so the MTP cases really speculate.
    - Why the wrapper had to go: a wrapping `LanguageModel` makes the MTP path fall back. `Qwen35MTPDraftModel.validateCompatibility` and its embedding/LM-head lookup only accept `Qwen35TextModel`/`Qwen35Model`. The recorder is now `CacheRecordingTarget: Qwen35TextModel`, a subclass under `@testable import MLXLLM` (the class is public, not final). It copies the fixture target's configuration and weights (`update(parameters:verify: .noUnusedKeys)`) and overrides `callAsFunction(_:cache:state:)`, the only forward GuidedTarget calls (prefill, feedOneAtATime, verify). The default LLMModel.prepare hands a 5-token prompt over whole, so the prompt goes through the same override.
    - What gets compared: the tokens the target's cache keeps, not the raw forward inputs. Before each forward the record is cut back to the full-attention cache's `offset`, which drops verify rows that `keepVerified`/`rewindSpeculativePromptCache` rewound. After the run it is cut to the offset of the last cache. Prefix and count invariants are the same as on the plain path. (`cachedTokens` is nil when no attention layer was seen, and `#require` then fails the test.)
    - Regression proof, both temporary and both restored (GuidedSpeculation.swift is not in the diff): (1) `feedForcedTokens` dropping the decided token made 8/8 MTP cases fail with 15 issues while 4/4 plain cases passed; (2) `verifyBatch` using `last.feed.dropFirst()` (the draft-chain feed) made 8/8 MTP cases fail with 16 issues while plain passed.
    - Running one Swift Testing suite: a copy of swiftpm-testing-helper sits in .build/out/Products/Debug (gitignored), run as in the earlier comment.
  timestamp: 2026-10-29T10:47:08.775590+00:00
- actor: claude-code
  id: 01m63marqbk823p4ck9y48r54t
  text: |-
    ### implement — changed
    - evidence: 1 file — Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift; swift build --build-tests 0 compiler warnings (only the pre-existing build-system note about mlx-swift_Cmlx.bundle); xcrun xctest: CXGrammarTests 7, MLXGuidedGenerationTests 72, MLXHuggingFaceMacrosTests 6, MLXFoundationModelsTests 415 (GuidedGenerationModelFeedTests 12 cases), MLXLMTests 920 XCTest + 2011 Swift Testing — all passed, 0 failures, 0 skipped; MTP feed regressions caught (15 and 16 issues), then restored
    - next: /review
  timestamp: 2026-10-29T10:47:13.899546+00:00
- actor: claude-code
  id: 01m63md1v9s15fdf0g9bnm4n5c
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (0da10dd0) — 0 findings, 0 confirmed, 0 refuted (7 attempted, 0 failed); prior item from Review Findings (2026-10-29 05:34) is checked
    - next: none — task moved to done
  timestamp: 2026-10-29T10:48:28.777320+00:00
- actor: claude-code
  id: 01m63mdgf78jkrbsy5sd9wnnx3
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 1 test file (MTP paths in theTargetIsFedEveryEmittedToken, CacheRecordingTarget)
    - test: green — 5 bundles 0 failures 0 skipped; MTP feed regressions caught
    - commit: 0da10dd0
    - review: clean — task moved to done
  timestamp: 2026-10-29T10:48:43.751831+00:00
position_column: done
position_ordinal: ffff8380
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

## Review Findings (2026-10-29 05:34)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 5 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M62ZZ9BTD0JN7650FASHRW1B.jsonl` — no validator matches this file
> - `.kanban/tasks/01M62ZZ9BTD0JN7650FASHRW1B.md` — no validator matches this file

- [x] `Tests/MLXFoundationModelsTests/GuidedGenerationModelFeedTests.swift:39` `completeness/invariant-propagation` — The feed-every-emitted-token test runs only the plain path (`blockSize: nil`). The change's stated purpose is that the MTP guided path also feeds the decided token plus its forced tokens, and that path has its own feed code (`GuidedSpeculation.feedForcedTokens`, plus draft chains that call `modelFeed` through `GuidedDraftChain`). Nothing in the diff asserts that the MTP path's target receives the same token sequence, so a regression confined to the MTP feed would pass this test. Parameterize `theTargetIsFedEveryEmittedToken` over the MTP path as well (a non-nil `blockSize` with the fixture drafter), and assert the same prefix and count invariants as the plain path. If the MTP path cannot run on the test target, state that in the test's doc comment and say which path the test covers.
