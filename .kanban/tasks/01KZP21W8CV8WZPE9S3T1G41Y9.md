---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzq9c2r6b8mwye6sxg23x6z9
  text: |-
    Research done. A tree-wide grep for the three old names shows only two files hold the DeepSeek-V4 uses:

    - `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` — property, `CodingKeys` case, `init` line, and `Default` line for each of the three.
    - `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` — `ExpectedValues` property, `assertMatches` comparison, `flashDefaults` row, and `distinctValues` row for each of the three.

    `DeepseekV4MathHelpers.swift` and `DeepseekV4QuantizationPlanTests.swift` hold none of the three names, thus they need no change.

    Important: the same two names `nSharedExperts` and `scoringFunc` also live in NemotronH, MiniMax, MiniMaxM3, DeepseekV3, MiMoV2Flash, GLM4MOE and GLM4MOELite. Those are other model types and stay as they are. The edits are limited to the two DeepSeek-V4 files.
  timestamp: 2026-08-11T02:10:38.470770+00:00
- actor: claude-code
  id: 01kzqa534rgybaedc5p6dta538
  text: |-
    Implementation landed. The diff is 24 lines in 2 files, and every line is a pure rename: 12 lines in `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` (property, `CodingKeys` case, `init` line and `Default` line for each of the three) and 12 lines in `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` (`ExpectedValues` property, `assertMatches` comparison, `flashDefaults` row and `distinctValues` row for each of the three). `git diff -U0` shows no key string and no default value changed.

    `swift-format` ran on the two touched files only. It made no change.

    Mutation proof, run after the rename:

    1. A misspelled key string. `case indexTopK = "index_topk"` became `"index_topq"`. Result: `testDistinctJSONDecodesEveryKey` and `testDistinctJSONSurvivesEncodeAndDecode` failed at line 119 with `XCTAssertEqual failed: ("512") is not equal to ("34")`. The default came through in place of the value the file gives, which is the failure the test is built to give.
    2. A corrupted default. `static let indexTopK = 512` became `513`. Result: `testMinimalJSONDecodesToFlashDefaults` failed at line 119 with `XCTAssertEqual failed: ("513") is not equal to ("512")`.

    Both mutations were reverted. `git status` then showed only the two intended files.

    Discovery: `Gemma4ChunkedPrefillTests.chunkSizeInvariance(chunkSize:)` failed one time in a full `swift test`, at `chunkSize` 16. It passed on the next full run and on three runs of the filter. The cause is in that test: `makeTinyModel()` gives the model random weights with no seed, thus the `allClose(rtol: 1e-4, atol: 1e-5)` comparison is not stable. The DeepSeek-V4 rename cannot reach Gemma4. Recorded as new task ^ztca9vv.
  timestamp: 2026-08-11T02:24:18.072134+00:00
- actor: claude-code
  id: 01kzqa59mzc57fr1qzpcctq181
  text: |-
    ### implement — changed
    - evidence: 2 files, 24 lines, all renames — Libraries/MLXLLM/Models/DeepseekV4Configuration.swift, Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift. `swift build` clean. `swift test --filter DeepseekV4` = 25 tests in 2 suites passed, plus 8 XCTest cases in DeepseekV4ConfigurationTests passed. Full `swift test` passed: 383 + 80 + 282 + 7 tests, 0 failures. Mutation proof held for both a misspelled key string and a corrupted default; both reverted.
    - next: /review
  timestamp: 2026-08-11T02:24:24.735707+00:00
- actor: claude-code
  id: 01kzqarpce6gdps5ggsxmvc74y
  text: |
    ### review — findings

    - scope: `review sha 7aee90b~1..7aee90b` (round 1). Commit `2840caa` is a kanban chore and was not read.
    - engine: 9 files attempted, 0 failed, 0 skipped. 1 finding, 1 confirmed, 0 refuted.
    - open finding: `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:78` — `normTopkProb` Boolean naming.

    Verification of the rename itself, in the order the reviewer was asked:

    1. JSON contract held. `git diff 7aee90b~1 7aee90b` gives 24 insertions and 24
       deletions across 2 files. Each changed line is an identifier only. The three
       `CodingKeys` raw strings are byte-identical on both sides:
       `case numSharedExperts = "n_shared_experts"` (line 155),
       `case scoringFunction = "scoring_func"` (line 159),
       `case indexTopK = "index_topk"` (line 173). No `Default` value changed:
       `1`, `"sqrtsoftplus"`, `512` are the same before and after. The test
       `ExpectedValues` numbers `23`, `"distinct-scoring-func"`, `34` are the same.
       All 37 keys and 36 defaults still agree with the earlier verification.
    2. The rename is complete. `git grep -w indexTopk` gives zero hits in the tree.
       No `nSharedExperts` or `scoringFunc` remains on `DeepseekV4Configuration`.
       The same-named properties on NemotronH, MiniMax, MiniMaxM3, DeepseekV3,
       MiMoV2Flash, GLM4MOE and GLM4MOELite are all still present and were not
       touched: `git diff --name-only 7aee90b~1 7aee90b` lists only
       `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` and
       `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift`.
    3. The mutation property survived. Two mutations were made and put back:
       - A wrong key string `"n_shared_expertz"` failed
         `testDistinctJSONDecodesEveryKey` and
         `testDistinctJSONSurvivesEncodeAndDecode` (`"1"` is not equal to `"23"`).
       - A wrong default `indexTopK = 511` failed
         `testMinimalJSONDecodesToFlashDefaults` (`"511"` is not equal to `"512"`).
       After the revert, `git status` and `git diff 7aee90b` are both empty.
    4. Suite green. Full `swift test`: 383, 80, 282 and 7 tests in four runs, plus 8
       XCTest cases. Zero failures, zero errors, zero warnings. No unrelated file is
       in the commit.

    The `Gemma4ChunkedPrefillTests.chunkSizeInvariance(chunkSize:)` flake is not
    related to this card. Gemma4 holds no reference to `DeepseekV4Configuration`.
    It stays on ^ztca9vv. It did not appear in this suite run.

    - next: rename `normTopkProb`, keep the `"norm_topk_prob"` key string, then run
      `/review` again.
  timestamp: 2026-08-11T02:35:00.366777+00:00
- actor: claude-code
  id: 01kzqasyee3sd8qk4r3e4a49gt
  text: |
    ### finish iteration 1 — findings
    - implement: changed — DeepseekV4Configuration.swift and DeepseekV4ConfigurationTests.swift, 24 lines, each one a rename
    - test: green — `swift build` clean; `--filter DeepseekV4` 25 tests + 8 XCTest cases; full `swift test` 383/80/282/7, 0 failures
    - commit: 7aee90b refactor(mlx-lm): rename DeepseekV4Configuration properties to match Swift style; 2840caa (kanban chore)
    - review: findings — DeepseekV4Configuration.swift:78 (1 finding). 9 files attempted.
    - VERIFIED: the JSON contract did not change. The reviewer read each of the 24 changed lines. The three key strings `"n_shared_experts"`, `"scoring_func"` and `"index_topk"` are the same on both sides of the diff, and no `Default` value changed. `git grep -w indexTopk` gives zero hits. The same property names on NemotronH, MiniMax, MiniMaxM3, DeepseekV3, MiMoV2Flash, GLM4MOE and GLM4MOELite were not touched, because the diff lists only two files. The mutation property survived: a misspelled key string and a corrupted default each fail the suite.
    - FINDING: `normTopkProb` does not read as an assertion about the receiver. The reviewer says this is not a conflict, because the Swift name can change while the `CodingKeys` string `"norm_topk_prob"` stays, the same way the three renames of this card work.
    - note: the `Gemma4ChunkedPrefillTests` flake is not related. Gemma4 reads no property of `DeepseekV4Configuration`. It stays on ^ztca9vv.
    - next: implement — rename `normTopkProb`
  timestamp: 2026-08-11T02:35:41.390423+00:00
- actor: claude-code
  id: 01kzqb4dqc9mzb2darwdbqyrj6
  text: |-
    Round 2, the single open finding. `normTopkProb` on `DeepseekV4Configuration` is now `normalizeTopkProb`. The reviewer gave two names; the card takes the first, because it matches the imperative-verb form of `useAttnSink` in the same file, thus the file stays consistent with itself.

    The diff is 8 lines in 2 files, and each line is a pure rename:

    - `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift` — the property, the `CodingKeys` case, the `init` line and the `Default` line.
    - `Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift` — the `ExpectedValues` property, the `assertMatches` comparison, the `flashDefaults` row and the `distinctValues` row.

    JSON contract held. `git diff -U0` shows `case normalizeTopkProb = "norm_topk_prob"`. The raw string is byte-identical on both sides. No `Default` value changed: `static let normalizeTopkProb = true` keeps `true`. The test rows keep `true` and `false`. All 37 key strings and 36 defaults are still those the earlier review verified, because no other line of the file moved.

    `git grep normTopkProb` on the two DeepSeek-V4 files gives zero hits. The same-named properties on DeepseekV3, GLM4MOE, MiniMaxM3, NemotronH, Qwen3MoE, Qwen3Next, Qwen35 and LFM2 are untouched, because `git status` lists only the two intended files.

    `swift format --in-place` ran on the two touched files only. It made no change.

    Mutation proof, run after the rename:

    1. A misspelled key string. `case normalizeTopkProb = "norm_topk_prob"` became `"norm_topk_probz"`. Result: `testDistinctJSONDecodesEveryKey` and `testDistinctJSONSurvivesEncodeAndDecode` failed at line 106 with `XCTAssertEqual failed: ("true") is not equal to ("false")`. The default came through in place of the value the file gives.
    2. A corrupted default. `static let normalizeTopkProb = true` became `false`. Result: `testMinimalJSONDecodesToFlashDefaults` failed at line 106 with `XCTAssertEqual failed: ("false") is not equal to ("true")`.

    Both mutations were put back. `git diff` then holds only the 8 intended rename lines, and `git status` lists only the two intended Swift files.

    The `Gemma4ChunkedPrefillTests.chunkSizeInvariance` flake on ^ztca9vv did not appear in this full run.
  timestamp: 2026-08-11T02:41:24.716182+00:00
- actor: claude-code
  id: 01kzqb4krpab3fr4rxr074ss3q
  text: |-
    ### implement — changed
    - evidence: 2 files, 8 lines, each one a rename — Libraries/MLXLLM/Models/DeepseekV4Configuration.swift, Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift. Finding at DeepseekV4Configuration.swift:78 closed and flipped to `- [x]`. `swift build` clean. `swift test --filter DeepseekV4` = 25 tests in 2 suites passed, plus 8 XCTest cases with 0 failures. Full `swift test` passed: 383 + 80 + 282 + 7 tests in four runs, 0 failures, 0 errors. Mutation proof held for a misspelled key string (2 tests failed) and for a corrupted default (1 test failed); both were put back and the tree is clean apart from the two intended files.
    - next: /review
  timestamp: 2026-08-11T02:41:30.902213+00:00
position_column: doing
position_ordinal: '80'
title: Rename three DeepseekV4Configuration properties to match Swift naming
---
## What

The review engine found three naming defects in
`Libraries/MLXLLM/Models/DeepseekV4Configuration.swift`. The findings came from
a diagnostic probe during the review of task ^6f6qmf7. The probe read a file
that was not in the commit under review, thus the findings stayed open when
^6f6qmf7 moved to `done`.

Make these three renames:

- `DeepseekV4Configuration.swift:66` — rename `nSharedExperts` to
  `numSharedExperts`. The other properties in the struct use the `num` prefix.
- `DeepseekV4Configuration.swift:76` — rename `scoringFunc` to
  `scoringFunction`. `func` is an abbreviation of an ordinary word.
- `DeepseekV4Configuration.swift:119` — rename `indexTopk` to `indexTopK`.

## Important

Change the Swift property names only. Do NOT change the JSON `CodingKeys`
string values. The strings must stay the same, because they must agree with the
published config files.

The renames also touch the `ExpectedValues` table and the `Default` enum in
`Tests/MLXLMTests/DeepseekV4ConfigurationTests.swift`.

A review verified that all 37 decoded keys and all 36 default values agree with
the reference. Do not change a key or a default value.

## Acceptance Criteria

- [x] The three properties have their new names.
- [x] The `CodingKeys` string values did not change.
- [x] No default value changed.
- [x] Each use of the old names is updated.

## Tests

- [x] `swift build`
- [x] `swift test --filter DeepseekV4ConfigurationTests`
- [x] Full `swift test` stays green.
- [x] The mutation proof still holds: a corrupted default must fail the suite.

#deepseek-v4

## Review Findings (2026-08-10 21:26)

- [x] `Libraries/MLXLLM/Models/DeepseekV4Configuration.swift:78` — Boolean property does not read as an assertion about the receiver; should follow the pattern of properties like `isEmpty`, `isEnabled`, or `useAttnSink`. Rename to express the condition as an assertion, such as `normalizeTopkProb` (using imperative verb style like `useAttnSink` on line 124) or `topkProbIsNormalized`.

### Note for the implementer

The `CodingKeys` raw string `"norm_topk_prob"` must not change. Rename the Swift
property only, in the same manner as the three renames above.
