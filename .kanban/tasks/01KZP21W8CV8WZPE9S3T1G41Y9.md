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
