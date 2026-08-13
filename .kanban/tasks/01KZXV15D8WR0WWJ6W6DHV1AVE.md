---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzxz3hgjnwexjy4msyq0cvne
  text: |-
    Research done. What the two files hold now:

    `Libraries/MLXLLM/Models/DeepSeekV4.swift`
    - Three classes, none of them `final`: `DeepSeekV4DecoderLayer`, `DeepSeekV4ModelInner`, `DeepSeekV4Model`. A scan of every Swift file of the repository finds no subclass of any of the three.
    - The `@ModuleInfo` properties of `DeepSeekV4DecoderLayer` (`attention`, `ffn`, `attentionNorm`, `ffnNorm`, `attentionConnection`, `ffnConnection`) and of `DeepSeekV4ModelInner` (`embedTokens`, `hcHead`, `norm`) carry no doc comment. This is the same cause the `DeepSeekV4Attention.swift` finding names.
    - `projectionTensors` ALREADY carries a doc comment. The review line 449 points at the first line of that comment, thus the named line is correct only as an example of the cause, and the cause sits on the `@ModuleInfo` properties above.
    - `stackRoutedExperts` holds three for-loops, a guard and a fourth for-loop, which is depth 4.

    `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`
    - Two classes, neither `final`: `DeepSeekV4RoPE` and `DeepSeekV4Attention`. Nothing subclasses either.
    - Seven `@ModuleInfo` properties carry no doc comment: `wqA`, `wqB`, `wkv`, `woA`, `woB`, `qNorm`, `kvNorm`. The review names four; the cause covers all seven.
    - The four stored properties of `YarnScaling` carry no doc comment.
    - No function of this file nests deeper than three.

    The rest of the DeepSeek-V4 family already marks its classes `final` (`DeepSeekV4Compressor`, `DeepSeekV4Indexer`, `DeepSeekV4MLP`, `DeepSeekV4SwitchGLU`, `DeepSeekV4MoEGate`, `DeepSeekV4MoE`), thus `final` is the prevailing pattern.

    Test cover: `DeepSeekV4ModelTests.sanitizeStacksThePerExpertWeightsIntoTheSwitchLayer` covers the full-set path, and `sanitizeLeavesAnAlreadyConvertedCheckpointAlone` covers the no-key path. No test covers the guard the extraction moves: a PARTIAL set of per-expert tensors. A test for that comes first.
  timestamp: 2026-08-13T16:25:53.938677+00:00
- actor: claude-code
  id: 01kzxzn0004g7ht3582ejcjfcc
  text: |-
    Implementation landed. What changed, and why.

    1. `final` on every class of both files. `DeepSeekV4DecoderLayer`, `DeepSeekV4ModelInner`, `DeepSeekV4Model`, `DeepSeekV4RoPE` and `DeepSeekV4Attention` are now `final`. Nothing in the repository subclasses any of them, and the rest of the DeepSeek-V4 family already spells its classes this way.

    2. A documentation comment on every member that had none. `DeepSeekV4.swift`: the six `@ModuleInfo` properties of `DeepSeekV4DecoderLayer` and the three of `DeepSeekV4ModelInner`. `DeepSeekV4Attention.swift`: the seven `@ModuleInfo` properties `wqA`, `wqB`, `wkv`, `woA`, `woB`, `qNorm` and `kvNorm`, and the four stored properties of `YarnScaling`. A script that reads each declaration of both files and looks at the line above it now finds no member without a comment.

    3. `stackRoutedExperts` is three levels deep. The guard, the stack, the write and the removal loop all moved into a new `stackPerExpertWeights(at:into:in:)`. The card suggested a helper that takes the guard and the inner loop alone, but the finding counts the guard as a level, thus a helper that left the guard behind would still read as level 4. The new helper leaves three plain for-loops with one call inside, and the removal loop sits at level 1 of the helper.

    Interesting discovery, and the reason for the new test: the guard that the extraction moved had NO test. `sanitizeStacksThePerExpertWeightsIntoTheSwitchLayer` covers a full set of per-expert tensors, and `sanitizeLeavesAnAlreadyConvertedCheckpointAlone` covers an empty set, but no test covered a SHORT set. A new test, `sanitizeLeavesAPartialSetOfPerExpertWeightsAlone`, covers it. The test was proved first: with the guard changed to `!stack.isEmpty` the test failed with two issues, and it passes with the correct guard. A lost guard would stack the wrong number of rows into `switch_mlp` and give a silent wrong answer, thus this test is worth its place.

    swiftlint: every finding it gives on the three changed files is present at `HEAD` on the same construct -- `file_length`, `trailing_comma`, `identifier_name` for `x`, and `type_body_length`. The repository holds no `.swiftlint.yml`, thus swiftlint runs on its defaults, and its `trailing_comma` rule disagrees with what `swift format` writes from `.swift-format`. This pass added no new kind of swiftlint finding. The `file_length` and `type_body_length` counts grew, because the card asks for documentation comments and the extraction.
  timestamp: 2026-08-13T16:35:25.824942+00:00
- actor: claude-code
  id: 01kzxzn75ga16e6r21rarsfwhm
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLLM/Models/DeepSeekV4.swift, Libraries/MLXLLM/Models/DeepSeekV4Attention.swift, Tests/MLXLMTests/DeepSeekV4ModelTests.swift. All 6 findings closed. `swift test`: 870 tests in 105 suites passed, 0 failed, 0 compiler warnings, 129 s. `swift format lint --strict --configuration .swift-format` on the 3 files: exit 0. `swift format` gives each of the 3 files back unchanged. Real weights: `xcodebuild test` limited to `DeepseekV4IntegrationTests/loadsTheRealCheckpointEndToEnd()` and `greedyFirstTokensMatchThePythonFixture()`: 2 tests passed, 41.2 s, TEST SUCCEEDED. swiftlint: each finding on the 3 files is present at HEAD on the same construct; no new kind.
    - next: /review
  timestamp: 2026-08-13T16:35:33.168470+00:00
position_column: doing
position_ordinal: '80'
title: Correct the two review findings of DeepSeekV4.swift that earlier commits left
---
## What

The review of task `^r92pjcr` swept the whole of `Libraries/MLXLLM/Models/DeepSeekV4.swift` and gave findings on lines that the commit of that task did not touch. A later review gave four more of the same kind on `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`. The standing rule from task `^ag7ant0` makes such a finding a record only, thus each review left them open. This task corrects all six, because they are one class of debt.

`git blame` gives commit `a75a043a` for the first line and commit `2624f899` for the second.

## Findings

`Libraries/MLXLLM/Models/DeepSeekV4.swift`

- [x] `DeepSeekV4DecoderLayer` — Class is not designed for subclassing and should be marked `final` to prevent accidental subclassing and enable compiler optimizations. Mark the class as final: `final class DeepSeekV4DecoderLayer: Module {`.
- [x] `projectionTensors` — the property lacks a documentation comment.
- [x] `stackRoutedExperts` — The for-key loop sits at nesting depth 4 (inside three for-loops and a guard statement), exceeding the maximum recommended depth of 3, which impairs readability and makes control flow difficult to follow. Extract the guard and inner for-key loop into a separate helper function, for example `removePerExpertWeights(from:perExpert:)`, to reduce `stackRoutedExperts` nesting to 3 levels.

`Libraries/MLXLLM/Models/DeepSeekV4Attention.swift`

- [x] `wqB` — the `@ModuleInfo` property lacks a documentation comment.
- [x] `wkv` — the `@ModuleInfo` property lacks a documentation comment.
- [x] `woA` — the `@ModuleInfo` property lacks a documentation comment.
- [x] `woB` — the `@ModuleInfo` property lacks a documentation comment.

A finding gives one example of a cause. Remove each cause from the whole of each file: mark every class that nothing subclasses `final`, give every member a documentation comment, and take every loop deeper than three levels down to three.

## What the work found

`projectionTensors` already carried a documentation comment. The cause the finding names sits on the `@ModuleInfo` properties of both files, and the pass gave a comment to each one.

## Acceptance Criteria

- [x] Each class of `DeepSeekV4.swift` and of `DeepSeekV4Attention.swift` that nothing subclasses is `final`.
- [x] Each member of both files carries a documentation comment.
- [x] No loop of either file sits deeper than three levels.
- [x] `swift test` is green, with no new warning.
- [x] `swift format --configuration .swift-format` gives each changed file back unchanged.
- [x] The two real-weights DeepSeek-V4 integration tests pass, which shows that the `stackRoutedExperts` change moved no number.

#deepseek-v4
