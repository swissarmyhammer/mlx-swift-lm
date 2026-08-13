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
- actor: claude-code
  id: 01kzy0cph12dxq3jg3rhkc6tbq
  text: |-
    ### review — findings
    - evidence: 2 findings, both confirmed — Libraries/MLXLLM/Models/DeepSeekV4.swift:162, Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:462. Engine ops: `review file` on each of the two changed Swift sources, one at a time, because range mode counts validators and not files. Each file gave 8 validators; DeepSeekV4.swift refuted 3 more. `review file Tests/MLXLMTests/DeepSeekV4ModelTests.swift` gave "Nothing in scope to review" with 0 validators, thus the engine read no test file. Commit 507d5fa touched neither of the two named lines, thus rule ^ag7ant0 holds both out of scope and each is a record only. No new card opened for them. None of the six findings of this card came back, and a direct check confirms all five classes `final`, every `@ModuleInfo` property and every `YarnScaling` property documented, and `stackRoutedExperts` at depth 3. The moved guard keeps its meaning: `perExpert.count` is always `expertCount`, thus `stack.count == perExpert.count` and `stack.count == expertCount` agree for every input, and `return` in the helper gives the same control flow as the old `continue`.
    - next: a person must say if this card corrects the two out-of-scope findings, or if the card advances to done with them as a record.
  timestamp: 2026-08-13T16:48:22.561251+00:00
- actor: claude-code
  id: 01kzy16zt5barqw11268vdgjv5
  text: |-
    The card came back from `review` to `doing`, because a person put the two out-of-scope findings IN SCOPE. What the pass did, and what it found.

    **The `ffn` finding is not a conflict.** The card that dispatched this pass asked first whether `ffn` is the checkpoint key that the Swift name must match. It is not. The key sits in the attribute, `@ModuleInfo(key: "ffn")`, and the property name is free of it. The file proves this itself: `@ModuleInfo(key: "attn") var attention`, `@ModuleInfo(key: "attn_norm") var attentionNorm` and `@ModuleInfo(key: "hc_attn") var attentionConnection` all carry a checkpoint key that is not the property name. Thus the rename is safe and the three keys `"ffn"`, `"ffn_norm"` and `"hc_ffn"` did not move. A grep of both files after the rename shows every remaining `ffn` is a string literal or a comment about a checkpoint key.

    **The sweep of the abbreviation cause went to all three names of the pair.** `ffnNorm` and `ffnConnection` break the same rule as `ffn`, beside the parallel `attentionNorm` and `attentionConnection`. A rename of `ffn` alone would leave `mixtureOfExperts` next to `ffnNorm`, which is a worse inconsistency than the one the finding names. All three now read `mixtureOfExperts*`.

    **The `private` finding is a true conflict.** The finding says the method "is called only from within the same class". That is not correct. `Tests/MLXLMTests/DeepSeekV4AttentionTests.swift` calls `block.groupedOutputProjection(input)` at line 385, in the test that holds the numeric parity fixture of `V4Attention._grouped_output_projection` of the Python reference, and a second test uses the same helper path. The pass did not guess: it applied `private`, ran `swift build --build-tests`, and got

        Tests/MLXLMTests/DeepSeekV4AttentionTests.swift:385:31: error: 'groupedOutputProjection' is inaccessible due to 'private' protection level
        error: Build failed

    `@testable import` opens `internal` and never `private`, thus no build setting makes the fix compile. The declaration went back to what it was, and `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift` is unchanged in the working tree. The pass invented no workaround: it did not spell an explicit `internal`, it did not delete a test, and it did not rewrite a test to go through `callAsFunction`, which would throw the parity fixture away.

    **New work found.** The local `comb` of `wrapped` is an abbreviation, but its name is the tuple label of `DeepSeekV4HyperConnection.collapse`, in a file this card does not cover. Card `^dhzt3cr` holds the cross-file correction.

    **A check of the other declarations.** The description now carries the full list, with a reason for each name the pass left alone. In short: `hcHead` and `hcMult` follow the key-mirror pattern of their own type and of the whole DeepSeek-V4 family, every abbreviated name of `DeepSeekV4Attention.swift` mirrors a checkpoint key or a configuration key with no exception, and every method of both files that is reachable only from inside its type is already `private`.
  timestamp: 2026-08-13T17:02:44.037377+00:00
- actor: claude-code
  id: 01kzy17bkvkhpm2dewte39cym8
  text: |-
    ### implement — stuck
    - evidence: 3 files changed — Libraries/MLXLLM/Models/DeepSeekV4.swift, Tests/MLXLMTests/DeepSeekV4ModelTests.swift, Tests/MLXLMTests/DeepSeekV4GateCheckpointLoadTests.swift. Finding 1 of 2 closed: `DeepSeekV4.swift:162` `ffn` -> `mixtureOfExperts`, and the sweep took `ffnNorm` -> `mixtureOfExpertsNorm` and `ffnConnection` -> `mixtureOfExpertsConnection` with it; every `@ModuleInfo(key:)` string is unchanged. Finding 2 of 2 recorded as a TRUE CONFLICT, not corrected: `DeepSeekV4Attention.swift:462` `private func groupedOutputProjection` gives `Tests/MLXLMTests/DeepSeekV4AttentionTests.swift:385:31: error: 'groupedOutputProjection' is inaccessible due to 'private' protection level` and `error: Build failed` from `swift build --build-tests`; `DeepSeekV4Attention.swift` is unchanged. `swift test`: 870 tests in 105 suites (501 + 80 + 282 + 7), 0 failed, 135 s; the one `warning:` line of the log is the llbuild message `missing creator for mutated node: .../mlx-swift_Cmlx.bundle`, which is a build-system message and not a compiler warning. `swift format lint --strict --configuration .swift-format` on the 3 changed files: exit 0. `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`: ** TEST BUILD SUCCEEDED **, 0 errors. No real-weights run: the change is a rename and moves no number.
    - next: a person must decide the `private` conflict. Three ways out: (a) drop the finding, because the method is the surface two parity tests read; (b) accept a different fix that the finding does not give; (c) permit the two tests to change, which throws the parity fixture of `V4Attention._grouped_output_projection` away. The pass will not choose one of these.
  timestamp: 2026-08-13T17:02:56.123477+00:00
- actor: claude-code
  id: 01kzy1af50sgtepckk694zxhfc
  text: |-
    ### finish iteration 2 — stuck
    - implement: changed for one finding, stuck on the other. `ffn`, `ffnNorm` and `ffnConnection` became `mixtureOfExperts`, `mixtureOfExpertsNorm` and `mixtureOfExpertsConnection`; each `@ModuleInfo` key string stays as it was.
    - test: green — `swift test`, 870 tests in 105 suites, 0 failures, 135 s. `xcodebuild build-for-testing`: TEST BUILD SUCCEEDED.
    - commit: 465085d fix(mlx-lm): give the DeepSeekV4 ffn properties their full name
    - STUCK on the second finding, and a person must decide. The finding asks for `private` on `DeepSeekV4Attention.groupedOutputProjection`. `Tests/MLXLMTests/DeepSeekV4AttentionTests.swift:385` calls that method in the numeric parity test of the Python `V4Attention._grouped_output_projection`. With `private` the build stops: "error: 'groupedOutputProjection' is inaccessible due to 'private' protection level". `@testable import` opens `internal` and never `private`, thus the finding cannot hold together with the test that is there now. The two ways out are (a) drop the finding under the written rule that releases a finding which asks to rewrite a test that already existed, or (b) keep the finding and delete or move the parity test. This agent took neither, because the choice is a person's.

    The loop stops here. The card stays out of `done` with one finding open. Do not force it.
  timestamp: 2026-08-13T17:04:38.048918+00:00
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

## Review Findings (2026-08-13 11:42)

A person put both findings below IN SCOPE for this card, because this card pays this class of debt.

- [x] `Libraries/MLXLLM/Models/DeepSeekV4.swift:162` — Property `ffn` uses an abbreviation while the parallel component `attention` (line 159) uses the full word. This inconsistency reduces clarity and violates the guidance to avoid abbreviating to save characters. Rename `ffn` to `mixtureOfExperts` to match the full-word style of `attention` and improve clarity. Update call sites at lines 188, 196, and 224.
- [ ] `Libraries/MLXLLM/Models/DeepSeekV4Attention.swift:462` — Instance method `groupedOutputProjection` lacks an explicit access modifier and defaults to `internal`, but it is called only from within the same class (line 419: `return woB(groupedOutputProjection(flattened))`). Helper methods used only internally should be marked `private`. Add `private` before `func` on line 462: `private func groupedOutputProjection(_ output: MLXArray) -> MLXArray {`. **BLOCKED — true conflict, see below.**

## Blocker: the `private` finding cannot compile

The premise of the finding is not correct. `groupedOutputProjection` is NOT called only from
inside the class. `Tests/MLXLMTests/DeepSeekV4AttentionTests.swift` calls it from two tests,
`groupedOutputProjectionMatchesThePythonReference` (line 385) and
`groupedOutputProjectionReachesTheFullHiddenSize`. The first test holds the numeric parity
fixture of `V4Attention._grouped_output_projection` of the Python reference.

`private` reaches only the same declaration and the same-file extensions of that declaration.
`@testable import` gives a test target the `internal` members, and never the `private` ones,
thus the fix the finding gives makes the test target fail to build. This is not an opinion. The
pass applied `private` and ran `swift build --build-tests`:

```
Tests/MLXLMTests/DeepSeekV4AttentionTests.swift:385:31: error: 'groupedOutputProjection'
is inaccessible due to 'private' protection level
error: Build failed
```

The pass then put the declaration back as it was.

This is a true conflict of the second kind: a rule that needs code that cannot compile. The
only ways to make `private` compile are to delete the two tests or to rewrite them to go
through `callAsFunction`, which throws away the parity fixture of one Python routine. A
finding does not permit either, thus the pass changed nothing here, invented no workaround,
and records the conflict. A person must decide.

## The sweep of each cause

A finding gives one example of a cause. The pass looked at every declaration of both files.

**The abbreviation cause.** The finding names the inconsistency: an abbreviation beside a
parallel declaration that spells the same idea in full. `DeepSeekV4DecoderLayer` holds six
module properties, and three of them spelled the idea in full (`attention`, `attentionNorm`,
`attentionConnection`) while three abbreviated it. All three now spell it in full:

- `ffn` -> `mixtureOfExperts`
- `ffnNorm` -> `mixtureOfExpertsNorm`
- `ffnConnection` -> `mixtureOfExpertsConnection`

Every `@ModuleInfo(key:)` string stays as it was (`"ffn"`, `"ffn_norm"`, `"hc_ffn"`), thus the
checkpoint contract does not move. The three call sites of the file and the six call sites of
`Tests/MLXLMTests/DeepSeekV4ModelTests.swift` and
`Tests/MLXLMTests/DeepSeekV4GateCheckpointLoadTests.swift` take the new names.

The pass left these abbreviations, each with a reason:

- `hcHead` and `hcMult` of `DeepSeekV4ModelInner`. That type mirrors its checkpoint key in
  every name it holds -- `embedTokens` for `embed_tokens`, `hcHead` for `hc_head`, `norm` for
  `norm` -- thus `hcHead` is the pattern of its own type, not a break from it. `hcMult` is a
  local copy of `configuration.hcMult`, which is the name the whole DeepSeek-V4 family gives
  the number of parallel copies, in `DeepSeekV4Configuration`, `DeepSeekV4HyperConnection`,
  `DeepSeekV4MathHelpers` and in the shape lines of this file. A rename here alone would make
  this file disagree with its four siblings.
- Every abbreviated name of `DeepSeekV4Attention.swift`: `wqA`, `wqB`, `wkv`, `woA`, `woB`,
  `qNorm`, `kvNorm`, `attnSink`, `useAttnSink`, `headDim`, `ropeDim`, `normEps`. Each one
  mirrors a checkpoint key (`wq_a`, `attn_sink`, `kv_norm`) or a configuration key
  (`use_attn_sink`, `head_dim`). That file mirrors its keys with no exception, thus it holds
  no parallel pair of the kind the finding names, and a rename of one name alone would make
  the inconsistency the finding punishes.
- The local `comb` of `wrapped`. The name is the tuple label of
  `DeepSeekV4HyperConnection.collapse`, which is a different file that this card does not
  cover. A rename of the local half alone would read `comb: mixing` at the call. Card
  `^dhzt3cr` holds the cross-file correction.

**The missing-`private` cause.** The finding words the cause as "Helper methods used only
internally should be marked `private`". The pass read every method of both files:

- Already `private`: `wrapped`, `isLoaded`, `layerIndex`, `modulePath`, `stackRoutedExperts`,
  `stackPerExpertWeights`, `write`, `yarnScaling`, `rotatedQueries`, `rotatedKeyValues`,
  `quantizedGroupedProjection`.
- Part of the used surface: `DeepSeekV4NumericTrace.tokens` and `.tensor` (three different
  types of the file call them, thus `private` would not compile), `cosSin(offset:length:)` and
  `cosSin(positions:)` (`DeepSeekV4Compressor` and `DeepSeekV4Indexer` call them), every
  `callAsFunction`, every `init`, `layers`, `sanitize`, `loraLayers`.
- `groupedOutputProjection`: the blocker above.

No other method of either file needs the change.
