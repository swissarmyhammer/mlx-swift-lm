---
assignees:
- claude-code
position_column: todo
position_ordinal: '9780'
title: Correct the two review findings of DeepSeekV4.swift that earlier commits left
---
## What

The review of task `^r92pjcr` swept the whole of `Libraries/MLXLLM/Models/DeepSeekV4.swift` and gave two findings on lines that the commit of that task did not touch. The standing rule from task `^ag7ant0` makes such a finding a record only, thus `^r92pjcr` left them open. This task corrects them.

`git blame` gives commit `a75a043a` for the first line and commit `2624f899` for the second.

## Findings

- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:156` — Class is not designed for subclassing and should be marked `final` to prevent accidental subclassing and enable compiler optimizations. Mark the class as final: `final class DeepSeekV4DecoderLayer: Module {`.
- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift:581` — The for-key loop sits at nesting depth 4 (inside three for-loops and a guard statement), exceeding the maximum recommended depth of 3, which impairs readability and makes control flow difficult to follow. Extract the guard and inner for-key loop (lines 579–583) into a separate helper function, e.g. `removePerExpertWeights(from:perExpert:)`, to reduce stackRoutedExperts nesting to 3 levels.

A finding gives one example of a cause. Remove each cause from the whole file: mark every class that nothing subclasses `final`, and take every loop deeper than three levels down to three.

## Acceptance Criteria

- [ ] Each class of `DeepSeekV4.swift` that nothing subclasses is `final`.
- [ ] No loop of `DeepSeekV4.swift` sits deeper than three levels.
- [ ] `swift test` is green, with no new warning.
- [ ] `swift format --configuration .swift-format` gives each changed file back unchanged.

#deepseek-v4
