---
assignees:
- claude-code
position_column: todo
position_ordinal: 9b80
title: Reduce the nesting of DeepSeekV4Model.stackRoutedExperts
---
## What

The review of task `^ccygfkn` swept `Libraries/MLXLLM/Models/DeepSeekV4.swift` and gave one finding on a function that task did not touch. Task `^ccygfkn` is a documentation-comment task, and its scope states "comments only, no code change", thus the finding cannot land there. This task holds it.

The standing rule from task `^ag7ant0` makes a finding on a line the diff never touched a record only.

## Finding

- [ ] `Libraries/MLXLLM/Models/DeepSeekV4.swift` `complexity/cognitive-complexity` — Function has 4 levels of nesting (for layer -> for projection -> for tensor -> map closure), making control flow and reasoning about behavior harder. Extract the innermost operation into a named helper function. For example, refactor the map and stackPerExpertWeights call into a single method that takes layer, projection, and tensor as parameters, reducing the nesting depth from 4 to 2 or 3.

The function the finding names is `private func stackRoutedExperts(in weights: inout [String: MLXArray])` of `DeepSeekV4Model`. `git log -L` on the function gives the commit that wrote it; the documentation task `^ccygfkn` only moved its line numbers.

## Acceptance Criteria

- [ ] `stackRoutedExperts` nests no more than 3 levels.
- [ ] The behaviour does not change. The load filter maps the same checkpoint keys onto the same module paths.
- [ ] `review file Libraries/MLXLLM/Models/DeepSeekV4.swift` gives no `cognitive-complexity` finding.
- [ ] `swift build` is green, and `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test` is green with no new warning.
- [ ] `swift format lint --strict --configuration .swift-format Libraries/MLXLLM/Models/DeepSeekV4.swift` exits 0.

#deepseek-v4