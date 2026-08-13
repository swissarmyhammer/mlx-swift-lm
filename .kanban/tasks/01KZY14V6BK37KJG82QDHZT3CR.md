---
assignees:
- claude-code
position_column: todo
position_ordinal: '9980'
title: Give the `comb` tuple label of DeepSeekV4HyperConnection its full word
---
## What

`DeepSeekV4HyperConnection.collapse` answers a tuple with the labels
`(collapsed:, post:, comb:)`, and `expand` takes a parameter with the label `comb:`. The
documentation of `expand` spells the same thing in full: "The doubly stochastic mixing
matrix". `comb` is thus an abbreviation that the file itself spells in full one line above.

The rule the reviews apply is "Clarity over brevity. Don't abbreviate to save characters".

Card `^dhv1ave` found this while it swept the abbreviations of
`Libraries/MLXLLM/Models/DeepSeekV4.swift`. That card left the local binding
`let (collapsed, post, comb) = connection.collapse(stream)` alone, because the name comes
from the tuple label of another file, and a rename of the local half alone would read
`comb: mixing` at the call and would make the two files disagree.

## What to do

- Rename the tuple label of `DeepSeekV4HyperConnection.collapse` from `comb` to `mixing`.
- Rename the parameter label of `DeepSeekV4HyperConnection.expand` from `comb` to `mixing`.
- Take every call site with it, in `Libraries/MLXLLM/Models/DeepSeekV4.swift`,
  in `Libraries/MLXLLM/Models/DeepSeekV4HyperConnection.swift` and in the tests.
- Keep the documentation comments, and give them the new word.
- Change no number. The rename touches names alone.

Look at `DeepSeekV4Math.hcSplitSinkhorn`, which answers the same three values, and give it
the same word if it carries the abbreviation too.

## Acceptance Criteria

- [ ] No declaration and no label of the hyper-connection path spells the mixing matrix
      `comb`.
- [ ] `swift test` is green, with no new warning.
- [ ] `swift format lint --strict --configuration .swift-format` passes on each changed file.

#deepseek-v4