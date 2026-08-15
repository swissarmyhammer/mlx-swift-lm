---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyvggzw5yj89s9v4nh6m72y
  text: |-
    ### Closed without the rename — a decision, not work

    The user closed this card on 2026-08-14 and asked for no rename.

    **The code still spells the name `comb`.** Nothing changed. `DeepSeekV4HyperConnection.collapse` still answers `(collapsed:, post:, comb:)`, `expand` still takes `comb:`, and `DeepSeekV4Math.hcSplitSinkhorn` still answers the same third value under the same name. A later reader must not read this card as done work.

    The rename would have touched 59 sites in five files: `DeepSeekV4HyperConnection.swift`, `DeepSeekV4.swift`, `DeepSeekV4MathHelpers.swift`, `DeepSeekV4HyperConnectionTests.swift`, `DeepSeekV4MathHelpersTests.swift` and `DeepSeekV4ModelTests.swift`, plus the name `expectedComb` and the documentation text. That is a wide change of names alone, and it gives no behaviour.

    Open this card again if the abbreviation gets in the way of a reader later.
  timestamp: 2026-08-14T00:42:19.516680+00:00
position_column: done
position_ordinal: f580
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