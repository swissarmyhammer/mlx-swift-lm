---
assignees:
- claude-code
position_column: todo
position_ordinal: 8f80
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

- [ ] The three properties have their new names.
- [ ] The `CodingKeys` string values did not change.
- [ ] No default value changed.
- [ ] Each use of the old names is updated.

## Tests

- [ ] `swift build`
- [ ] `swift test --filter DeepseekV4ConfigurationTests`
- [ ] Full `swift test` stays green.
- [ ] The mutation proof still holds: a corrupted default must fail the suite.

#deepseek-v4
