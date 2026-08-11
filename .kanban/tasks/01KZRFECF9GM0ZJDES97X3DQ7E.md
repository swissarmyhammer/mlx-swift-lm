---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzrg5kgq8zj1xvsh92hfnt03
  text: |
    ## Two names that fall between this task and `^sbwyq83`

    `^sbwyq83` corrected the two type declarations of its own new source file to
    `DeepSeekV4HyperConnection` and `DeepSeekV4HyperHead`. Its card drew the
    boundary at those two names.

    These carry the `Deepseek` spelling and NEITHER task covers them today:

    - `struct DeepseekV4HyperConnectionTests` in
      `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift`. This task scopes its
      rename to "every type declaration under `Libraries/`", thus a declaration
      under `Tests/` is outside it. The name is also the argument of
      `swift test --filter DeepseekV4HyperConnectionTests`, which the acceptance
      criteria of `^sbwyq83` record.
    - The two file names, `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift`
      and `Tests/MLXLMTests/DeepseekV4HyperConnectionTests.swift`. This task says
      "Do NOT rename file names, unless the whole rename stays green with them".

    Decide both when you run this task. The other test files of the family --
    `DeepseekV4AttentionTests`, `DeepseekV4MoETests` and the rest -- carry the same
    spelling, thus the answer should be one answer for all of them.
  timestamp: 2026-08-11T13:28:40.727665+00:00
- actor: claude-code
  id: 01kzrkmyf4ec0bn2sj82cjnhgv
  text: |
    ### finish iteration 1 — stuck, then put back to todo

    The agent made no change. It ran

    ```
    grep -rn 'Deepseek' --include='*' . --exclude-dir=.git --exclude-dir=.kanban -l
    ```

    from the repository root. That walks `.build`, which holds the whole mlx-swift
    checkout and its build products, thus the command never gave an answer. The MCP
    shell stopped it after 1800 seconds, and the agent then ran the same pattern a
    second time. I stopped the agent. `git status --porcelain -- '*.swift'` is
    empty, thus the tree carries no partial edit.

    **Read this before you start the task again.** Never walk the repository root.
    Two ways that work:

    ```
    git ls-files '*.swift' | xargs grep -n 'Deepseek'
    grep -rn 'Deepseek' Libraries/ Tests/
    ```

    `git ls-files` is the safer of the two, because it reads only what git tracks.

    ### The order changed

    The card says to run this task before the model assembly task `^pwr8r3h`. That
    was to stop assembly from writing a second spelling. The reason no longer holds:
    this task cost more than 40 minutes and gave nothing, and the goal of the night
    is a model that loads and answers. Assembly runs first. Assembly may draw the
    same casing finding on the names it declares, and one more iteration there costs
    less than this task has already cost.
  timestamp: 2026-08-11T14:29:29.188748+00:00
position_column: todo
position_ordinal: '9480'
title: Use the canonical DeepSeek spelling for every DeepSeek type name
---
## Why

The review of `^sbwyq83` gave this finding on 2026-08-11:

> `Libraries/MLXLLM/Models/DeepseekV4HyperConnection.swift:313` — Type name should use canonical spelling 'DeepSeek' (not 'Deepseek'); established mixed-case terms keep their canonical spelling per the casing rule.

`^sbwyq83` corrected the two type names that its own new file declares. It did
not correct the type names that other files declare, because its acceptance
criteria say "No changes to existing files".

That leaves the repository with two spellings. This task removes the second one.

## What

Rename every type declaration under `Libraries/` that spells the name
`Deepseek` to the canonical `DeepSeek`. The review counted 16 such
declarations. They include `DeepseekV4Configuration`, `DeepseekV4Attention`,
`DeepseekV4MoE`, `DeepseekV4Math` and the DeepSeek-V3 types.

Correct every place that reads those names: the other model files, the model
registry, and the tests.

## Do NOT rename

- Any string that a checkpoint gives. `model_type` is `deepseek_v4`, and the
  configuration `CodingKeys` hold snake_case keys such as `n_shared_experts`.
  A checkpoint decides those spellings, not this repository.
- File names, unless the whole rename stays green with them.

## Order

Run this task BEFORE the model assembly task `^pwr8r3h`, so that assembly
writes one spelling and not two.

## Acceptance Criteria

- [ ] No type declaration under `Libraries/` spells the name `Deepseek`.
- [ ] `swift build` and the full `swift test` stay green.
- [ ] No checkpoint-facing string changed. Prove it: the DeepSeek-V4
      configuration tests and the quantization-plan tests still pass without a
      change to their fixtures.
#deepseek-v4