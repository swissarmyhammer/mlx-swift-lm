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
- actor: claude-code
  id: 01kzs65m1s3tdb9zdbvvrf9va4
  text: |-
    ## Implementation record

    ### How the search was done

    The command `git ls-files '*.swift' | xargs grep -n 'Deepseek'` gave all 234 occurrences in less than one second. NOTE: a grep of `IntegrationTesting/` as a directory does not complete, because that folder holds its own `.build` with a full mlx-swift checkout. Use `git ls-files` for each search in this repository.

    ### What changed

    - 30 type declarations now spell `DeepSeek`: 19 under `Libraries/` (7 DeepSeek-V3 types, 12 DeepSeek-V4 types), 10 test suites under `Tests/MLXLMTests/`, and 1 under `IntegrationTesting/`.
    - Every reader changed with them: the model files, `LLMModelFactory.swift`, `MiniMaxM3.swift`, the tests, and the DocC references.
    - All 19 files that spelled `Deepseek` in their names moved with `git mv`, thus git keeps their history. The boundary decision is one answer for the full family: the test types and the file names all take the `DeepSeek` spelling, and the build and the tests stay green with them.

    ### What did NOT change, and why

    - 32 comment lines keep the `Deepseek` spelling. Each one is a citation: a file path of an external repository at a 40-character commit id (osaurus-ai/vmlx-swift-lm, scouzi1966/mlx-swift-lm), or a Python class of the reference (`mlx_lm/models/deepseek_v4.py` and Hugging Face `transformers`), which spell the name `Deepseek`. A rename there would make the citation point at a name that does not exist.
    - No checkpoint-facing string changed: `"deepseek_v3"` in the registry, `"DeepseekV4ForCausalLM"` in the fixture `Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json`, and every snake_case `CodingKeys` value. `git status` shows zero changes under `Tests/MLXLMTests/Resources/`.
    - `THIRD-PARTY-NOTICES.md` keeps its citation of the external repository.

    ### Proof

    - `swift build`: complete, 0 errors.
    - `swift test --filter DeepSeek`: 96 tests in 8 Swift Testing suites passed, and 38 XCTest tests in 3 suites passed (DeepSeekV4ChatEncoderTests 19, DeepSeekV4ConfigurationTests 8, DeepSeekV4ToolEncodingTests 11). 0 failures. The configuration tests and the quantization-plan tests passed with no fixture change.
    - `git ls-files '*.swift' | xargs grep -n 'Deepseek'` now returns only the 32 citation lines.

    ### Known items for the later full-suite step

    - The test build shows one Sendable warning in `Tests/MLXLMTests/ParoQuantTests.swift`. That file is not part of this change; the warning was there before it.
    - The `IntegrationTesting` Xcode project uses file-system-synchronized groups and names no `.swift` file in its `project.pbxproj`, thus the file rename there is safe. That project builds with `xcodebuild`, not with `swift test`.
  timestamp: 2026-08-11T19:53:09.945382+00:00
- actor: claude-code
  id: 01kzs65p7xvtxy24zdef3bdrh5
  text: |-
    ### implement — changed
    - evidence: 30 type declarations renamed to the DeepSeek spelling and 19 files moved with `git mv`. 20 Swift files edited. `swift build` passed. `swift test --filter DeepSeek` passed: 96 Swift Testing tests and 38 XCTest tests, 0 failures. No checkpoint-facing string changed and no fixture changed. Only the 32 external-citation comment lines keep the old spelling.
    - next: run the full test suite, then review.
  timestamp: 2026-08-11T19:53:12.189867+00:00
position_column: doing
position_ordinal: '8180'
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