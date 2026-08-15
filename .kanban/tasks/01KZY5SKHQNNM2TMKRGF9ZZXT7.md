---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyvw6xmp23tnpappdrje82m
  text: |-
    ### Closed — take the upstream value, and do not measure

    The user does not plan to use GLM, and decided on 2026-08-14 to keep the official upstream form and to leave our own GLM work alone.

    **No generation run was made.** Thus this card gives NO answer to the question it asked: nobody knows which format `mlx-community/GLM-4-9B-0414-4bit` really emits. The card is closed by a decision, not by evidence.

    **What the live tree holds now:**

    - `LLMRegistry.glm4_9b_4bit` keeps `toolCallFormat: .glm4`, the upstream value. No registry change was needed.
    - `Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift` no longer looks for `.glm4Bare` in `glm4FormatAutoDetection`. That assertion was left over from our own registry and had become false, thus the integration test would have failed as soon as somebody ran it against the real checkpoint. It looks for `.glm4` now.
    - `swift build` stays clean.

    **What stays in the tree and nothing selects:**

    - `ToolCallFormat.glm4Bare`
    - `GLM4BareToolCallParser`
    - three test sites in `Tests/MLXLMTests/ToolTests.swift` (lines 1009, 1033 and 1427)

    They build, they pass, and no registry entry names them. They are dead code, and they are cheap to keep. Delete them when somebody wants the tree smaller.

    **The risk this decision accepts:** our own documentation says the bare format belongs to the pre-4.7, non-MoE GLM-4 checkpoints and names GLM-4-9B-0414 as the example, and our old `infer` table put dense GLM-4 on `.glm4Bare` on purpose. If that reading was right, GLM-4-9B tool calls do not parse now, and they fail without a sign: the model gives a call, the parser reads prose, and the caller sees text. That is a cost only a GLM user pays.
  timestamp: 2026-08-14T00:48:42.420405+00:00
position_column: done
position_ordinal: f880
title: Confirm the GLM-4-9B tool-call format after the upstream catch-up
---
A behaviour difference that the "prefer upstream" rule settled one way, which a person should confirm.

## The difference

Before the merge our `LLMRegistry` entry for `mlx-community/GLM-4-9B-0414-4bit` read:

```swift
toolCallFormat: .glm4Bare
```

The official upstream entry reads:

```swift
toolCallFormat: .glm4
```

The upstream catch-up took the upstream `LLMModelFactory.swift`, so the live value is now `.glm4`.

## Why it may be wrong

`GLM4BareToolCallParser` and the `.glm4Bare` format are ours. Their doc comment says the bare format belongs to the original, pre-4.7, non-MoE GLM-4 checkpoints, and it names GLM-4-9B-0414 as the example:

> No wrapper tags or JSON envelope: the function name appears alone, followed by a bare JSON object of just the arguments.
> Example: `get_weather\n{"location": "Paris", "unit": "celsius"}`

Our pre-merge `ToolCallFormat.infer` table also held `(.exact, "glm4", .glm4Bare)`, which put dense GLM-4 on the bare format and left the `glm4` prefix, the MoE checkpoints, on `.glm4`.

`.glm4Bare` and its parser both still exist and still build. Only the registry entry that selects them changed.

## Decide

Run GLM-4-9B-0414-4bit with a tool and read what it emits.

- If it emits the bare format, set the registry entry back to `.glm4Bare` and think about sending the fix upstream.
- If it emits `arg_key`/`arg_value`, keep `.glm4` and remove `.glm4Bare` along with its parser, since nothing would then select it. #upstream-catch-up-tool-calling