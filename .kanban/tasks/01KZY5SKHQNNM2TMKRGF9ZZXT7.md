---
assignees:
- claude-code
position_column: todo
position_ordinal: 9c80
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