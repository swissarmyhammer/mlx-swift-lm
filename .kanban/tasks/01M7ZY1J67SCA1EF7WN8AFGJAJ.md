---
assignees:
- claude-code
position_column: todo
position_ordinal: '8380'
title: 'Inkling 2: decide the FoundationModels required tool mode (grammar + think-then-call) for the frame protocol'
---
Found while working ^1sg8f5t. `MLXLanguageModel` `.required` tool mode (`runRequiredToolTurn`) constrains output with the `<tool_call>` JSON structural-tag grammar and, for a `.templateFlag` reasoning protocol with reasoning declared, runs a think-then-call phase that reasons until the protocol end delimiter. Inkling 2 writes tool calls in `<|message_model|>NAME<|content_invoke_tool_json|>{...}<|end_message|>` frames, so the grammar shape does not match the model's protocol. Since ^1sg8f5t, `InklingReasoningProtocol` is `.templateFlag(reasoning_effort, thinkingOff: .templateValue("none"))`, so the think-then-call phase now applies to Inkling in required mode (before, `.none` skipped it). Router/ACPAgent use `.allowed`, which works end to end.

- [ ] Measure required mode on mlx-community/Inkling-2-Mini-4bit (IntegrationTesting) with a tool loop.
- [ ] Either give the frame protocols (gptOSS/atem/inkling) a required-mode path that matches their frames, or reject required mode for them with a typed error, with unit tests.

#inkling