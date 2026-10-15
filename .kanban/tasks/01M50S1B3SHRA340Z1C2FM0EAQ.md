---
assignees:
- claude-code
position_column: todo
position_ordinal: '9280'
title: 'ChatSession: no model-independent thinking switch for MLXLMCommon callers'
---
## What

Found in ^yp1jxca. `ChatSession` and `UserInput` have no thinking switch. A caller must know the template key of each model (`enable_thinking` for Qwen, `thinking` for DeepSeek-V4). The qwen38bench used the DeepSeek-V4 key on Qwen3.8, and thinking stayed on. `ReasoningConfig.promptStrategy` is the shared switch, but only `MLXLanguageModel` (MLXFoundationModels) applies it. For Qwen 3.5 the switch keeps `enable_thinking` true and needs `closingReasoning(in:with:tokenizer:)` after the render; `ChatSession` does not do that step.

## Decision for the user

Must the MLXLMCommon path get one thinking switch (for example on `ChatSession` or `UserInput`) that applies `reasoningConfig.promptStrategy`, closed block included? Or do callers of that path keep the model keys?

#qwen3-8