---
assignees:
- claude-code
position_column: todo
position_ordinal: '9380'
title: 'DeepSeek-V4: the doc of thinkingKey names a ReasoningConfig row that does not exist'
---
## What

Found in ^yp1jxca. The doc of `DeepSeekV4EncodingTokenizer.thinkingKey` (Libraries/MLXLMCommon/Tokenizer.swift) says the key is "the same key the family's `ReasoningConfig` row injects". DeepSeek-V4 declares no `reasoningConfig` (DeepSeekV4ChatEncoder.swift says so), and no resolver row gives one. Thus the shared thinking switch of MLXFoundationModels does not reach DeepSeek-V4.

## Acceptance Criteria

- [ ] The doc and the code agree: correct the doc, or add the row with a test.

#bug