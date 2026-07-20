---
assignees:
- claude-code
depends_on:
- 01KXY0ZVCCPBKZ1ANETWZ8Y8QQ
position_column: todo
position_ordinal: 8d80
title: 'MiniMax-M3: tool calling + reasoning wiring through FoundationModels'
---
## What

Wire M3 into the tool-calling and reasoning inference tables, to the same bar the M2 task (^9mv1q33) met:

1. **`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`**: add an inference-table row for model_type `minimax_m3_vl` (and the bare `minimax_m3` text variant id if mlx-vlm/model cards use it). FIRST verify against the real repo's `chat_template.jinja` (fetch from `mlx-community/MiniMax-M3-4bit`) whether M3 uses M2's tool-call format (`.minimaxM2`, parsed by `MiniMaxM2ToolCallParser`) or a new one; if new, add a parser following `Libraries/MLXLMCommon/Tool/Parsers/` conventions. Do not assume — the M2 task's failure mode was exactly a template/render mismatch.
2. **`Libraries/MLXLMCommon/ReasoningConfig.swift`**: add an inference row for `minimax_m3_vl` (M3 is an interleaved-thinking model; expect `<think>`/`</think>` with `.alwaysOn` strategy like M2 — verify against the chat template's handling of thinking content before wiring).
3. **`Libraries/MLXVLM/VLMModelFactory.swift`** — CRITICAL, found by plan double-check: `VLMModelFactory._load` calls `ToolCallFormat.infer` but NEVER calls `ReasoningConfig.infer` (the LLM factory does, at `LLMModelFactory._load`); a ReasoningConfig table row alone is a runtime no-op for M3, which loads through the VLM factory — thinking would leak into text replies exactly like the M2 failure mode. Wire `ReasoningConfig.infer` into `VLMModelFactory._load` and thread the resulting `reasoningConfig` into the VLM-built `ModelConfiguration`, mirroring the LLM factory. This is a factory-wide behavior change: verify other VLM models' behavior is unchanged (their infer rows return nil unless matched).
4. **`Libraries/MLXFoundationModels/TranscriptConverter.swift`**: confirm the structured tool-call rendering path added for `.minimaxM2` covers M3's template validation (tool message must follow assistant `tool_calls`); extend if the format differs.

## Acceptance Criteria

- [ ] `ToolCallFormat.infer` and `ReasoningConfig.infer` return the verified format/config for model_type `minimax_m3_vl` (unit tests pin both)
- [ ] `VLMModelFactory._load` threads an inferred `reasoningConfig` into `ModelConfiguration` (unit/regression: a VLM model with no matching infer row loads with `reasoningConfig == nil`, unchanged behavior)
- [ ] Multi-turn tool exchange renders through M3's real `chat_template.jinja` without TemplateException (template-validator mirror test, same approach as the M2 fix)
- [ ] One real tool-call round trip completes end to end on `mlx-community/MiniMax-M3-4bit` (gated integration test: `toolCalls → toolOutput → response`)
- [ ] Thinking output lands in `.reasoning` entries, never leaking into the text reply (assert in the integration test)
- [ ] M2/Mistral/Qwen/GLM rendering unchanged (existing regression tests stay green)

## Tests

- [ ] Extend `Tests/MLXLMTests/ToolTests.swift` + `Tests/MLXLMTests/ReasoningConfigTests.swift`: inference-table rows for `minimax_m3_vl`
- [ ] New/extended VLM factory test: reasoningConfig threading (no-match → nil; minimax_m3_vl → the M3 config)
- [ ] Extend `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift`: M3 multi-turn tool rendering (mirroring the existing minimax tests)
- [ ] Gated integration: tool round trip + think-leak assertion against real weights (FoundationModels multitool suite pattern)
- [ ] Run: `swift test` → green; integration case passes

## Workflow

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax