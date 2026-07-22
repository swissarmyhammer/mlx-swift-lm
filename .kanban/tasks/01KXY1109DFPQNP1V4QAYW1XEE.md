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

1. **`Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`**: add an inference-table row for model_type `minimax_m3_vl` (and the bare `minimax_m3` text variant id if mlx-vlm/model cards use it). FIRST verify against the real repo's `chat_template.jinja` (fetch from `mlx-community/MiniMax-M3-4bit`) whether M3 uses M2's tool-call format (`.minimaxM2`, parsed by `MiniMaxM2ToolCallParser`) or a new one; if new, add a parser following `Libraries/MLXLMCommon/Tool/Parsers/` conventions. Do not assume — the M2 task's failure mode was exactly a template/render mismatch. (See the ^a7swy35 folded findings below: the format was already verified to be NEW, not M2's.)
2. **`Libraries/MLXLMCommon/ReasoningConfig.swift`**: add an inference row for `minimax_m3_vl` (M3 is an interleaved-thinking model — see the ^a7swy35 folded findings below: the template was verified to use toggleable `<mm:think>`, NOT M2's always-on `<think>`).
3. **`Libraries/MLXVLM/VLMModelFactory.swift`** — the `ReasoningConfig.infer` wiring into `VLMModelFactory._load` was ALREADY IMPLEMENTED by task ^05zt40g (commit 5891f01, landed for the qwen work): the factory now infers and threads `reasoningConfig` into the VLM-built `ModelConfiguration`, mirroring the LLM factory. For this task, VERIFY the existing inference handles `minimax_m3_vl`'s new row correctly — the factory wiring itself is done; do not re-implement it.
4. **`Libraries/MLXFoundationModels/TranscriptConverter.swift`**: confirm the structured tool-call rendering path added for `.minimaxM2` covers M3's template validation (tool message must follow assistant `tool_calls`); extend if the format differs.

### Folded from ^a7swy35 (chain reconciliation 2026-07-22)

Verified format findings (against the checkpoint's `chat_template.jinja` and upstream mlx-lm PR #1416, which adds a separate `minimax_m3.py` parser) — M3's formats are NOT M2's:

- **Tool calls**: namespaced XML with parameters as arbitrary `<key>value</key>` children — unlike M2's `<parameter name="k">v</parameter>`. The M2 parser cannot be reused: add `Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM3ToolCallParser.swift` following the M2 parser + `ParserUtilities.swift` conventions, and a new `ToolCallFormat` case (e.g. `.minimaxM3`) registered for the `minimax_m3` / `minimax_m3_vl` model types. Take parser fixture strings verbatim from the chat template / PR #1416 test cases; cover multi-parameter and nested-value cases, and match the streaming/partial parse contract used by the M2 parser tests.
- **Reasoning tags**: the template uses toggleable `<mm:think>` — NOT M2's always-on `<think>` (`ReasoningConfig.swift:201-208`). Wire the `<mm:think>` delimiters with the correct toggle semantics as expressed by the template, not `.alwaysOn`.
- **MLXFoundationModels**: add the new case to `structuredToolCallFormats` in `Libraries/MLXFoundationModels/TranscriptConverter.swift:218` (currently `[.mistral, .minimaxM2]`) so M3 tool turns replay through the chat template's `tool_calls` branch.
- **Chat-template probe**: the M3 template uses macros and namespace-token concatenation that may stress swift-transformers' Jinja engine — add an `ApplyChatTemplateProbeTests`-style test that renders the real M3 chat template (checked-in fixture) with a system+user+tool conversation, catching engine gaps before the real-weights integration test.
- Keep all M2 behavior unchanged.

### Folded from ^b90razv (chain reconciliation 2026-07-22)

- The gated tool-round-trip integration test follows the same runner conventions as ^wz8y8qq's coherence test: it lives in `IntegrationTesting/` (Xcode project, run via `xcodebuild` — NOT `swift test`), is gated by the `DeviceTier.swift` convention, allows overriding the checkpoint source via environment variable, and skips gracefully (not fails) when the checkpoint is absent or memory is insufficient.

## Acceptance Criteria

- [ ] `ToolCallFormat.infer` and `ReasoningConfig.infer` return the verified format/config for model_type `minimax_m3_vl` (unit tests pin both)
- [ ] The existing `VLMModelFactory._load` reasoning inference (landed in commit 5891f01, task ^05zt40g) is VERIFIED to resolve `minimax_m3_vl`'s row into the VLM-built `ModelConfiguration.reasoningConfig` (unit test pins it; regression: a VLM model with no matching infer row still loads with `reasoningConfig == nil`)
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

- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax #minimax-m3