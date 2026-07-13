---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxerdejj4tfhdtnh28xs9sq9
  text: |-
    Implementation complete, verified against the real cached model, adversarial double-check in progress.

    Discriminator found: `mlx-community/GLM-4-9B-0414-4bit`'s actual cached config.json has `model_type: "glm4"` (exact). Confirmed via web research that GLM-4.7-descended checkpoints (e.g. mlx-community/glm-4.7-flash-abliterated-8bit) use `model_type: "glm4_moe"` (MoE architecture) -- a real, load-bearing architectural discriminator (dense vs MoE), available at format-detection time via config.json, matching acceptance criterion path (a).

    Format/parser: added `ToolCallFormat.glm4Bare` ("glm4_bare") + `GLM4BareToolCallParser` (Libraries/MLXLMCommon/Tool/Parsers/GLM4BareToolCallParser.swift). `infer(from:)` now returns `.glm4Bare` for exact `model_type == "glm4"`, keeps `.glm4` (GLM-4.7 envelope, unchanged) for `hasPrefix("glm4")` variants (glm4_moe, glm4_moe_lite, glm4_5, etc.).

    Non-obvious architecture addition: the existing `ToolCallProcessor` inline-format (`startTag == nil`) path assumes the JSON envelope embeds the function name (true for its only prior consumer, Llama3ToolCallParser) and flushes any text before the first `{` as plain chat text immediately -- which would permanently lose GLM-4-9B-0414's bare function-name line before the parser ever saw it, since name and JSON typically arrive in separate streamed chunks. Added a new `ToolCallParser.buffersEntireResponse` protocol property (default false, zero behavior change for all existing parsers) that GLM4BareToolCallParser sets true; `ToolCallProcessor` routes such parsers through a new `processFullBufferChunk` that silently accumulates everything and defers all parsing to `processEOS`. Tradeoff: models using `.glm4Bare` get no incremental token streaming (single flush at EOS) since there's no reliable mid-stream marker for "is this the start of a tool call" in a bare, tag-free format -- verified this doesn't break plain chat via the real-model CoherenceIntegrationTests.glm4_9B test (planets question, non-tool).

    Also updated `Libraries/MLXLLM/LLMModelFactory.swift`'s `glm4_9b_4bit` registration from hardcoded `toolCallFormat: .glm4` to `.glm4Bare` -- necessary because that hardcoded value bypasses `infer()` entirely (`LLMModelFactory._load` only calls `infer()` when `toolCallFormat == nil`), so the format-detection fix alone wouldn't have taken effect for the real registered model.

    Unit tests (Tests/MLXLMTests/ToolTests.swift): 6 new tests for the parser + processor streaming behavior, updated `testToolCallFormatRawValues` and `testToolCallFormatInference` (exact-match vs prefix-match split). `swift test --filter ToolTests`: 63/63 pass. Full `swift test`: 533 tests, 0 failures.

    Real-model verification (xcodebuild build-for-testing + test-without-building against actual cached mlx-community/GLM-4-9B-0414-4bit, not mocked):
    - ToolCallIntegrationTests/glm4FormatAutoDetection(): PASS (.glm4Bare)
    - ToolCallIntegrationTests/glm4EndToEnd(): PASS -- real model output correctly parsed into ToolCall(get_weather, location=Paris, unit=celsius), matching the task's exact reproduction.
    - CoherenceIntegrationTests/glm4_9B() (plain chat, no tools): PASS -- confirms buffersEntireResponse tradeoff doesn't break ordinary generation for this model.

    `.disabled` removed from glm4EndToEnd in IntegrationTesting/IntegrationTestingTests/ToolCallIntegrationTests.swift.

    Adversarial double-check agent launched to scrutinize the buffersEntireResponse architecture change, parser edge cases, discriminator soundness, and scope. Will report back once it returns.
  timestamp: 2026-07-13T22:08:58.450044+00:00
- actor: claude-code
  id: 01kxerkfncytmssn9ys4hbzywr
  text: |-
    Adversarial double-check: PASS, no findings. Independently re-ran `swift test --filter ToolTests` (63/63, matched claim), traced the real production call path in Evaluate.swift's TextToolTokenLoopHandler.onGenerationEnd to confirm processEOS is genuinely exercised (not just test-only), confirmed every existing parser inherits buffersEntireResponse=false via the protocol extension default (verified this branch is dead code for all of them, zero behavior change), verified the config.json discriminator directly against the cached snapshot file (not fabricated), confirmed the LLMModelFactory.swift one-line edit was necessary (traced _load's `if toolCallFormat == nil` guard), and confirmed the .disabled trait removal is clean with no orphaned comments.

    Task is green and ready for review. Leaving in doing per the implement workflow.
  timestamp: 2026-07-13T22:12:16.172419+00:00
position_column: doing
position_ordinal: '80'
title: 'Tool-calling: GLM-4-9B-0414 (older GLM-4, non-4.7) needs a different tool-call parser than GLM4ToolCallParser'
---
## What
Root-caused while investigating kanban 68bc1rt (`ToolCallIntegrationTests.glm4EndToEnd` -- "Expected at least one tool call, got none").

Format auto-detection in `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift` routes any `model_type` with `type.hasPrefix("glm4")` to `.glm4`, which uses `GLM4ToolCallParser` (`func<arg_key>k</arg_key><arg_value>v</arg_value>` XML-ish envelope, wrapped in `<tool_call>...</tool_call>`). That parser's doc comment cites `mlx_lm/tool_parsers/glm47.py` -- it's modeled on **GLM-4.7**'s tool-calling convention.

The integration test target is `mlx-community/GLM-4-9B-0414-4bit`, an older GLM-4 checkpoint (April release, predates 4.7). Its own official `tokenizer_config.json` chat template renders tools like this (no XML envelope at all):
```
# 可用工具
{% for tool in tools %}...
## {{ function.name }}
{{ function | tojson(indent=4) }}
在调用上述函数时，请使用 Json 格式表示调用的参数。{% endfor %}
```
(Translation: "Available tools" / function name+full JSON schema / "When calling the above function, please use JSON format to represent the parameters of the call.") There is no `<tool_call>`, no `<arg_key>`/`<arg_value>` instruction anywhere -- GLM-4-9B-0414 was never taught the GLM-4.7 envelope.

Reproduced directly: given the weather-tool schema and "What's the weather in Paris?", the model's raw completion is:
```
get_weather
{"location": "Paris", "unit": "celsius"}
```
-- the correct function name and correct, complete, valid arguments, but as **plain text** (function name on its own line, then a bare JSON object of just the arguments -- no `name`/`arguments` JSON envelope, no tags of any kind). This doesn't match ANY existing parser:
- `GLM4ToolCallParser` -- expects `<tool_call>`/`<arg_key>` (absent entirely)
- `JSONToolCallParser` -- expects the JSON blob itself to contain `"name"`/`"arguments"` keys (this JSON is just the bare arguments, with the name as separate preceding plain text)
- `XMLFunctionParser`, `PythonicToolCallParser`, `MistralToolCallParser` -- don't match either

This is a genuine format/detection mismatch (fixable), but a full fix requires either (a) a new `ToolCallParser` for this "bare function-name line + bare JSON-args object" shape, plus a new `ToolCallFormat` case and a detection rule that distinguishes this older/bare GLM-4 `model_type` from GLM-4.7's variants, or (b) confirming there's no reliable discriminator in `config.json` between the two and deciding on a different mitigation (e.g. prompting the model with an explicit envelope instruction via `additionalContext`, if the chat template supports parameterizing that). Either path is a real, scoped feature addition -- not a one-line bug fix -- so it wasn't done under 68bc1rt (whose scope was the 4 named failing tests + checkpoint-guard hygiene, not new parser development). `ToolCallIntegrationTests.glm4EndToEnd` was `.disabled` there with this exact finding.

## Acceptance Criteria
- [ ] Decide: new parser for "bare name + bare JSON args" vs. template-side mitigation vs. something else
- [ ] Implement + unit tests for the chosen approach in `Libraries/MLXLMCommon/Tool`
- [ ] `ToolCallIntegrationTests.glm4EndToEnd` (currently `.disabled`) re-enabled and passing
- [ ] Confirm `glm4FormatAutoDetection` (which currently just asserts `.glm4`) is updated if the new format needs its own `ToolCallFormat` case
