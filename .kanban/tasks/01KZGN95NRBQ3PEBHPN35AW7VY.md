---
assignees:
- claude-code
depends_on:
- 01KZGMWJPM4016GPFFVGBSAQC2
position_column: todo
position_ordinal: '8e80'
title: Port DeepseekV4 DSML tool-call encoding and tool_result merging
---
## What

Extend `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` with the tool-calling half of DeepSeek-V4's prompt format — the **write** side. The matching **read** side (a DSML response parser) lives in the registry-wiring task `mjrzkgm`.

Split out from `gbsaqc2` because the combined reference encoder is ~1200+ lines.

Port the tool-related portion of `scouzi1966/mlx-swift-lm` @ `main`, `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`.

In scope:
- **DSML tool-call encoding** — DeepSeek's native tool format (not JSON, not XML-function). Upstream `ml-explore/mlx-lm` PR 1337 adds an equivalent `deepseek_dsml` parser; read it as a cross-check on the format.
- Tool **definitions** rendered into the prompt.
- `tool_result` merging into user `contentBlocks`.
- `sort_tool_results_by_call_order` — results are emitted in the order the calls were made, not the order they returned.
- `drop_earlier_reasoning` **forced off when tools are present** — the interaction hook left by `gbsaqc2`. This is easy to get backwards and silently degrades multi-turn tool use.

## Provenance
- Reference: `scouzi1966/mlx-swift-lm` @ `main` — `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift`, tool portions (MIT; header attributes Osaurus AI).
- Format cross-check: `ml-explore/mlx-lm` PR 1337 (`deepseek_dsml` tool parser).
- Original Python source of truth: DeepSeek's `encoding/encoding_dsv4.py`.
- Apply the attribution header decided in task `jhk0apk`.

## Acceptance Criteria

- [ ] Tool definitions and tool calls render in DSML, byte-identical to the Python reference on fixtures.
- [ ] `tool_result` blocks merge into the user turn.
- [ ] Results sort by original call order, verified with a fixture where return order differs from call order.
- [ ] With tools present, earlier reasoning is **retained** even when `drop_earlier_reasoning` is set; with tools absent it is dropped.
- [ ] Existing `gbsaqc2` tests still pass — no regression to non-tool rendering.

## Tests

- [ ] Extend `Tests/MLXLMTests/DeepseekV4ChatEncoderTests.swift` (or add `DeepseekV4ToolEncodingTests.swift`) with Python-generated fixtures.
- [ ] Test: a single tool call renders byte-identically.
- [ ] Test: two tool definitions plus a call render byte-identically.
- [ ] Test: results returned in order [B, A] for calls made in order [A, B] render in order [A, B]. This is the sort guard.
- [ ] Test: tools present plus `drop_earlier_reasoning` set implies earlier reasoning retained; tools absent implies dropped. Both directions asserted.
- [ ] Run: `swift test --filter 'DeepseekV4ChatEncoderTests|DeepseekV4ToolEncodingTests'` — all pass.

## Workflow
- Use `/tdd` — write the call-order and drop-earlier-reasoning tests first; both catch silent logic inversions.
#deepseek-v4