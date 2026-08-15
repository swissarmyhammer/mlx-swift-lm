---
comments:
- actor: claude-code
  id: 01ky60866n8rs5j354v22jdjbg
  text: 'Closed as duplicate in chain reconciliation (user decision 2026-07-22): superseded by ^ayw1xee; unique content (verified tool-call format, <mm:think> reasoning tags, chat-template probe) folded there.'
  timestamp: 2026-07-22T22:48:26.581136+00:00
depends_on:
- 01KXX99P1H2Z3DV0TM0AGBFEBR
position_column: done
position_ordinal: cf80
title: 'MiniMax M3: tool-call parsing and reasoning-tag integration'
---
#minimax-m3

## What
Wire MiniMax M3 into the tool-calling, reasoning, and FoundationModels plumbing. M3's formats are NOT M2's — verified against the checkpoint's `chat_template.jinja` and upstream mlx-lm PR #1416 (which adds a separate `minimax_m3.py` parser):

- **Tool calls**: namespaced XML with parameters as arbitrary `<key>value</key>` children — unlike M2's `<parameter name="k">v</parameter>`. The M2 parser cannot be reused. Add `Libraries/MLXLMCommon/Tool/Parsers/MiniMaxM3ToolCallParser.swift` following the M2 parser + `ParserUtilities.swift` conventions, with fixture strings taken verbatim from the chat template / PR #1416 test cases.
- **New `ToolCallFormat` case** (e.g. `.minimaxM3`) in `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`, registered for the `minimax_m3` / `minimax_m3_vl` model types.
- **MLXFoundationModels**: add the new case to `structuredToolCallFormats` in `Libraries/MLXFoundationModels/TranscriptConverter.swift:218` (currently `[.mistral, .minimaxM2]`) so M3 tool turns replay through the chat template's `tool_calls` branch.
- **Reasoning tags**: the template uses toggleable `<mm:think>` — not M2's always-on `<think>` (`ReasoningConfig.swift:201-208`). Add a new `ReasoningConfig` entry for the M3 model types with the `<mm:think>` delimiters and the correct toggle semantics as expressed by the template.
- **Chat-template probe**: the M3 template uses macros and namespace-token concatenation that may stress swift-transformers' Jinja engine. Add an `ApplyChatTemplateProbeTests`-style test that renders the real M3 chat template (checked-in fixture) with a system+user+tool conversation — catching engine gaps here instead of during the 214 GB integration test.
- Keep all M2 behavior unchanged.

## Acceptance Criteria
- [ ] Captured M3-format tool-call strings (verbatim from template/PR #1416) parse into expected tool name + arguments, including multi-parameter and nested-value cases
- [ ] Streaming/partial parse behavior matches the parser contract used by the M2 parser tests
- [ ] `<mm:think>` reasoning segmentation works for M3 model types per `ReasoningConfig` semantics
- [ ] `TranscriptConverter` round-trips an M3 tool-call turn (structured format includes `.minimaxM3`)
- [ ] The real M3 chat template renders through swift-transformers without error for a tool-bearing conversation
- [ ] M2 parsing and reasoning tests still pass unchanged

## Tests
- [ ] Extend `Tests/MLXLMTests/ToolTests.swift`, `Tests/MLXLMTests/ReasoningConfigTests.swift`, and `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift` with M3 cases; add the chat-template probe test
- [ ] Run `swift test --filter 'ToolTests|ReasoningConfig|TranscriptConverter'`; expect pass

## Workflow
- Use `/tdd` — write failing tests first, then implement to make them pass. #minimax