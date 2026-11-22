---
assignees:
- claude-code
position_column: done
position_ordinal: '8380'
title: 'Inkling 2: decide the FoundationModels required tool mode (grammar + think-then-call) for the frame protocol'
---
Found while working ^1sg8f5t. `MLXLanguageModel` `.required` tool mode (`runRequiredToolTurn`) constrains output with the `<tool_call>` JSON structural-tag grammar and, for a `.templateFlag` reasoning protocol with reasoning declared, runs a think-then-call phase that reasons until the protocol end delimiter. Inkling 2 writes tool calls in `<|message_model|>NAME<|content_invoke_tool_json|>{...}<|end_message|>` frames, so the grammar shape does not match the model's protocol. Since ^1sg8f5t, `InklingReasoningProtocol` is `.templateFlag(reasoning_effort, thinkingOff: .templateValue("none"))`, so the think-then-call phase now applies to Inkling in required mode (before, `.none` skipped it). Router/ACPAgent use `.allowed`, which works end to end.

- [x] Measure required mode on mlx-community/Inkling-2-Mini-4bit (IntegrationTesting) with a tool loop.
- [x] Either give the frame protocols (gptOSS/atem/inkling) a required-mode path that matches their frames, or reject required mode for them with a typed error, with unit tests.

#inkling

## Resolution

Commits 39b879f4, 8aad7adb, dc8ffe42 (branch inkling2-side).

Measured with `Inkling2FoundationModelsToolLoopTests.requiredModeCallsEachToolOfTheLoop(thinking:)` (new): the two tool rounds in `.required`, then the answer and the follow-up in `.allowed`, on mlx-community/Inkling-2-Mini-4bit (Release, under the GPU lock).

Before (HEAD 7a1fe592, `<tool_call>` grammar):
- thinking: both calls right, but `<|message_model|>` streamed as response text in each required round (the think phase let the model open its own frame).
- effort none: round 1 wrote the model's frames inside the JSON envelope -> `find_customer({"name":"<|message_model|><|content_thinking|>Need get balance.<|end_message|>...` ; the answer round generated 0 tokens; the follow-up reused 0 tokens.

Decision: give Inkling its own frame (feasible; xgrammar treats the literal special-token strings as ordinary vocabulary), refuse Harmony and ATEM.
- `RequiredToolCallFrame` picks the frame from the `ToolCallFormat`: plain syntaxes keep the `<tool_call>` envelope; `.inkling` constrains to `<|message_model|>NAME<|content_invoke_tool_json|>{"name": "NAME", "args": ARGS}<|end_message|>` (`SchemaConverter.encodeInklingToolCallingGrammar`) after a think-then-call phase inside a `<|message_model|><|content_thinking|>` frame that the turn opens, at every effort (with effort none and the tool frame forced at once, the model called `get_balance(C-1000)` — the example id of the parameter description — before `find_customer`); `.gptOSS` / `.atem` throw `LanguageModelError.unsupportedCapability(.toolCalling)` naming the protocol, before generating (no checkpoint here to measure a Harmony/ATEM required frame).
- `emitRequiredToolCallEvent` parses the call of the turn's frame.

After (dc8ffe42):
- thinking: find_customer(Ada Lovelace) 32 tok, get_balance(C-7731) 6 tok, no text; answer "Ada Lovelace's account balance is 4,812.55 EUR."; follow-up reused 529 of 546 tokens, "C-7731".
- effort none: find_customer 22 tok (reasoning 79 chars), get_balance 6 tok; same answer; follow-up reused 517 of 534 tokens.
- A required round runs the guided loop, which carries no prompt cache (unchanged), so the answer round after it starts cold (0 cached); the follow-up reuses the answer round's cache.
- Allowed mode unchanged: `languageModelSessionCompletesATwoStepToolLoop` and `executorReusesThePromptCacheAfterEachToolResult` pass (rounds after a tool result reuse 414/476 and 402/457 tokens).
- Unit tests: `RequiredToolCallFrameTests` (selection, grammar, parsing) and scripted Inkling required passes with and without declared reasoning, plus the Harmony/ATEM refusal, in `InklingExecutorToolLoopTests`.
- All 5 bundles green (MLXLMTests 1258 XCTest + 2573 Swift Testing, MLXGuidedGenerationTests 86, MLXFoundationModelsTests 583, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11).
- Review: duplication and a test `guard` fixed in 8aad7adb; the later passes are clean.
