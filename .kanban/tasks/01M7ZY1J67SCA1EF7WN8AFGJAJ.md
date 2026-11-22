---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m81zssvr36m646cg14krf98f
  text: |-
    ### review — findings
    - evidence: main-checkout re-review. `review file Libraries/MLXFoundationModels/ToolCalling/RequiredToolCallFrame.swift`: 0 findings. `review file Tests/MLXFoundationModelsTests/RequiredToolCallFrameTests.swift`: 1 finding (`RequiredToolCallFrameTests.swift:13`). `review sha 2dafc7c4..inkling2-side`: 3 findings, 1 on this card (`Inkling2FoundationModelsToolLoopTests.swift:416`).
    - next: fix the 2 open items, then re-review.
  timestamp: 2026-11-22T16:00:35.704590+00:00
position_column: review
position_ordinal: '80'
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

## Review Findings (2026-11-22 09:40)

> Scope: `review file Libraries/MLXFoundationModels/ToolCalling/RequiredToolCallFrame.swift` — reviewed the whole of each named file. 1 file(s) reviewed, 0 not reviewed.

## Review Findings (2026-11-22 09:41)

> Scope: `review file Tests/MLXFoundationModelsTests/RequiredToolCallFrameTests.swift` — reviewed the whole of each named file. 1 file(s) reviewed, 0 not reviewed.

- [ ] `Tests/MLXFoundationModelsTests/RequiredToolCallFrameTests.swift:13` `completeness/inverse-operation-coverage` — The suite's doc comment says each frame "reads back the call its grammar constrains", which promises a grammar-to-reader round trip for the Inkling 2 frame. The body never makes that trip. `inklingGrammarWritesTheToolFrame` (line 57) checks only the grammar's structural text, and `inklingFrameReadsItsCall` (line 89) reads a hand-written literal, `inklingCall`. No test generates output shaped by `inklingFrame.grammar(tools:)` and feeds it to `inklingFrame.call(in:)`. If the grammar's begin/end tags drift from the reader's expected envelope, both tests still pass. Add one round-trip test. Build a call string for a tool from the grammar's begin/end tags (for example `begin + #"{"city": "Rome"}"# + end`, or the `inklingCall` shape), feed it to `RequiredToolCallFrame.inklingFrame.call(in:)`, and assert the name and arguments come back. Or narrow the doc comment at line 13 so it does not claim a round trip the tests do not make.

## Review Findings (2026-11-22 09:53)

> Scope: `review sha 2dafc7c4..inkling2-side` — reviewed the diffs only — lines this change added or modified. 20 file(s) reviewed, 6 not reviewed. (Items of this card's commits only; the `ModelConfiguration.swift` items of this pass are on ^4z5df0d.)

- [ ] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift:416` `duplication/duplication` — `expectRequiredModeCompletesTheLoop` re-implements the executor tool-loop scaffold that `expectTheExecutorReusesThePromptCache` already runs: the same `.instructions` plus `Self.prompt(...)` entry setup, the same round loop with `runRound` and `Self.entries(after:...)`, the same answer-then-`followUp` append, and the same `expectTheLoopAnswered` tail. The two copies differ only in the tool-calling mode (`.required` vs `.allowed`) and in the per-round assertions. Copies of this scaffold drift apart, so a fix to one loop (for example to the follow-up or cache handling) will miss the other. Extract one shared helper that takes the `toolCallingMode` (and any per-round assertion hook) and runs the entries, round loop, answer, and follow-up. Call it from both `expectTheExecutorReusesThePromptCache` and `expectRequiredModeCompletesTheLoop`, passing `.required` for the required path and `.allowed` for the default path. Keep the required-only `#expect(round.text.isEmpty)` and `#require(round.toolCalls.count == 1)` checks in the required caller.
