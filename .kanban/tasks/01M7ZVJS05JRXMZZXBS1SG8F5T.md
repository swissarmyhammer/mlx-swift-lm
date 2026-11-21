---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7zwqna5bgnysnmy8ftzx489
  text: |-
    Research (picked up):
    - Inkling template (snapshot chat_template.jinja): no generation prompt; assistant turn = [<|message_model|><|content_thinking|>R<|end_message|>] [<|message_model|><|content_text|>T<|end_message|>] [<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{sorted}}<|end_message|>]* <|content_model_end_sampling|>; tool result = <|message_tool|>NAME<|content_text|>R<|end_message|> (name from message.name or tool_call_id lookup). reasoning_content is ALWAYS rendered when given. Thinking off = reasoning_effort "none" (0); `true` as value would raise (true|float = 1.0 > 0.99).
    - ToolCallFormat.inkling.promptCacheReuseRules returns [] — Qwen/DSML have a CommittedTurnSplice rule, Harmony/Onyx a restart rule. Inkling is hybrid recurrent (cannot trim), and the re-render of a tool call (sorted, compact JSON) need not match the tokens the model wrote, so a committed-turn splice on <|content_model_end_sampling|> (think block <|content_thinking|> / <|end_message|>) is the missing piece. Both ChatSession and ExecutorPromptCache read these rules.
    - InklingReasoningProtocol: promptStrategy .none => ChatSession.thinkingEnabled=false and FM sessions without .reasoning throw cannotDisableReasoning. replaysReasoningIntoHistory=false => TranscriptConverter drops reasoning while ChatSession keeps finished reasoning (inconsistent renders between paths).
    - FM allowed tool path already uses format.makeProtocolTokenStreamDecoder, so the Inkling frame decoder serves it.
    - HF cache has only the pinned snapshot, no refs/main; FM integration test needs its own weightsLocation.
  timestamp: 2026-11-21T20:28:30.917453+00:00
- actor: claude-code
  id: 01m7zxagqb1y1a1kk5v092p8ew
  text: |-
    Unit layer landed (RED then GREEN, MLXLMTests + MLXFoundationModelsTests filtered runs):
    - New InklingCommittedTurnRule (CommittedTurnSplice at first <|content_model_end_sampling|>, think block <|content_thinking|>/<|end_message|>), registered in ToolCallFormat.inkling.promptCacheReuseRules. RED showed ChatSession trimming back to the tool-call token and re-feeding (offset 18 instead of 21), and the FM executor on a hybrid model falling back to the checkpoint at the previous prompt end (reused 15 of 25).
    - ThinkingOffRender.templateValue(String): off => [key: value], on => nil (template default / caller level). InklingReasoningProtocol now .templateFlag(key: reasoning_effort, defaultOn: true, thinkingOff: .templateValue("none")) + replaysReasoningIntoHistory. RED: ChatSession.thinkingEnabled=false threw cannotDisableReasoning; FM without .reasoning threw "This model always reasons"; FM history dropped the reasoning frame.
    - Test double: Tests/MLXLMTestSupport/InklingFrameTokenizer.swift (Inkling frame template, renders past calls as compact sorted JSON like the real template).
    - Filtering Swift Testing under xctest works: `xcrun xctest -XCTest "MLXLMTests.SuiteName,..." bundle`.
    Discovered: with .templateFlag the FM required-mode think-then-call phase now applies to Inkling (out of scope; Router uses .allowed).
  timestamp: 2026-11-21T20:38:48.811191+00:00
- actor: claude-code
  id: 01m7zyqajff57163xc3rj5qbzd
  text: |-
    Real checkpoint found one more prefix break: Inkling2Processor rendered history through Qwen2VLMessageGenerator, which never writes reasoning_content, so on BOTH paths the history render dropped every past thinking frame (the scripted tokenizer used DefaultMessageGenerator and could not see it). Fixed with Inkling2MessageGenerator (Qwen2VL content layout + addReasoningMetadata), RED/GREEN in Inkling2ProcessorTests. After the fix ChatSession pass-1 suffixStart moved 290 -> 340 (render now matches the 255+84-token ledger) while still feeding only the 19-token tool result.

    Real-model results (Release, Inkling-2-Mini-4bit, serial run, all 7 Inkling tests pass):
    - ChatSession thinking: pass0 find_customer(Ada Lovelace) prompt 255; pass1 fed 19 (cached 340) -> get_balance(C-7731); pass2 fed 32 (cached 402) -> "Ada Lovelace's account balance is 4,812.55 EUR."; follow-up fed 17 (cached 467) -> "C-7731". Decode 82-84 tok/s.
    - ChatSession no-thinking (effort none): same calls/answers, cached 320/382/444; reasoning 295 vs 446 chars (model still writes a short thinking frame at effort 0).
    - Executor thinking: round1 433 tok cached 414; round2 508 cached 476; follow-up 546 cached 529; decode ~82.5 tok/s. No-thinking: 421/402, 489/457, 527/510.
    - LanguageModelSession with two real Tools: answer cached 476 of 508 (thinking) / 457 of 489 (no .reasoning); follow-up 539/556, 519/536.
    Thinking-off: reasoning_effort "none" lowers but does not remove thinking on Mini; frames still stream as reasoning, never text.
    Follow-up task ^8afgjaj (FM required tool mode for frame protocols).
  timestamp: 2026-11-21T21:03:17.071672+00:00
- actor: claude-code
  id: 01m7zyqftwzp35f2t3c25bmcev
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests clean; xctest MLXLMTests 1258 XCTest + 2539 Swift Testing passed (3 pre-existing known issues), MLXFoundationModelsTests 562 passed, MLXGuidedGenerationTests 86, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11 — 0 failures, 0 skipped. IntegrationTesting Release (serial): Inkling2CheckpointIntegrationTests 4, Inkling2ToolLoopIntegrationTests 1, Inkling2FoundationModelsToolLoopTests 2 (3 cases) passed. Files: Libraries/MLXLMCommon/{InklingReasoningProtocol,ReasoningConfig}.swift, Tool/ToolCallFormat.swift, Tool/Parsers/{InklingCommittedTurnRule (new),CommittedTurnSplice}.swift, Libraries/MLXVLM/Models/Inkling2Vision.swift, Libraries/MLXVLM/README.md, Tests/MLXLMTestSupport/InklingFrameTokenizer.swift (new), Tests/MLXLMTests/{InklingCommittedTurnRuleTests,InklingReasoningProtocolTests,Inkling2ToolLoopSessionTests (new),CommittedTurnThinkBlockTests,MarkerTokenizer,Inkling2Tests}.swift, Tests/MLXFoundationModelsTests/InklingExecutorToolLoopTests.swift (new), IntegrationTesting: Inkling2ToolLoopIntegrationTests.swift (new), MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift (new), Inkling2CheckpointIntegrationTests.swift (shared checkpoint/revision made internal).
    - next: /review (not committed)
  timestamp: 2026-11-21T21:03:22.460302+00:00
depends_on:
- 01M7ZVJJNB56PTH6BP74Y1ZBGN
position_column: doing
position_ordinal: '80'
title: 'Inkling 2 Mini: tool calling usable end to end (ChatSession and MLXFoundationModels), with cache reuse'
---
Make Inkling 2 Mini usable as a coding agent with tools. This means both of these, not just one parsed call:
- the ChatSession path;
- the MLXFoundationModels `MLXLanguageModel` path that the Router and ACPAgent sessions use.

Work on branch `inkling2-mini`, at real checkpoint `mlx-community/Inkling-2-Mini-4bit`.

- A full tool loop must work on both paths:
  - The model calls a tool.
  - The tool result goes back into the transcript in Inkling's format.
  - The model uses the result.
  - Multiple tool calls per turn and parallel calls must work, if the format allows them.
- Tool results and history must render through the Inkling chat template on both paths, including the TranscriptConverter / ExecutorPromptCache path.
- Multi-turn prompt-cache reuse must hold across tool turns:
  - The cached prompt tokens after a tool turn are > 0.
  - Find any place where re-rendering breaks the prefix, and fix it.
- Thinking (reasoning frames) must stream as reasoning, not as text, on both paths, and a thinking-off setting must work.
- Unit tests come first where it is possible with scripted models. Add real-checkpoint integration tests in IntegrationTesting with no environment gate, for:
  - a tool loop through ChatSession;
  - a tool loop through MLXLanguageModel;
  - cache reuse after a tool turn.

Acceptance:
- [x] Real Inkling 2 Mini completes a 2-step tool loop on both paths, with correct answers.
- [x] The cached prompt tokens are > 0 on the turn after a tool result, on both paths.
- [x] All 5 bundles are green, and the new integration tests pass.

#inkling