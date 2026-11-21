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
- actor: claude-code
  id: 01m7zz8es7m4ewz3tkzarrcm7j
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (e047b956) — 16 confirmed, 6 refuted. IntegrationTesting/IntegrationTestingTests/Inkling2ToolLoopIntegrationTests.swift:122,144,204; IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift:130; Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:113,118,119,120,122,129,133,140; Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift:24,71,110; Tests/MLXLMTests/InklingCommittedTurnRuleTests.swift:77. No confirmed finding on library (non-test) code, including the committed-turn splice and the non-Inkling formats.
    - next: resolve the 16 items in the 2026-11-21 15:04 Review Findings section, then re-run /review ^1sg8f5t.
  timestamp: 2026-11-21T21:12:38.439247+00:00
- actor: claude-code
  id: 01m7zz903jhmgqhbhbdd8x2nwr
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 17 files
    - test: green — 5 bundles; Inkling integration 7/7
    - commit: e047b956
    - review: findings — 16 in test code (Inkling2ToolLoopIntegrationTests:122,144,204; Inkling2FoundationModelsToolLoopTests:130; InklingFrameTokenizer:113-140; Inkling2ToolLoopSessionTests:24,71,110; InklingCommittedTurnRuleTests:77)
  timestamp: 2026-11-21T21:12:56.178617+00:00
- actor: claude-code
  id: 01m8000ybnmca0ex4h3y8vp45r
  text: |-
    Resolved the 16 findings of the 2026-11-21 15:04 review (test code only, no library change):
    - @unchecked Sendable gone from both files: DispatchLog (now a Sendable struct) and Inkling2ToolCallLog (Sendable final class) hold an OSAllocatedUnfairLock, the prevailing test pattern (PrefillProgressLog, ChatSessionReasoningHistoryTests).
    - Magic numbers in Inkling2ToolLoopIntegrationTests: Inkling2ToolCallLog.twoStepCallCount; the suite time limit is the file-level suiteTimeLimitMinutes (as in the DeepseekV4 and KVCacheSession suites).
    - print -> os Logger (subsystem com.apple.FoundationModels-MLX, categories Inkling2ToolLoop / Inkling2FoundationModelsToolLoop, privacy .public), as the other integration suites log. Result lines now go to the unified log, not stdout.
    - InklingFrameTokenizer.renderedCall is private and throws: a guard on JSONSerialization.isValidJSONObject throws ArgumentsAreNotJSON (JSONSerialization raises an NSException for a non-JSON value, which try? never caught anyway), and render/assistantFrames propagate through applyChatTemplate. New InklingFrameTokenizerTests covers it. Docs on bosToken/eosToken/unknownToken/encode/decode/convertTokenToId/convertIdToToken.
    - New shared fixture Tests/MLXLMTestSupport/InklingWeatherToolLoop.swift: toolName, toolDescription, cityArgument, weather, toolSpec, writtenCall, renderedCall, tokenizer construction, thinking/call/result frames, callingTurn, answeringTurn. Both Inkling2ToolLoopSessionTests and InklingExecutorToolLoopTests use it; each keeps its own question/reasoning/answer and assertions. renderedCall in the fixture is the literal compact sorted-key text, so the tokenizer's render is checked against an independent statement of the wire format (a mismatch renders the unknown token and fails the suites).
    - alignedCache(_:previousRender:) -> PromptCacheState.aligned(_:previousRender:) in Tests/MLXLMTests/PromptCacheState+Aligned.swift, used by CommittedTurnThinkBlockTests and InklingCommittedTurnRuleTests; both local copies deleted. The other alignedCache helpers (Qwen, DSML, Harmony, PromptCacheReusePolicy, CommittedTurnUncommittedToken) have different signatures and pre-date this card, left alone.
    Note: Inkling2CheckpointIntegrationTests.swift still has a print (report helper); it was not in the findings.
  timestamp: 2026-11-21T21:26:00.821381+00:00
- actor: claude-code
  id: 01m80013emvecf3cyd6x205pp9
  text: |-
    ### implement — changed
    - evidence: swift build --build-tests: Build complete, 0 errors, 0 repo warnings. xctest MLXLMTests: 1258 XCTest + 2540 Swift Testing passed (3 pre-existing known issues), 0 failures. MLXFoundationModelsTests: 562 passed. IntegrationTesting build-for-testing Release (ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES): TEST BUILD SUCCEEDED; test-without-building Inkling2ToolLoopIntegrationTests + Inkling2FoundationModelsToolLoopTests: 3 tests in 2 suites passed. swift-format applied. Files (12): IntegrationTesting/IntegrationTestingTests/Inkling2ToolLoopIntegrationTests.swift, IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift, Tests/MLXLMTestSupport/InklingFrameTokenizer.swift, Tests/MLXLMTestSupport/InklingWeatherToolLoop.swift (new), Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift, Tests/MLXLMTests/InklingCommittedTurnRuleTests.swift, Tests/MLXLMTests/CommittedTurnThinkBlockTests.swift, Tests/MLXLMTests/PromptCacheState+Aligned.swift (new), Tests/MLXLMTests/InklingFrameTokenizerTests.swift (new), Tests/MLXFoundationModelsTests/InklingExecutorToolLoopTests.swift. All 16 findings checked. Not committed.
    - next: /review ^1sg8f5t
  timestamp: 2026-11-21T21:26:06.036083+00:00
- actor: claude-code
  id: 01m8006160kgt1phtqbva77gky
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (5ba4a62b) — 1 finding (confirmed 1, refuted 2, attempted 7): IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift:135 (reuse/reuse). All 16 items of the 15:04 section are checked.
    - next: implement the 2026-11-21 15:26 finding (share the Logger and report helper in the IntegrationTesting target), commit, re-review.
  timestamp: 2026-11-21T21:28:47.552877+00:00
depends_on:
- 01M7ZVJJNB56PTH6BP74Y1ZBGN
position_column: review
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

## Review Findings (2026-11-21 15:04)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 17 file(s) reviewed, 7 not reviewed.

> 7 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7ZVJJNB56PTH6BP74Y1ZBGN.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7ZVJJNB56PTH6BP74Y1ZBGN.md` — no validator matches this file
> - `.kanban/tasks/01M7ZVJS05JRXMZZXBS1SG8F5T.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7ZVJS05JRXMZZXBS1SG8F5T.md` — no validator matches this file
> - `.kanban/tasks/01M7ZY1J67SCA1EF7WN8AFGJAJ.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7ZY1J67SCA1EF7WN8AFGJAJ.md` — no validator matches this file
> - `Libraries/MLXVLM/README.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/Inkling2ToolLoopIntegrationTests.swift:122` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.
- [x] `IntegrationTesting/IntegrationTestingTests/Inkling2ToolLoopIntegrationTests.swift:144` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/Inkling2ToolLoopIntegrationTests.swift:204` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift:130` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:113` `swift/preconditions` — `renderedCall` swallows a JSON serialization failure with `try?` and substitutes `"{}"`. A failure to serialize the test's arguments is an unexpected condition, and the silent fallback hides the defect from both the test author and the log, so a broken call renders as an empty argument object without any trace. Make the failure visible. Either make `renderedCall` `throws` and propagate the error, or validate the arguments first with a `guard` and call `assertionFailure` plus a logged message before returning the `{}` fallback, so the default is recorded and not silent.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:118` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'bosToken'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:119` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'eosToken'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:120` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'unknownToken'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:122` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'encode(text:addSpecialTokens:)'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:129` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'decode(tokenIds:skipSpecialTokens:)'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:133` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'convertTokenToId(_:)'.
- [x] `Tests/MLXLMTestSupport/InklingFrameTokenizer.swift:140` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for 'convertIdToToken(_:)'.
- [x] `Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift:24` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.
- [x] `Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift:71` `reuse/reuse` — The `writtenCall` fixture, with the `weather` table, the `weatherTool` spec and the `tokenizer` built from them, duplicates the same fixture in `InklingExecutorToolLoopTests`. Both suites now define the model's written-call text, the same weather table and the same tool declaration independently. The two copies must stay in step by hand, and a change to one that is not made in the other makes the two suites test different wire formats. Move `writtenCall`, the `weather` table, `toolName`, and the tokenizer construction next to `InklingFrameTokenizer` in `Tests/MLXLMTestSupport`, which both bundles already import. Call that shared fixture from both suites. Keep each suite's own assertions local.
- [x] `Tests/MLXLMTests/Inkling2ToolLoopSessionTests.swift:110` `reuse/reuse` — `callingTurn` and `answeringTurn` rebuild the model's scripted turn frame by frame, duplicating the scripted-turn builders in `InklingExecutorToolLoopTests`. Both build the thinking frame, the calls and `<|content_model_end_sampling|>` with the same marker sequence. The duplicate hard-codes the Inkling frame order once more. Put the calling and answering turn builders in the shared Inkling test-support file together with the fixture from the previous finding, and have both suites call them. This removes the second copy of the frame ordering.
- [x] `Tests/MLXLMTests/InklingCommittedTurnRuleTests.swift:77` `reuse/reuse` — The new `alignedCache(_:previousRender:)` helper reimplements an existing helper with the same signature and the same body. Each copy builds a `PromptCacheState` with `mainCacheIsAligned: true` and `processedTokenCount` equal to the cache count. A later change to the aligned-cache shape has to be made in several files, and this copy can drift from the others. Use the existing helper instead of a fourth copy. Move one `alignedCache(_:previousRender:)` to a shared internal test-support location (for example beside `MarkerTokenizer` in MLXLMTests) and call it from both `CommittedTurnThinkBlockTests` and `InklingCommittedTurnRuleTests`. Delete the local copy.

## Review Findings (2026-11-21 15:26)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 10 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M7ZVJS05JRXMZZXBS1SG8F5T.jsonl` — no validator matches this file
> - `.kanban/tasks/01M7ZVJS05JRXMZZXBS1SG8F5T.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/Inkling2FoundationModelsToolLoopTests.swift:135` `reuse/reuse` — The new `report` helper and its `resultLog` Logger in this file reimplement the same logging helper that already exists in `Inkling2ToolLoopIntegrationTests.swift`. Only the category string and the tag prefix differ, so the logging setup is duplicated in the same test target and will drift. Move the Logger and a `report(_ prefix: String, _ line: String)` helper into one shared file in the IntegrationTesting target (for example, next to `Inkling2BankFixture`) and call it from both suites, passing the tag prefix as a parameter.
