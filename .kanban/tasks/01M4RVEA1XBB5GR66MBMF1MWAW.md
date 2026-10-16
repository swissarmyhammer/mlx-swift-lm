---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m52jf1617detrj0ks4d9s2b1
  text: |-
    Research (implement step):
    - ChatSession.ReasoningHistory is nil when `replaysReasoningIntoHistory` is false, thus DeepSeek-V4 records the whole text as content. A cut thinking turn then renders `<Assistant><think></think>partial<eos>`.
    - DSMLCommittedTurnRule splices at the commit when the ledger does not hold it. After a cut turn the cache holds `<think>partial`, thus the splice feeds `<eos>` after `partial`, and the model reads `<think>partial<eos>`. The cold render with Option A is `<think>partial</think><eos>`. Thus the DSML rule must decline a cache that ends inside an open think block, as QwenCommittedTurnRule does since ^ef2zynk. Plan: move the think-block check (`ThinkBlockTokens`) from QwenCommittedTurnRule into CommittedTurnSplice, so both rules use one body.
    - `<think>` (128821) and `</think>` (128822) are single added tokens in mlx-community/DeepSeek-V4-Flash-4bit, thus the rule can resolve them.
    - Scope: the change must apply to DeepSeek-V4 only. Other non-replay configs (Qwen3 tagged, R1) use Jinja templates that can ignore `reasoning_content` for earlier turns, thus a record with empty content would lose the text there. Plan: a new ReasoningConfig flag that DeepSeekV4EncodingTokenizer.reasoningConfig sets.
    - Reference parser: `parse_message_from_completion_text` reads until `</think>` or the tool-call start and asserts `stop_token == thinking_end_token` ("Invalid thinking format: missing </think>"), then asserts the EOS. Thus the reference cannot parse a cut turn at all. The reference render of `reasoning_content=partial, content=""` is `partial</think><eos>` after `<Assistant><think>`, which is the Option A record.
    - The MLXFoundationModels path (TranscriptConverter, historyMessages) is not changed by this card.
  timestamp: 2026-10-16T14:39:40.225722+00:00
- actor: claude-code
  id: 01m52khczcefebe3mmzsf6ghaq
  text: |-
    Implementation landed (unit level):
    - New flag `ReasoningConfig.replaysUnclosedReasoningIntoHistory` (default false). DeepSeekV4EncodingTokenizer.reasoningConfig sets it. `replaysReasoningIntoHistory` stays false, thus closed turns keep the whole text as content and `holdsItsOwnReasoning` still applies to them.
    - New helper `ReasoningEventEmitter.staysInsidePrimedReasoning(_:config:primedInside:)`: true when the prompt ended inside an open block and the text holds no end delimiter. Reuses the emitter and `promptEndsInsideReasoning`.
    - ChatSession.ReasoningHistory now has `message(content:toolCalls:)`. Replay protocol: split as before. Unclosed-only protocol: a turn that stays inside the primed block is recorded as `.assistant("", reasoning: text)`; every other turn keeps the whole text.
    - The think-block check moved from QwenCommittedTurnRule into CommittedTurnSplice (`ThinkBlockTokens`, `thinkBlock`). Both rules pass their markers; DSMLCommittedTurnRule now resolves `<think>`/`</think>` and declines a cache that ends inside an open block. The standard extend rule then feeds `</think><eos>...`.
    - Tests: CommittedTurnThinkBlockTests (the ^ef2zynk table moved out of QwenCommittedTurnRuleTests, run against both rules), DeepSeekV4CutTurnSessionTests (warm turn 2 == cold prompt, cut row and closed control; answer-only history renders `<think></think>answer<eos>`), ReasoningEventEmitterTests rows, ReasoningConfigTests. Shared test helpers: PieceTokenizer, ChatSession.completionInfo(respondingTo:), ScriptedTokenModel.attentionCacheTokens.
    - RED: DSML rows "declines a turn that stopped on the budget inside its reasoning" and "...ended its turn inside its reasoning" failed; the cut session row failed (warm `<think>The user says teal<eos>`, cold `<think>The user says teal</think><eos>`). GREEN: MLXLMTests 721 XCTest + 1522 Swift Testing pass.
    - Mutant: ChatSession records the cut turn as whole content -> the cut session row fails (2 issues), all other tests pass. Code restored.
    - Note: xctest `-XCTest <name>` does not filter Swift Testing tests; the whole bundle runs (about 150 s).
  timestamp: 2026-10-16T14:58:26.412450+00:00
- actor: claude-code
  id: 01m52naw8gdg109ez8gmmrkp3f
  text: |-
    ### implement — changed
    - evidence: 18 files. Libraries/MLXLMCommon: ChatSession.swift, ReasoningConfig.swift, ReasoningEventEmitter.swift, Tokenizer.swift, DeepSeekV4ChatEncoder.swift (docs), Tool/Parsers/CommittedTurnSplice.swift, Tool/Parsers/DSMLCommittedTurnRule.swift, Tool/Parsers/QwenCommittedTurnRule.swift. Tests/MLXLMTests: new CommittedTurnThinkBlockTests.swift, DeepSeekV4CutTurnSessionTests.swift, PieceTokenizer.swift, ChatSessionCompletionInfo.swift; changed ChatSessionReasoningHistoryTests.swift, QwenCommittedTurnRuleTests.swift, ReasoningConfigTests.swift, ReasoningEventEmitterTests.swift, ScriptedTokenModel.swift, MarkerTokenizer.swift.
    - swift build --build-tests: 0 warnings (only the known "missing creator" note). swift-format lint on changed files: clean.
    - MLXLMTests: 721 XCTest + 1522 Swift Testing pass. MLXFoundationModelsTests: 369 pass.
    - RED before the fix (2 DSML rule rows, 1 session row); one mutant run (record as content) killed (2 issues).
    - Real weights, dsv4bench, DeepSeek-V4-Flash-4bit, 4k, greedy: maxTokens 48 thinking: turn 1 stop=length, recorded as reasoning; turn 2 warm cached=4064 prefilled=12 vs cold prefilled=4076; text differs at char 55 at a one-bf16-step tie (warm token is the cold teacher-forced top-1, 45.25 vs 45.0), tie-aware PASS, HARD reuse PASS. Chat mode: cached=4044 prefilled=10, warm == cold YES. maxTokens 1024: thinking turn 2 cached=4137 prefilled=10, warm == cold YES; chat YES.
    - xcodebuild -only-testing DeepseekV4IntegrationTests + DeepseekV4AgenticPromptCacheAssessmentTests: 15 tests in 2 suites pass, TEST SUCCEEDED.
    - next: /review. The task stays in doing.
  timestamp: 2026-10-16T15:29:49.840660+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 thinking mode: a turn cut by maxTokens before </think> renders differently cold and warm'
---
## What

A thinking-mode turn that stops at `maxTokens` before `</think>` has no close in its content. ChatSession records that content as the assistant turn. A later render of the history (a cold session, or any re-render) writes `</think>` in front of the content: `<Assistant><think></think>partial-reasoning<eos>`. The model wrote `<Assistant><think>partial-reasoning`. Thus the cold history is not the text the model wrote.

Measured on 2026-10-12 with `scratchpad/dsv4bench` (mlx-community/DeepSeek-V4-Flash-4bit, 4k context, `maxTokens` 48, thinking mode), after ^6ytd2tg:
- turn 1 stopped at length, content without `</think>`.
- turn 2 warm: cached=4065, prefilled=11 (the committed-turn splice keeps the live trajectory).
- turn 2 cold from `[.user(p1), .assistant(t1.text)]`: the output differs from warm at character 55.
With `maxTokens` 1024 every turn closes and warm == cold.

## Cause

`DeepSeekV4ChatEncoder.body` writes `(reasoning ?? "") + </think>` before the content when `holdsItsOwnReasoning` is false. A content without `</think>` from a live thinking turn is reasoning, not an answer. A hand-built history `.assistant("answer")` is an answer. The encoder cannot tell the two apart from the content alone.

## Approach (to decide)

- Option A: ChatSession records a turn that stopped inside the reasoning block with the content as `reasoning` (or closes it in the record), so that the render writes `partial</think><eos>`, which is what the reference renders for `reasoning_content=partial, content=""`.
- Option B: mark the recorded turn so the encoder renders the content unchanged.
Record the decision and compare with the reference parser (`parse_message_from_completion_text` asserts a `</think>` in thinking mode).

## Decision

Option A. The user asked to finish this task, and the orchestrator selected Option A. Reason: Option A is consistent with the Qwen fix ^ef2zynk, and the render is the same as the reference render for `reasoning_content=partial, content=""`. ChatSession records a thinking turn that stopped inside the reasoning block (no `</think>`) with the text as `reasoning` and an empty content. Turns that close their reasoning do not change.

## Acceptance Criteria

- [x] A thinking-mode turn cut by `maxTokens` re-renders as a prefix-compatible text, and warm turn 2 equals a cold run from the full history (greedy) on real weights. (dsv4bench, mlx-community/DeepSeek-V4-Flash-4bit, 4k context, `maxTokens` 48, thinking: turn 1 stop=length, recorded as reasoning; turn 2 warm cached=4064 prefilled=12, cold prefilled=4076 = 4064 + 12; the text differs at character 55, at a tie: under the cold prompt the teacher-forced top-1 is the warm token " mentions" (45.25) and the cold token " repeatedly" is 45.0, one bf16 step; tie-aware check PASS, HARD reuse check PASS. `maxTokens` 1024: every turn closes, recorded as content, turn 2 cached=4137 prefilled=10, warm == cold text YES in both modes.)
- [x] A unit test pins the render of such a turn. (DeepSeekV4CutTurnSessionTests: the warm turn-2 prompt is `<｜Assistant｜><think>partial</think><｜end▁of▁sentence｜>` and equals the cold prompt; CommittedTurnThinkBlockTests for the DSML decline.)
- [x] A hand-built answer-only history renders as before. (DeepSeekV4CutTurnSessionTests.anAnswerOnlyHistoryRendersAsBefore: `<think></think>answer<｜end▁of▁sentence｜>`.)

## Reference parser

`parse_message_from_completion_text` (mlx_lm `chat_templates/deepseek_v4.py`) reads the thinking part until `</think>` or the tool-call start, and asserts `stop_token == thinking_end_token` ("Invalid thinking format: missing </think>"). Then it asserts the EOS. Thus the reference cannot parse a turn cut by the token budget at all, and has no record for it. The record of this card is the record a caller gives the reference for such a turn (`reasoning_content=partial, content=""`), and the reference render of that record is `partial</think><｜end▁of▁sentence｜>` after `<｜Assistant｜><think>`. The Swift encoder writes the same text.
#deepseek-v4