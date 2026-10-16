---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m513rjaxdq3b873fwzhzbnj5
  text: |-
    Research (implement step):
    - The splice decision is in `QwenCommittedTurnRule.reuse` (Libraries/MLXLMCommon/Tool/Parsers/QwenCommittedTurnRule.swift), which calls the shared `CommittedTurnSplice` (also used by `DSMLCommittedTurnRule`). The Qwen rule is the correct place for the change: the DSML rule stays as it is.
    - `ToolCallFormat.qwen35.promptCacheReuseRules(tokenizer:)` makes the rule for BOTH paths: ChatSession (`PromptCacheReusePolicy`) and the MLXFoundationModels executor (`reconcilePromptCache` in ExecutorPromptCache). Thus one change in the rule covers both paths.
    - When the rule declines, the standard rules decide. For a hybrid model (caches that cannot rewind), `RewindToCommonPrefixRule` gives `.rebuild`. ChatSession then restores the longest checkpoint the render starts with (`restoringCheckpoint`); the end-of-prompt checkpoint is the render of turn 1, which is a prefix of the turn-2 render. The executor restores the one checkpoint its entry carries (`restoring(_:input:promptTokens:)`).
    - If the render of turn 2 extends the cached tokens exactly (no trim of the reasoning, same tokens), `ExtendCachedPrefixRule` reuses the whole ledger. That is also warm == cold.
    - Plan: the rule resolves the ids of `<think>` and `</think>` and declines when the last think marker in the cached tokens is `<think>` (the cache ends inside an open think block). Thinking off ends the prompt with a closed block, thus the splice stays. A turn that writes `</think>` keeps the splice.
    - Note: a turn that goes from reasoning to `<tool_call>` with no `</think>` (implicit end) also ends with no close of the think block in the generated tokens, thus the rule declines it too. This agrees with the decision text "no close of the think block in the generated tokens".
  timestamp: 2026-10-16T01:03:29.629819+00:00
- actor: claude-code
  id: 01m514j8d36st4vbvr6tp9c1x6
  text: |-
    TDD record:
    - Tests: `ChatSessionReasoningHistoryTests.theTurnAfterTurnOneEnds` (3 rows, hybrid `ScriptedTokenModel` with a recurrent layer; it checks where turn 2 starts to feed the caches and whether the caches hold the cold render of turn 2) and `QwenCommittedTurnRuleTests` "the splice after a turn that ends" (4 rows) plus "a tokenizer without think markers keeps the splice inside reasoning". `ScriptedTokenModel` got an option for one recurrent layer (MambaCache) and records the tokens of each pass.
    - RED (before the rule change): 5 issues. Session row "stops inside reasoning that the template trims": turn 2 started at 50, expected 37 (the end-of-prompt checkpoint); the held tokens were `<think>\nParis first. <|im_end|>`, no close of the think block. Session row "the render extends": held tokens not the cold render. Rule rows "stopped on the budget inside its reasoning" and "ended its turn inside its reasoning": the rule spliced. Control rows passed.
    - GREEN: MLXLMTests 721 XCTest 0 failures, 1511 Swift Testing tests passed.
    - Mutant (`endsInsideBlock` = `tokens.contains(open)`, which ignores `</think>`): 4 issues, all in the controls (session control row, rule rows "closed its reasoning" and "thinking off"). Mutant reverted.
    - Discovery: when the turn-2 render extends the cached reasoning tokens exactly, the decline lets `ExtendCachedPrefixRule` reuse the whole ledger (warm == cold, fewest tokens). The checkpoint is used only when the template trims or retokenizes the reasoning.
  timestamp: 2026-10-16T01:17:31.427879+00:00
- actor: claude-code
  id: 01m515z1jtkzx7a1v8c9zectjc
  text: |-
    Real-model measurement (Qwen3.8-27B-mxfp4, qwen38bench full mode, 32k, thinking on, QWEN38_THINKING_KEY=enable_thinking, one GPU job at a time, idle GPU checked before each run). Binary: scratchpad/qwen38bench/new-ef2zynk (xcodebuild Release of the final tree).

    48-token decode budget (scratchpad/qwen38/ef2-new-48-1.txt):
    - turn 1 stop=length closed=no (stops inside reasoning, as in z4a-new-48-1.txt).
    - turn 2 warm: prefilled=22 cached=32242 prompt 0.26 s. cached = 32194 (turn-1 prompt) + 48 (generated tokens).
    - warm == cold turn 2 (text): YES. Tie-aware: PASS. HARD reuse check: PASS. Warm and cold text are the same ("User asks: ...").
    - Before (z4a-new-48-1.txt, splice): prefilled 19, 0.25 s, warm "We" against cold "User".

    This differs from the expected numbers (about 73 tokens, about 0.4 s). Reason: the rule declines the splice, and then `ExtendCachedPrefixRule` applies first, because on the real model the turn-2 render holds the 48 generated reasoning tokens exactly (the render only adds `\n</think>\n\n<|im_end|>...` after them). That reuse is an exact prefix, thus warm == cold, with fewer tokens than the checkpoint. The end-of-prompt checkpoint is the fallback when the render does not extend the cached tokens (the template trims whitespace or the tokens differ); the unit test row "stops inside reasoning that the template trims" covers that path.

    256-token decode budget (scratchpad/qwen38/ef2-new-256-1.txt): unchanged. turn 1 closed=yes; turn 2 prefilled=18 cached=32432 prompt 0.26 s; tie-aware PASS (gap 0.0, the same tie as z4a-new-256-1.txt); HARD reuse PASS.

    thinking=false (in both runs): turn 2 prefilled=20 cached=32180, warm == cold YES, unchanged.

    Executor path (MLXFoundationModels): it uses the same rule through `ToolCallFormat.qwen35.promptCacheReuseRules` and `reconcilePromptCache`. When the rule declines, the standard rules decide; when they give no reuse, `restoring(_:input:promptTokens:)` restores the checkpoint the entry carries. No executor code change is necessary.
  timestamp: 2026-10-16T01:41:59.002188+00:00
- actor: claude-code
  id: 01m515z6bmgj99f7emkknp1k4e
  text: |-
    ### implement — changed
    - evidence: 4 source files — Libraries/MLXLMCommon/Tool/Parsers/QwenCommittedTurnRule.swift (decline when the cached tokens end inside an open think block; resolves `<think>` and `</think>` ids), Tests/MLXLMTests/ScriptedTokenModel.swift (optional recurrent layer, records fed tokens), Tests/MLXLMTests/ChatSessionReasoningHistoryTests.swift (3-row hybrid session test), Tests/MLXLMTests/QwenCommittedTurnRuleTests.swift (4-row rule table and a no-think-markers case). `swift build --build-tests`: 0 warnings (only the known "missing creator" note). `xcrun xctest MLXLMTests.xctest`: 721 XCTest 0 failures, 1511 Swift Testing passed. `MLXFoundationModelsTests.xctest`: 368 passed. xcodebuild -only-testing Qwen35SessionPromptCacheTests, Qwen35CheckpointIntegrationTests, Qwen35ThinkingOffTemplateTests: 13 tests passed, TEST SUCCEEDED. RED 5 issues before the fix; mutant run killed (4 issues). swift format lint clean on the 4 files; swiftlint no_magic_numbers reports nothing. Bench 48 tokens: turn 2 22 tokens 0.26 s warm == cold YES; 256 tokens unchanged (18 tokens, 0.26 s). Not committed.
    - next: /review
  timestamp: 2026-10-16T01:42:03.892179+00:00
- actor: claude-code
  id: 01m5162neapk7wbfb0sp57kdhw
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (836b90e). 0 findings, 0 confirmed, 0 refuted. 7 validator runs. 4 files reviewed. The validators do not apply to the 2 .kanban files.
    - next: none. The task is in done.
  timestamp: 2026-10-16T01:43:57.642130+00:00
- actor: claude-code
  id: 01m51630j7sj3gss6qv6574m7n
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — 4 files; option 2; 48-token case at 32k: 22 tokens, 0.26 s, warm == cold
    - test: green — run on the final tree by implement: MLXLMTests 721 + 1511, MLXFoundationModelsTests 368, integration 13; 0 compiler warnings
    - commit: 836b90e
    - review: clean — 0 findings; task moved to done
  timestamp: 2026-10-16T01:44:09.031359+00:00
position_column: done
position_ordinal: ffdd80
title: 'Qwen3.8: after a turn that stops inside its reasoning, the splice of ChatSession omits the close of the think block'
---
## What

Found in ^z4aax33. ChatSession now records the reasoning of a Qwen 3.8 turn in `reasoning_content`. When turn 1 stops on the token budget inside its reasoning (no `</think>`), the render of turn 2 writes `<think>\n` + reasoning + `\n</think>\n\n` + `<|im_end|>`. The cache holds the reasoning without `\n</think>\n\n`.

`QwenCommittedTurnRule` (CommittedTurnSplice) splices at the commit: the warm model reads its unclosed reasoning, then `<|im_end|>`. The cold render has the closed block. In the qwen38bench at 32k with 48 decode tokens (scratchpad/qwen38/z4a-new-48-1.txt), turn 2 warm and cold gave different first tokens ("We" against "User"). The tie-aware check passed (gap 2.625, margin 4.0).

Before ^z4aax33 this case went back to the checkpoint at the end of the last message and prefilled the canonical render (warm == cold, 73 tokens, 0.42 s). Now it prefills 19 tokens (0.25 s).

## Options

1. Keep the splice (the documented behavior of CommittedTurnSplice for a budget stop).
2. Decline the splice when the generation stopped inside reasoning, and go back to the checkpoint at the end of the prompt. That checkpoint is now a prefix of the turn-2 render. The cost is a prefill of the generated tokens again.
3. Feed the tokens of the close of the think block before the commit.

A person must select the option.

## Decision

The user selected option 2. Decline the splice when the generation stopped inside reasoning (the generated tokens do not close the think block). Go back to the checkpoint at the end of the prompt.

Reason: this gives an exact match (warm == cold) for a rare case. The cost is a prefill of the generated tokens again.

## Acceptance Criteria

- [x] The card records the selected option and the reason.
- [x] A test shows the turn-2 reuse after a turn that stops inside its reasoning. #qwen3-8