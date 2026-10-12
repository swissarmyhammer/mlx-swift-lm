---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m4rtfzshk6j0bqnej7f35fbk
  text: |-
    Research (implement step):

    - `DeepSeekV4EncodingTokenizer.applyChatTemplate` (Libraries/MLXLMCommon/Tokenizer.swift) calls the encoder with the default `dropsEarlierReasoning: true`. In thinking mode, the turn-2 render writes the tail of the turn-1 user turn as `<Assistant></think>` where turn 1 primed `<Assistant><think>`. Thus the turn-2 render parts from the ledger directly after the turn-1 user text.
    - ChatSession records the whole generated text as the assistant content. DeepSeek-V4 declares no ReasoningConfig, thus no decoder splits the reasoning out: the content is `reasoning</think>answer`. `holdsItsOwnReasoning` (card ^v7z7v99) renders such a content unchanged. Thus the reasoning is already replayed into the history through the content. With the drop rule on, the render is even malformed: `<Assistant></think>reasoning</think>answer<eos>`.
    - A caller that splits the reasoning (`reasoning_content` key) reaches `DeepSeekV4ChatEncoder.Message.reasoning` through `Chat.swift` (`message(from:)`). That is the same key that the Qwen replay (^05zt40g, `MessageGenerator.addReasoningMetadata`) writes. Thus no new replay path is necessary.
    - Reference check: downloaded `encoding/encoding_dsv4.py` of deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1 (sha256 bdbd57c1...055e0, the same hash the encoder records). With `drop_thinking=False`: `render_message` writes `reasoning_content + </think>` for every assistant turn in thinking mode, and writes `<Assistant><think>` after every user turn. `encode_messages` itself sets `effective_drop_thinking = False` for every conversation that carries tools. The Swift transcription has the same two branches (`body` and `tail`). The reference parser (`parse_message_from_completion_text`) reads `reasoning` up to `</think>` and the content after it, thus a content of `reasoning</think>answer` rendered with the drop rule off gives the same bytes as the reference render of the parsed turn.

    Decision: the wrapper renders with `dropsEarlierReasoning: false` in all cases. The model then sees exactly the reference text for `drop_thinking=False`, which is the mode the reference uses for every agentic conversation. No new context key: the card permits either form, and a key adds an option that no caller asks for. The encoder default stays `true`, thus direct encoder callers keep the reference default. Cost: earlier reasoning stays in context, the same tradeoff as Qwen `preserve_thinking`.

    Known limit (not in this card): a thinking turn cut by `maxTokens` before `</think>` has no close in its content. The encoder then writes `</think>` in front of the content, and the render parts from the ledger directly after `<think>`. A rule that reads such a content as reasoning would render a hand-built answer-only history wrong, thus it is not part of this change. The scratch benchmark decodes only 48 tokens, thus I raise its budget so that each thinking turn closes.
  timestamp: 2026-10-12T19:47:35.857969+00:00
- actor: claude-code
  id: 01m4rvf1naefb64844581pzdax
  text: |-
    Implementation landed (not committed):

    - `Libraries/MLXLMCommon/Tokenizer.swift`: `DeepSeekV4EncodingTokenizer.applyChatTemplate` now calls the encoder with `dropsEarlierReasoning: false`. The doc comment records why, and that the render is the reference text for `drop_thinking=False`.
    - `Tests/MLXLMTests/DeepSeekV4EncoderWiringTests.swift`: two new tests. `thinkingModeTurnOneRenderIsATurnTwoPrefix` (content `reasoning</think>answer`, as ChatSession records it) and `replayedReasoningRendersAsTheReferenceKeepsIt` (separate `reasoning`, as the `reasoning_content` replay writes it). Both failed before the change (RED, run log: 3 issues, only these two tests) and pass after it.
    - `IntegrationTesting/.../DeepseekV4IntegrationTests.swift`: doc comment of `thinkingModeRoundOneTranscriptIsNotARoundTwoPrefix` only. The test still pins the ENCODER default, which did not change. Comment-only edit, thus no xcodebuild compile run.

    Tests: `swift build --build-tests` clean. `xcrun xctest` MLXLMTests: XCTest 716 passed, Swift Testing 1362 passed. MLXGuidedGenerationTests 70, MLXFoundationModelsTests 367, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5: all passed, 0 failures. The build prints `warning: missing creator for mutated node: (.../mlx-swift_Cmlx.bundle/Contents/MacOS)`. It is a SwiftPM build-graph message about the Cmlx resource bundle. It came in the first build, before the production change, thus it does not come from this change.

    Real weights (dsv4bench, snapshot 38c0bd20, 4k context, greedy). I added an optional 5th argument (decode budget) to the scratch bench:
    - maxTokens 1024, chat: turn1 prefilled=4016 cached=0; turn2 prefilled=10 cached=4043; turn3 prefilled=11 cached=4057; warm == cold.
    - maxTokens 1024, thinking: turn1 prefilled=4016 cached=0 (107 tokens, closed); turn2 prefilled=10 cached=4124 (0.48 s); turn3 prefilled=11 cached=4174; warm == cold ("The user asks to name one bird from the text...").
    - maxTokens 48, thinking (turn 1 cut before `</think>`): turn2 prefilled=11 cached=4065, but warm != cold at char 55. Cause: the cold render writes `<think></think>partial<eos>` for a content without a close. New task ^mf1mwaw records this edge. It is outside the four criteria of this card, because a closed turn (the normal case) gives warm == cold.
  timestamp: 2026-10-12T20:04:33.578630+00:00
- actor: claude-code
  id: 01m4rvf3mhefwngae7qwrmqnar
  text: |-
    ### implement — changed
    - evidence: 3 files — Libraries/MLXLMCommon/Tokenizer.swift, Tests/MLXLMTests/DeepSeekV4EncoderWiringTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift (doc comment). All five xctest bundles green (MLXLMTests 716 XCTest + 1362 Swift Testing). dsv4bench thinking turn 2 cached=4124 prefilled=10, warm == cold.
    - next: /review
  timestamp: 2026-10-12T20:04:35.601509+00:00
- actor: claude-code
  id: 01m4rw3bdys6qb0seqtsmq1bqj
  text: |-
    ### test - green
    - evidence: swift build --build-tests - Build complete, 0 compiler warnings. xcrun xctest on 5 bundles: CXGrammarTests 0 tests, MLXFoundationModelsTests 7 Swift Testing tests passed, MLXGuidedGenerationTests 367 Swift Testing tests passed, MLXHuggingFaceMacrosTests 5 XCTest passed, MLXLMTests 716 XCTest passed (0 failures) and 1362 Swift Testing tests passed (DeepSeekV4EncoderWiringTests suite passed). 0 failed, 0 skipped. xcodebuild build-for-testing (IntegrationTesting) - TEST BUILD SUCCEEDED, 0 compiler warnings.
    - warning 1: `missing creator for mutated node` for mlx-swift_Cmlx.bundle. It is also there without the change: git stash, swift build --build-tests (Tokenizer.swift touched to force a rebuild) gave the same warning, then git stash pop restored the change. It comes from the build system, not from the source.
    - warning 2: appintentsmetadataprocessor says `Metadata extraction skipped, no AppIntents.framework dependency found`. It is an Xcode tool message, not a compiler warning. The package has no AppIntents dependency.
    - next: review. Nothing committed.
  timestamp: 2026-10-12T20:15:38.942919+00:00
- actor: claude-code
  id: 01m4rw6mpcsh3zpc65vp5f9g87
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (commit 0755ec0). counts: findings 0, confirmed 0, refuted 0, attempted 7, failed 0. 3 source files reviewed. 4 .kanban files not reviewed (no validator matches them). The task has no earlier Review Findings section.
    - acceptance criteria: all 4 are checked. The comments give evidence for each: turn 2 cached=4124 prefilled=10 in thinking mode; warm == cold in thinking mode; unit test thinkingModeTurnOneRenderIsATurnTwoPrefix passes; chat mode turn 2 cached=4043 prefilled=10, warm == cold.
    - next: done. The cut-before-</think> edge is in task ^mf1mwaw.
  timestamp: 2026-10-12T20:17:26.732645+00:00
- actor: claude-code
  id: 01m4rw717wxy8wzbz4hj7cnrxq
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — Libraries/MLXLMCommon/Tokenizer.swift, Tests/MLXLMTests/DeepSeekV4EncoderWiringTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift
    - test: green — MLXLMTests 716 XCTest + 1362 Swift Testing, other bundles 379, 0 failures; IntegrationTesting build-for-testing succeeded
    - commit: 0755ec0
    - review: clean — review sha HEAD~1..HEAD, 0 findings
  timestamp: 2026-10-12T20:17:39.580317+00:00
position_column: done
position_ordinal: ffb980
title: 'DeepSeek-V4 thinking mode: make the prompt cache reuse the prefix across turns'
---
## What

In thinking mode, a `ChatSession` with `mlx-community/DeepSeek-V4-Flash-4bit` never reuses the prompt cache. Each turn prefills the full conversation again.

Measured on 2026-10-12 (M3 Ultra, 8k-token context, `scratchpad/dsv4bench`):
- Chat mode: turn 2 reused 8043 tokens and prefilled 10 tokens in 0.69 s. The warm output was the same as the cold output.
- Thinking mode: turns 1, 2 and 3 each prefilled 8016, 8075 and 8135 tokens with `cached=0`, about 30 s for each turn.

Cause: `Tokenizer.applyChatTemplate` (`Libraries/MLXLMCommon/Tokenizer.swift:349-359`) calls `DeepSeekV4ChatEncoder().encode(messages:thinkingMode:)` with the default `dropsEarlierReasoning`. In thinking mode the encoder then removes the reasoning of earlier assistant turns (`DeepSeekV4ChatEncoder.swift`, `removingEarlierReasoning`). Thus the render of turn 1 is not a prefix of the render of turn 2.

This is the same class of problem as the Qwen 3.6 case. Task `01KXZP6CTSETM3YJS3E05ZT40G` fixed it with `preserve_thinking` and a replay of the reasoning content, so that the history stays a prefix. Also see commit `bcf4f54`, which keeps the system prompt the same for Qwen 3.5.

## Approach

- Make the session render keep the earlier reasoning (`dropsEarlierReasoning: false`, or a context key that selects it), and replay the reasoning of each assistant turn into the history.
- Make sure that the model sees the same text as the published reference encoder when the reasoning is kept. Record the decision.

## Acceptance Criteria

- [x] In thinking mode, turn 2 of a `ChatSession` reports `cachedPromptTokenCount > 0` and prefills only the new turn. (dsv4bench, 4k context, maxTokens 1024: turn 2 cached=4124 prefilled=10; turn 3 cached=4174 prefilled=11.)
- [x] The warm turn 2 output is the same as a cold run from the full history (greedy). (dsv4bench: "warm == cold turn 2: YES" in thinking mode.)
- [x] A unit test shows that the thinking-mode render of turn 1 is a prefix of the render of turn 2. (`DeepSeekV4EncoderWiringTests.thinkingModeTurnOneRenderIsATurnTwoPrefix`.)
- [x] Chat mode still reuses the cache. (dsv4bench: turn 2 cached=4043 prefilled=10, warm == cold.)
#deepseek-v4