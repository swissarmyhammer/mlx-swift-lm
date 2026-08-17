---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m07y6k1ab89jmwhp66y4qybq
  text: |
    ### The cause, measured with real weights

    One instrumented run (temporary `DSV4 PROBE:` prints beside `promptCachePolicy.decide`
    in `ChatSession`), both tool tests, one process, 1003 s. Each line is the state the
    policy read.

    ```
    chat    round 1   ledger=0    prompt=3814 common=0    trimmable=true  decision=prefillAll
    chat    tool      ledger=3875 prompt=3906 common=3830 trimmable=false decision=rebuild
    chat    followUp  ledger=3951 prompt=3967 common=3951 trimmable=false decision=appendSuffix(3951)
    think   round 1   ledger=0    prompt=3814 common=0    trimmable=true  decision=prefillAll
    think   tool      ledger=3888 prompt=3918 common=3814 trimmable=false decision=rebuild
    think   followUp  ledger=4100 prompt=4117 common=3918 trimmable=false decision=rebuild
    ```

    The reuse machinery is correct. `ExtendCachedPrefixRule` declines because the
    LEDGER is not a prefix of the render, and `RewindToCommonPrefixRule` cannot
    rewind a rotating cache past its window (`trimmable=false`), thus it rebuilds.

    The ledger is round 1's render PLUS the tokens the model wrote. The render of the
    same turn as HISTORY does not write those tokens again. There are TWO reasons,
    and both are text the encoder adds that the model never wrote.

    **1. Thinking mode: a second `</think>`.** Divergence at 3814, which is the first
    generated token.

    ```
    ledger tail = <<<I've read through the stock report. Now I need to call the get_stock_level tool for bay 7 as>>>
    prompt tail = <<<</think>I've read through the stock report. Now I need to call the get_stock_level tool for bay 7>>>
    ```

    `ChatSession.Conversation.record` stores the whole generated text as
    `Chat.Message.assistant(content:)` and never fills `Chat.Message.reasoning`.
    DeepSeek-V4 declares NO `reasoningConfig` (see the note in
    `Libraries/MLXLLM/Models/DeepSeekV4.swift`), thus no decoder splits the reasoning
    out and the content holds `reasoning + </think> + answer` as one string.
    `DeepSeekV4ChatEncoder.body(of:)` then writes `(reasoning ?? "") + thinkEnd`
    in FRONT of that content, thus the render holds `</think>` twice. The same defect
    breaks the thinking-mode follow-up round (divergence at 3918, same shape).

    **2. Both modes: two extra newlines before the DSML block.** Divergence at 3830,
    which is 16 tokens into the generated region.

    ```
    ledger tail = <<<.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_stock_level">\n<｜DSML｜parameter name="bay>>>
    prompt tail = <<<.\n\n\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_stock_level">>>>
    ```

    The model writes `answer.` then a blank line then the block. `ToolCallProcessor`
    records the text in front of the block as the response, blank line included, thus
    the content ends with `\n\n`. `DeepSeekV4ChatEncoder.toolCallsBlock` then adds its
    own `\n\n` separator, thus the render holds four newlines where the model wrote
    two.

    Both are defects of the history render, not of the policy. The fix makes the
    encoder write the assistant turn the model actually wrote, thus
    `ExtendCachedPrefixRule` fires on its own and no new rule is necessary.
  timestamp: 2026-08-17T13:22:29.546017+00:00
- actor: claude-code
  id: 01m082ejrweyr9t6kz4t3sxbe8
  text: |
    ### Causes 3 and 4, measured, and the design decision

    The two encoder defects of the comment above are removed and the tests hold them.
    A second instrumented real-weights run (1524 s, one load, both modes, probe now
    prints the token ids on each side of the divergence) shows the divergence MOVED
    but did not go away.

    ```
    chat   tool round   ledger=3875 prompt=3906 common=3830 -> 3869   decision=rebuild
    think  tool round   ledger=3888 prompt=3917 common=3814 -> 3880   decision=rebuild
    think  follow-up    ledger=4173 prompt=4189 common=3918 -> 3929   decision=rebuild
    chat   follow-up    ledger=4019 prompt=4035 common=4019           decision=appendSuffix(4019)
    ```

    **Cause 3: the model writes ABBREVIATED DSML closing tags.** At the divergence of
    the thinking tool round the next id of the ledger is 1018 (`>`) and the next id of
    the render is 5406 (`oke`):

    ```
    ledger ids = [1018, 1718, 128825, 72461, 4941, 12548, 32, 1]
    prompt ids = [5406, 1018, 1718, 128825, 72461, 4941, 12548, 32, 1, 128803, ...]
    ledger text = <<<>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>>>>
    prompt text = <<<oke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>...
    ```

    The model wrote `</｜DSML｜inv>` where the syntax states `</｜DSML｜invoke>`. In
    chat mode it abbreviates BOTH closing tags — the ledger holds
    `</｜DSML｜tool>` (ids 128825, 72461, 32) where the render writes
    `</｜DSML｜tool_calls>` (ids 128825, 72461, 4941, 12548, 32). Card `^z5xrzg6`
    already measured that abbreviation and taught `DSMLToolCallParser` to read it.
    The abbreviation is a property of the BYTES the model wrote; it is not part of the
    parsed `ToolCall`, thus no re-serialization of a structured call can reproduce it.

    **Cause 4: the model writes non-canonical token splits.** The thinking follow-up
    round diverges on ordinary prose, not on a tag:

    ```
    ledger ids = [6368, 1154, 4, 790, 270, 7717, ...]
    prompt ids = [39982, 273, 4, 790, 270, 7717, ...]
    ```

    Two tokens against two tokens, and the streams resync immediately after. The model
    emitted ` pal` + `les`; the tokenizer encodes the same text as ` pall` + `es`. The
    tokens a model generates are not always the canonical tokenization of the text
    they decode to, thus even a render of the model's exact TEXT cannot reproduce its
    TOKENS.

    ### The decision

    Cause 4 rules out every fix that goes through a render, the raw-text render
    included: a render passes through the tokenizer, and the tokenizer canonicalizes.
    Cause 3 rules out every fix that rebuilds the block of calls from the parsed
    `ToolCall`. Byte-exact reuse is therefore unreachable by rendering, and it is
    reachable only by keeping the model's own tokens in the cache and splicing the new
    tail onto them.

    That is the extension point the codebase already owns.
    `ToolCallFormat.promptCacheReuseRules(tokenizer:)` states it word for word: "A
    protocol that keeps state in the KV cache which the template cannot reproduce
    contributes a rule here." `.gptOSS` contributes `HarmonyToolRestartRule` and
    `.atem` contributes `OnyxToolRestartRule` for exactly this reason. `.dsml`
    contributes nothing today, and the measurement proves it belongs in that set.

    Thus: a new `DSMLCommittedTurnRule` for `.dsml`. It claims a turn whose cache
    holds a COMMITTED assistant turn, and it splices the new tail onto the live
    trajectory at the `<｜end▁of▁sentence｜>` the model itself wrote.

    It differs from the Harmony and Onyx rules in one way, and the measurement asks
    for it: those two claim a tool-result continuation alone, while the thinking
    follow-up round of this card is an ordinary user turn whose PRECEDING assistant
    turn is the unrenderable one. Thus the rule keys on the commit, not on the tool
    result.

    Its safety comes from one new fact, which the suite already measures as true: the
    new render must hold the render of the previous prefill as a WHOLE prefix (fact
    (a), 3814 of 3814). That proves the template rewrote no already-cached rendered
    region, thus the only region left to differ is the generation region, and the
    cache holds the true version of it. A render that DOES rewrite an earlier region —
    `dropsEarlierReasoning` on a tool-free thinking conversation, or a changed system
    prompt — fails that guard and falls through to the standard rules unchanged.

    Alternatives weighed and rejected:

    - Render the raw text the model wrote. Cause 4 defeats it, and it would also make
      every other model's transcript hold raw tool-call syntax that its own template
      re-serializes, thus a doubled block.
    - Carry the raw source bytes on the public `ToolCall`. Cause 4 defeats it as well,
      and it puts a byte-fidelity concern of one model into a public type every parser
      and the FoundationModels bridge share.
    - Rewind to the common prefix. The cache answers `trimmable=false` in every
      measurement above: a `RotatingKVCache` past its window drops nothing.
  timestamp: 2026-08-17T14:36:45.724841+00:00
- actor: claude-code
  id: 01m0845xa0rkeg1nbvgtbd0mnz
  text: |
    ### The fix landed, weight-free suites green, real-weights run next

    Three changes, each written test-first and each proved RED before it was made.

    **1. `DeepSeekV4ChatEncoder` writes the turn the model wrote** (causes 1 and 2).
    `holdsItsOwnReasoning(_:)` writes no second `</think>` in front of a content that
    already holds one. `contentBeforeToolCalls(of:)` takes away the newlines that
    belong to the block of calls. `Tests/MLXLMTests/DeepSeekV4ToolEncodingTests.swift`
    holds the rule, including the whole property: the render of the tool round must
    hold the render of round 1 and then the text the model wrote.

    **2. `DSMLCommittedTurnRule`** (causes 3 and 4). A protocol cache rule for
    `.dsml`, beside `HarmonyToolRestartRule` and `OnyxToolRestartRule`. It splices the
    new tail onto the live trajectory at the `<｜end▁of▁sentence｜>` of the last
    assistant turn. `Tests/MLXLMTests/DSMLCommittedTurnRuleTests.swift` holds the
    whole decision table: 6 splices and 10 declines.

    **3. The session records the render of each prefill.**
    `ChatSession.Conversation.renderedTokens` and
    `PromptCacheState.previousRenderTokens`. The rule reads it to prove that the new
    render holds the previous render as a WHOLE prefix, thus the generation region is
    the only region left that can differ.
    `Tests/MLXLMTests/DeepSeekV4CommittedTurnSessionTests.swift` drives a real
    `ChatSession` over a scripted DeepSeek-V4 conversation and asserts the splice
    arithmetic. Before the wiring it recorded `Pass(offset: 3, tokenCount: 5)` — the
    rewind — and after it records the splice.

    ### One more measured fact that shaped the rule

    The thinking tool round writes 256 tokens, which IS the whole token budget of the
    suite, thus that turn carries NO end-of-sentence marker. Its ledger does not end
    at a commit. A rule that demanded a committed ledger would decline the follow-up
    round and leave the third assertion failing. Thus the rule splices AT the commit
    in that case rather than after it: the suffix carries the render's commit, and the
    model reads its own unterminated answer in front of it. The same arithmetic serves
    a speculative iterator that returned the commit before the cache took it.

    ### Test evidence, weight-free

    All five package bundles, `xcodebuild build-for-testing` then `xcrun xctest`:

    ```
    MLXLMTests                 479 XCTest + 874 Swift Testing   0 failures
    MLXGuidedGenerationTests    70 Swift Testing                0 failures
    MLXFoundationModelsTests   160 Swift Testing                0 failures
    CXGrammarTests               7 Swift Testing                0 failures
    MLXHuggingFaceMacrosTests    5 XCTest                       0 failures
    ```

    The `IntegrationTesting` project builds. The real-weights confirmation run is
    next.

    ### What did not work, so the next agent does not repeat it

    - **A render can never reproduce the tokens.** Cause 4 is decisive: re-rendering
      the model's exact TEXT still gives different TOKENS, because a render goes
      through the tokenizer and the tokenizer canonicalizes. Do not try to fix this in
      the encoder.
    - **`Tokenizer.eosTokenId` is a protocol EXTENSION, not a requirement.** A test
      stub that declares `eosTokenId` sees it ignored through `any Tokenizer`; the
      generation never stops. Declare `eosToken` and let `convertTokenToId` resolve
      it. This cost one build cycle.
  timestamp: 2026-08-17T15:06:58.752563+00:00
- actor: claude-code
  id: 01m085bce6b9ypwtgf8spa6gyv
  text: |
    ### implement — changed

    - evidence, real weights, one load, both tests, one process, 1033 s:
      `xcodebuild test-without-building -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -only-testing:'…/chatModeToolRoundReusesThePromptCache()' -only-testing:'…/thinkingModeToolRoundReusesThePromptCache()'`
      → `✔ Test run with 2 tests in 1 suite passed after 1033.438 seconds.`
      chat tool round: fed 28 of 3906, skipped 3878, 2.55 s against a 15.33 s cold
      control. thinking tool round: fed 28 of 3917, skipped 3889, 2.77 s against
      16.26 s. chat follow-up: skipped 3951. thinking follow-up: skipped 4068.
      Not one assertion of the suite was weakened.
    - evidence, weight-free, all five package bundles green, zero failures:
      MLXLMTests 479 XCTest + 874 Swift Testing; MLXGuidedGenerationTests 70;
      MLXFoundationModelsTests 160; CXGrammarTests 7; MLXHuggingFaceMacrosTests 5.
      Both projects build: `xcodebuild build-for-testing` on `mlx-swift-lm-Package`
      and on `IntegrationTesting`.
    - evidence, 8 files:
      `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift`,
      `Libraries/MLXLMCommon/Tool/Parsers/DSMLCommittedTurnRule.swift` (new),
      `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`,
      `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`,
      `Libraries/MLXLMCommon/ChatSession.swift`,
      `Tests/MLXLMTests/DeepSeekV4ToolEncodingTests.swift`,
      `Tests/MLXLMTests/DSMLCommittedTurnRuleTests.swift` (new),
      `Tests/MLXLMTests/DeepSeekV4CommittedTurnSessionTests.swift` (new).
      `swift-format lint` is clean on all eight, and no other file was formatted.
    - next: `/review`. The card stays in `doing`.
  timestamp: 2026-08-17T15:27:26.662329+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4: the tool round reuses NO prompt cache, although its render extends round 1 whole'
---
Found on 2026-08-17 while card `^z5xrzg6` made DeepSeek-V4 tool calling work. The
tool round now completes, thus the prompt-cache assertions of
`DeepseekV4AgenticPromptCacheAssessmentTests` are REACHABLE for the first time.
Fact (a) passes. Fact (b) fails.

## The measurement, real weights, one load

`chatModeToolRoundReusesThePromptCache` and
`thinkingModeToolRoundReusesThePromptCache`, one process, 1034 s.

| quantity | chat | thinking |
| --- | ---: | ---: |
| round 1 rendered tokens | 3814 | 3814 |
| round 1 tool calls | `["get_stock_level"]` | `["get_stock_level"]` |
| tool round rendered tokens | 3906 | 3918 |
| tool round FED tokens | 3906 | 3918 |
| tool round tokens skipped by reuse | **0** | **0** |
| tool round prefill seconds | 15.71 | 18.14 |
| cold control prefill seconds | 15.63 | 21.39 |
| follow-up round tokens skipped by reuse | 3951 | **0** |
| follow-up round prefill seconds | 2.69 | 19.91 |

## Fact (a) PASSES, thus the seam is not the cause

- `chat tool round extends round 1 = true`, common prefix 3814 of 3814.
- `thinking tool round extends round 1 = true`, common prefix 3814 of 3814.

Round 1's divergent tail is EMPTY in both modes, thus the tool round's render
holds round 1's render as a whole prefix. `ExtendCachedPrefixRule` is reachable,
and it skips nothing.

## Fact (b) FAILS

- `toolRound.skippedTokenCount > 0` fails in both modes: 0 skipped.
- `toolRound.prefillSeconds < control.prefillSeconds * noReuseTimeFloorFraction`
  fails in both modes. The tool round takes as long as a COLD session on the same
  prompt, which is what 0 skipped tokens predicts.
- `followUp.skippedTokenCount > 0` fails in THINKING mode alone: 0 skipped.

## The one clue the numbers carry

Chat mode's FOLLOW-UP round DOES reuse: it skips 3951 of 3967 tokens and takes
2.69 s in place of about 15 s. Thus the reuse machinery works on that path. Only
the TOOL round resets it, in both modes, and in thinking mode the follow-up round
after the tool round stays reset as well.

The divergent tail of the thinking tool round is:

```
</think>I've read through the stock report. Now I need to call the get
```

Thus the tool round's render adds a `</think>` close and the assistant text of
round 1 before the tool result. Read whether the session drops or rebuilds the
cache when it appends a tool-result turn, rather than extending it.

## The cause, in four parts

The ledger of `ChatSession` is the render of round 1 PLUS the tokens the model
wrote. The render of that same turn as HISTORY does not write those tokens again,
for four measured reasons. The comments of this card hold the token ids of each.

1. Thinking mode: the render writes a second `</think>`, because the transcript
   keeps the whole generated text as the content and the encoder adds its own
   close in front of it.
2. Both modes: the render writes four newlines in front of the block of calls
   where the model wrote two.
3. The model ABBREVIATES its own DSML closing tags: `</｜DSML｜inv>` for
   `</｜DSML｜invoke>`, and `</｜DSML｜tool>` for `</｜DSML｜tool_calls>`.
4. The model writes NON-CANONICAL token splits: ` pal` + `les` where the tokenizer
   encodes the same text as ` pall` + `es`.

Parts 1 and 2 are defects of the history render and they are corrected in
`DeepSeekV4ChatEncoder`. Parts 3 and 4 cannot be corrected by any render, because
a render passes through the tokenizer. They are served by a protocol cache rule,
`DSMLCommittedTurnRule`, which splices the new tail onto the live trajectory at
the `<｜end▁of▁sentence｜>` of the last assistant turn.

## The measurement after the fix, real weights, one load

| quantity | chat | thinking |
| --- | ---: | ---: |
| tool round rendered tokens | 3906 | 3917 |
| tool round FED tokens | 28 | 28 |
| tool round tokens skipped by reuse | **3878** | **3889** |
| tool round prefill seconds | 2.55 | 2.77 |
| cold control prefill seconds | 15.33 | 16.26 |
| follow-up round FED tokens | 16 | 16 |
| follow-up round tokens skipped by reuse | **3951** | **4068** |
| follow-up round prefill seconds | 2.69 | 1.47 |

## Acceptance criteria

- [x] Name why the tool round feeds every token although its render extends
      round 1 whole
- [x] `chatModeToolRoundReusesThePromptCache` passes on the real weights
- [x] `thinkingModeToolRoundReusesThePromptCache` passes on the real weights

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process. The two
tests above run in one process, because the suite awaits one shared load. #deepseek-v4