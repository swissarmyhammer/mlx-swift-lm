---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m05vvejg3gjkdpxx6bwchkty
  text: |-
    Picked up. Research before the experiment.

    **1. The Python reference is NOT available on this machine.**

    - `python3` is `/opt/homebrew/bin/python3` (Homebrew 3.12/3.14).
    - `import mlx` -> `ModuleNotFoundError`.
    - `import mlx_lm` -> `ModuleNotFoundError`.
    - No conda, no virtual environment, no `mlx_lm.generate` on the path. `uv`,
      `uvx` and `pipx` are present, thus an install is possible, but nothing is
      installed now.

    Thus the named experiment needs an install of `mlx` and `mlx-lm` first, and a
    second 141 GiB load in a Python process. Both are held back until the cheaper
    evidence below is read.

    **2. A cheaper discriminator exists, and no test covers it today.**

    The published `config.json` states:

    - `sliding_window` = 128
    - `compress_ratios` = `[0, 0, 4, 128, 4, 128, ...]` over 44 entries
    - `index_topk` = 512

    Thus each layer reads at most the last 128 keys directly. Everything before
    that window reaches the layer ONLY through the pooled chunks of
    `DeepSeekV4ChunkCache` and the top-k of `DeepSeekV4Indexer`.

    The prompt lengths tell the story:

    | test | prompt tokens | crosses the 128 window |
    | --- | ---: | --- |
    | `greedyFirstTokensMatchThePythonFixture` | 18 + 32 generated = 50 | NO |
    | `aShortToolPromptEmitsOneDSMLToolCall` | 328 | YES |

    The one test that compares against the Python reference stays wholly inside
    the sliding window. It therefore proves nothing about the pooled-chunk path.
    The failing tool prompt is the first real-weights measurement that reads it.

    **3. The pooled cache states an invariant that nothing tests.**

    `DeepSeekV4ChunkCache.pooled(_:through:rope:offset:)` documents: "The answer
    is thus the same whatever the block was cut into." Nothing measures that. The
    synthetic checkpoint of `Tests/MLXLMTests/DeepSeekV4ModelTests.swift` sets
    `compress_ratios: []`, thus every model-level forward test runs plain dense
    attention and no compressor at all.

    Thus one weight-free test can discriminate: with a small compressed synthetic
    checkpoint and a sequence longer than the sliding window, a one-block prefill
    and a token-at-a-time decode must give the same logits. A difference names the
    port. This costs no checkpoint load and no Python.

    Next: write that equivalence test and watch it.
  timestamp: 2026-08-16T18:02:58.512335+00:00
- actor: claude-code
  id: 01m05wfzzqvp5hpjsyc2r2e61r
  text: |-
    Cheap experiment 1 is DONE, and it CLEARS one of the two candidates in part.

    New test, weight-free, in `Tests/MLXLMTests/DeepSeekV4CacheTests.swift`:
    `aPrefillFollowedByOneTokenStepsGivesTheAnswerOfOneCall`. It feeds a
    96-token prompt to the compressed synthetic checkpoint two ways and compares
    the last logit row:

    1. one call of 96 tokens;
    2. one prefill call of 85 tokens, then 11 calls of ONE token each.

    That second shape is the generation shape, and no test read it before. The
    nearest test, `aLongPromptGivesOneAnswerWhateverThePrefillCutIs`, cuts the
    prompt into blocks of 13 and never reads a block of one.

    RESULT: the test PASSES on the first run, inside the 2e-3 logit tolerance.

    Thus the pooled-chunk cache, the chunk visibility, the indexer top-k and the
    rotating window are SELF-CONSISTENT across a call boundary. A dropped token
    that came from an off-by-one in the chunk carry or in the cache offset would
    show here, because the synthetic checkpoint crosses its sliding window (8) 12
    times, pools 24 overlap chunks against a top-k budget of 3, and wraps its
    rotating window.

    What the test does NOT prove: that the port agrees with the reference. A port
    that is wrong in the SAME way in both shapes passes it.

    ## What the parity test does and does not reach

    `greedyFirstTokensMatchThePythonFixture` feeds 18 prompt tokens and reads 32
    generated ones, thus 50 positions. Against the published numbers:

    | path | reached at 50 positions | reached at 328 positions |
    | --- | --- | --- |
    | ratio-4 pooled chunks and their indexer | YES, 12 chunks | YES |
    | ratio-128 pooled chunks | NO, 0 chunks | YES, 2 chunks |
    | the rotating window WRAPS (past 128) | NO | YES |
    | index_topk pruning (past 512 chunks) | NO | NO |

    Thus the only structures the tool prompt reads that the Python parity never
    read are the ratio-128 pooled chunks and the wrapped window.

    ## Suite state after the change

    `xcrun xctest MLXLMTests.xctest`: 848 Swift Testing tests passed; 475 XCTest
    tests ran with 1 failure. The failure is
    `TurboQuantIntegrationTests.testRawKeyModeBFloat16MatchesReference`
    (`Tests/MLXLMTests/TurboQuantTests.swift`), which reads no DeepSeek-V4 code
    and which this change cannot reach.

    Next: the real-weights measurement the card asks for. One load, one process:
    the raw generated identifiers, whether two runs give the same ones, and the
    rank and the logit of 5406 at the step that writes the closing tag.
  timestamp: 2026-08-16T18:14:11.703572+00:00
- actor: claude-code
  id: 01m05x3tqj5f6pnypqg8jq08td
  text: |-
    ## The real-weights measurement, with the identifiers and the logits

    New suite: `IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift`,
    test `theGreedyRunWritesTheWholeInvokeClosingTag`. It drives `DeepSeekV4Model`
    directly, because `TokenIterator` answers identifiers and never logits. It
    reads the same 328-token prompt, greedy, budget 128, and runs it TWICE.

    Run of 2026-08-16, one load of the checkpoint, 257 s:

    ```
    DSV4 TOKEN: prompt identifiers = 328
    DSV4 TOKEN: closing tag identifiers = [1718, 128825, 40148, 5406, 32]
    DSV4 TOKEN: the closing tag opens at step 36
    DSV4 TOKEN: step 39 of run 1: chose 1018; candidates [1018 at 28.25,
      5406 at 21.25, 19 at 20.5, 32 at 20.25, 1 at 19.5]; tracked identifier at
      logit 21.25, rank 1, 7.0 below the winner
    DSV4 TOKEN: step 39 of run 2: chose 1018; candidates [1018 at 28.25,
      5406 at 21.25, 19 at 20.5, 32 at 20.25, 1 at 19.5]; tracked identifier at
      logit 21.25, rank 1, 7.0 below the winner
    ```

    The whole generated stream:

    ```
    271, 30, 128825, 72461, 4941, 12548, 1018,
    30, 128825, 40148, 5406, 2329, 1281, 1133, 13088, 1355, 59238, 3816,
    30, 128825, 41523, 2329, 1281, 61027, 4, 3418, 1281, 11476, 3320, 61027, 223, 25,
    1718, 128825, 41523, 1018,
    1718, 128825, 40148, 1018,
    1718, 128825, 72461, 4941, 12548, 32, 1, ...
    ```

    ## Three facts the run states

    1. **The two runs agree, identifier for identifier.** The defect is
       deterministic, thus no non-deterministic path takes part in it.

    2. **The gap is 7.0 logits, and it is NOT a close call.** `>\n` (1018) stands
       at 28.25 and `oke` (5406) at 21.25. The 4-bit quantization of a checkpoint
       moves a confident logit by a fraction of one unit; it does not turn a 30 into
       a 21. Thus the FIRST candidate of the card — the quantization — does not
       explain this measurement. The evidence points at the port.

    3. **The model writes every OTHER multi-token tag whole.** The opening element
       is `30, 128825, 40148, 5406` — `<`, the marker, `inv`, `oke` — thus the model
       writes the word `invoke` correctly when it opens the element. The closing
       `</｜DSML｜tool_calls>` is `1718, 128825, 72461, 4941, 12548, 32`, which is
       THREE name pieces and the bracket, all correct. Only the closing `invoke`
       loses its second piece.

    ## The distance hypothesis

    `DeepSeekV4ChatEncoder` writes the closing tag `</｜DSML｜invoke>` into the
    `## Tools` section, twice, and `theToolPromptTokenizesToThePublishedIdentifiers`
    proves those identifiers are the published ones. That section sits at the START
    of the 328-token prompt.

    `sliding_window` is 128. The step that loses the identifier stands at absolute
    position 367, thus its window covers positions 239 to 367 and the `## Tools`
    section stands OUTSIDE it. The syntax the model must copy reaches that step
    ONLY through the pooled chunks.

    Next: run `DeepseekV4AgenticPromptCacheAssessmentTests.longPromptWithoutToolsRecallsAPlantedFact`,
    which asks the model to read one number out of the body of a 3506-token prompt.
    It is the long-range recall measurement this branch already owns. A failure
    names the pooled-chunk path; a pass takes the distance hypothesis away.
  timestamp: 2026-08-16T18:25:01.682551+00:00
- actor: claude-code
  id: 01m05xprc5yp70dhckvtpejsqq
  text: |-
    ## The long-range control PASSES, thus the distance hypothesis goes away

    `DeepseekV4AgenticPromptCacheAssessmentTests.longPromptWithoutToolsRecallsAPlantedFact`,
    real weights, one load, 36 s:

    ```
    DSV4 CACHE: recall rendered prompt tokens = 3506
    DSV4 CACHE: recall answer = <<<4172>>>
    ```

    The model reads one exact 4-digit number out of row 7 of a 3506-token report
    and answers with it. That number stands about 3400 positions before the query,
    thus far outside the 128-token sliding window. It reaches the answer ONLY
    through the pooled chunks.

    Thus the pooled-chunk long-range path carries an exact string over thousands of
    positions. A path that could not deliver the `## Tools` section over 300
    positions could not do this.

    ## The state of the two candidates of the card

    | candidate | what the measurements say |
    | --- | --- |
    | the 4-bit quantization | UNLIKELY. The gap at the losing step is 7.0 logits. Quantization moves a confident logit by a fraction of one unit. |
    | a number in the attention port | NOT NAMED. Three port properties now measure correct: block-boundary consistency (new weight-free test), long-range recall over 3400 positions (above), and determinism. |

    Neither candidate is CONFIRMED. Each cheap discriminator this branch can reach
    is now spent.

    ## BLOCKER: the experiment the card names cannot run on this machine

    The card's discriminating experiment is the Python reference
    (`ml-explore/mlx-lm`) over the same checkpoint, greedy, reading whether it
    writes 5406. Measured on this machine on 2026-08-16:

    - `python3` is `/opt/homebrew/bin/python3` (Homebrew 3.12 and 3.14).
    - `python3 -c "import mlx"` -> `ModuleNotFoundError: No module named 'mlx'`.
    - `python3 -c "import mlx_lm"` -> `ModuleNotFoundError: No module named 'mlx_lm'`.
    - No conda, no virtual environment, no `mlx_lm.generate` on the path.

    What it would take:

    1. An install of `mlx` and `mlx-lm` into a Python environment. `uv`, `uvx` and
       `pipx` are on the path, thus `uv venv && uv pip install mlx mlx-lm` is the
       short way. It downloads several hundred megabytes.
    2. A SECOND load of the 141 GiB checkpoint, in the Python process. The machine
       holds 512 GiB, thus one Swift process and one Python process must NOT hold
       the checkpoint at the same time.
    3. The script must take one FULL forward over the growing sequence for each
       step, as `DeepseekV4ParityFixture` states. It must NOT use
       `mlx_lm.generate.generate_step`: the doc comment of that fixture records
       that the cached S=1 path of the reference diverges from the reference's own
       full-prompt forward on these weights.
    4. It must feed the same 328 prompt identifiers the Swift side feeds, which
       `Fixtures/deepseek-v4-flash-tool-prompt-tokens.json` already holds, and read
       whether 5406 stands in the answer.

    The step is STUCK on decision 1. The install is not made on my own initiative.

    ## What is in the diff

    - `Tests/MLXLMTests/DeepSeekV4CacheTests.swift`: the new weight-free test
      `aPrefillFollowedByOneTokenStepsGivesTheAnswerOfOneCall`.
    - `IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift`:
      the new real-weights measurement `theGreedyRunWritesTheWholeInvokeClosingTag`.
      It is RED on purpose: it holds the defect the card records, and it prints the
      identifiers and the logits the card asks for.

    No production file changed. The parser is untouched.

    ## The suites

    Five bundles, all green:

    - `MLXLMTests`: 848 Swift Testing tests passed; 475 XCTest tests, 0 failures.
    - `MLXGuidedGenerationTests`: 70 passed.
    - `MLXFoundationModelsTests`: 160 passed.
    - `CXGrammarTests`: 7 passed.
    - `MLXHuggingFaceMacrosTests`: 5 passed.

    One earlier run of `MLXLMTests` reported
    `TurboQuantIntegrationTests.testRawKeyModeBFloat16MatchesReference` at cosine
    0.754 against a bound of 0.97. That test passes alone, it passes in two later
    full runs, and XCTest runs BEFORE Swift Testing in this bundle, thus the new
    test cannot reach it. `git diff --stat main -- Tests/MLXLMTests/TurboQuantTests.swift`
    is empty, thus the file is not this branch's, and the standing rule tosses it.
  timestamp: 2026-08-16T18:35:21.861283+00:00
- actor: claude-code
  id: 01m05xq00ps9fzhwk6zc0eys54
  text: |-
    ### implement — stuck
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepSeekV4CacheTests.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift. Five bundles green (848 + 475 + 70 + 160 + 7 + 5). The real-weights measurement records: two runs agree identifier for identifier; at the losing step `>\n` (1018) stands at logit 28.25 and `oke` (5406) at 21.25, a gap of 7.0; long-range recall over 3400 positions passes.
    - next: a person decides whether to install `mlx` and `mlx-lm` in Python so that the reference run the card names can happen. Without it neither candidate can be confirmed, and the three round-completion criteria stay open.
  timestamp: 2026-08-16T18:35:29.686959+00:00
- actor: claude-code
  id: 01m05y1t3qkeeerj6swjp1z6pp
  text: |-
    ## The reproduction test is now DISABLED, and it stays checked in

    A red test in the tree breaks a clean run for everyone and for CI. The
    `no-test-cheating` rule permits a skip that carries a linked reference, thus
    `theGreedyRunWritesTheWholeInvokeClosingTag` now carries:

    ```swift
    @Test(
        .disabled(
            """
            ^z5xrzg6: DeepSeek-V4 drops identifier 5406 from the closing DSML tag. \
            Enable this test when that defect is corrected.
            """))
    ```

    The whole body and every `print` stay. The next person takes ONE attribute away
    and gets the identifiers, the candidates and the logits back.

    The suite doc comment and the file header each say the same thing: this suite
    holds the REPRODUCTION of an OPEN defect, and its one test is disabled until the
    defect is corrected.

    `aPrefillFollowedByOneTokenStepsGivesTheAnswerOfOneCall` stays ENABLED. It
    passes, and it is coverage of the generation shape that did not exist before.

    ## Evidence after the disable

    Both projects build:

    - `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` -> TEST BUILD SUCCEEDED
    - `xcodebuild build-for-testing -skipPackagePluginValidation -scheme mlx-swift-lm-Package -destination 'platform=macOS'` -> TEST BUILD SUCCEEDED

    The suite skips rather than fails, and it never loads the checkpoint:

    ```
    ◇ Suite DeepseekV4ToolCallTokenDiagnosticTests started.
    ✔ Suite DeepseekV4ToolCallTokenDiagnosticTests passed after 0.001 seconds.
    ✔ Test run with 1 test in 1 suite passed after 0.001 seconds.
    ```

    Five bundles, all green, exit 0 for each:

    | bundle | result |
    | --- | --- |
    | `MLXLMTests` | 848 Swift Testing passed; 475 XCTest, 0 failures |
    | `MLXGuidedGenerationTests` | 70 passed |
    | `MLXFoundationModelsTests` | 160 passed |
    | `CXGrammarTests` | 7 passed |
    | `MLXHuggingFaceMacrosTests` | 5 passed, 0 failures |

    The outcome stays `stuck`. The blocker is unchanged: the reference run the card
    names needs `mlx` and `mlx-lm` in Python, and that install is a person's call.
  timestamp: 2026-08-16T18:41:24.087909+00:00
- actor: claude-code
  id: 01m05y2102g233jahpw0kdt2tt
  text: |-
    ### implement — stuck
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DeepSeekV4CacheTests.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift. The reproduction test carries `.disabled("^z5xrzg6: ...")`, thus the suite reports 1 test skipped and loads no checkpoint. Both projects build. Five bundles green: MLXLMTests 848 + 475/0 failures, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 160, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5.
    - next: a person decides whether to install `mlx` and `mlx-lm` in Python so that the reference run the card names can happen. The three round-completion criteria stay open until it does.
  timestamp: 2026-08-16T18:41:31.138673+00:00
- actor: claude-code
  id: 01m05y4pqvkpd3s60renae0h3g
  text: |
    ### finish iteration 1 — stuck
    - implement: changed — 2 test files, no production code. Three new measurements narrow the card, but neither candidate is confirmed.
    - test: green — five bundles, 848 + 475 + 70 + 160 + 7 + 5, 0 failures. The reproduction test is disabled with a reference to this card, thus the suite stays green.
    - commit: 6e87da0
    - review: not run. The card is not done work.
    - next: BLOCKED on a decision for a person. The reference run needs `mlx` and `mlx-lm` in Python, which are not installed, and it needs a second 141 GiB load of the checkpoint.

    ## What the measurements took away

    - The 4-bit quantization does not explain the defect. The gap at the losing step is 7.0 logits, and a quantization moves a confident logit by a fraction of one unit.
    - The model cannot see the syntax is refuted. `longPromptWithoutToolsRecallsAPlantedFact` reads an exact string about 3400 positions back, thus the pooled-chunk path delivers.
    - A block-boundary defect is refuted at the synthetic level. A new weight-free test compares a whole-prompt prefill against a token-at-a-time decode, and the two agree.
  timestamp: 2026-08-16T18:42:58.939818+00:00
position_column: doing
position_ordinal: '80'
title: 'DeepSeek-V4 generation drops one token: the model writes `</｜DSML｜inv>` for `</｜DSML｜invoke>`'
---
Measured on 2026-08-16 against `mlx-community/DeepSeek-V4-Flash-4bit` with the
real weights, after card `^2dvj1g6` wired the DeepSeek-V4 prompt path into
`LLMModelFactory._load`.

The model now writes a nearly correct DSML tool call. One token is absent:

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_stock_level">
<｜DSML｜parameter name="bay" string="true">bay 7</｜DSML｜parameter>
</｜DSML｜inv>
</｜DSML｜tool_calls>
```

The closing tag reads `</｜DSML｜inv>` where the syntax states
`</｜DSML｜invoke>`. Every other byte of the 179 is correct: the block, the
invoke element, the tool name, the parameter name `bay`, and the closing
`</｜DSML｜tool_calls>`.

`DSMLToolCallParser` refuses the payload for that one tag, thus no tool round
completes.

## Acceptance criteria

This card owns the round-completion criterion. Card `^2dvj1g6` held it while the
cause looked like the prompt. That card corrected the prompt path and proved the
model now reads the `## Tools` section, thus the criterion moved here with the
defect that holds it back.

- [ ] Tell whether the 4-bit weights or the DeepSeek-V4 attention port omits the
      identifier, with the experiment below
- [ ] Correct the cause the experiment names
- [ ] Make one tool round complete:
      `DeepseekV4IntegrationTests.aShortToolPromptEmitsOneDSMLToolCall` passes on
      the real weights
- [ ] `chatModeToolRoundReusesThePromptCache` and
      `thinkingModeToolRoundReusesThePromptCache` both get past
      `#require(roundOne.toolCalls.first)`, which is the agentic measurement that
      every tool-round card waits for

## The absent token is exactly one vocabulary entry

The published `tokenizer.json` holds NO `invoke` token. It writes the word as
two tokens, and the tag ends with one more:

| piece | id |
| --- | ---: |
| `</` | 1718 |
| `｜DSML｜` | 128825 |
| `inv` | 40148 |
| `oke` | 5406 |
| `>\n` | 1018 |

The correct tag is the five identifiers `[1718, 128825, 40148, 5406, 1018]`. The
answer holds the four identifiers `[1718, 128825, 40148, 1018]`. The loss is one
whole token at a true token boundary, and both sequences are legal outputs of a
sampler. The correct text is 182 bytes and the observed text is 179.

## Our text path is proven clean, thus the defect is upstream of it

`DeepSeekV4TokenizerIntegrationTests.aToolCallStreamsBackWholeAcrossItsNewlines`
is weight-free and GREEN. It tokenizes the correct DSML answer with the
production tokenization, streams the identifiers one at a time through
`NaiveStreamingDetokenizer`, and gets the answer back byte for byte. The test
covers the newlines, which is where `NaiveStreamingDetokenizer` starts a new
segment.

Each other stage is clear as well:

- `NaiveStreamingDetokenizer`. `common` is by construction a prefix of
  `newSegment`, thus a divergence RE-EMITS a tail and never deletes one. The
  incomplete-character path returns `nil` without advancing `segment`.
  `startNewSegment()` is the one place with a deletion hazard, and it runs only
  after a newline, which `oke` does not follow.
- `StopStringFilter` is off: `generation_config.json` states no stop string, and
  `stopStrings` and `extraEOSTokens` are both empty. The filter is lossless in
  any case, because a partial match stays in its buffer.
- `ToolCallProcessor` gives the parser a CONTIGUOUS slice of its own buffer, and
  `rawText` carries no sanitizing. A slice cannot hold a gap, thus the buffer
  itself lacked the token.
- `DSMLToolCallParser` holds no regex. It walks literal ranges, and
  `invokeClose` is the literal `</｜DSML｜invoke>`. The parser is correct, and it
  must NOT be loosened: the published `parse_tool_calls` accepts the one syntax
  and raises on every other.
- Speculative decoding is not in play. The load filter of `DeepSeekV4Model` drops
  every `mtp.` tensor, thus the multi-token-prediction head never loads.

Thus the identifier 5406 never entered the stream, and the cause is in the
generation of the identifiers.

## The two candidates

1. The 4-bit quantization of `mlx-community/DeepSeek-V4-Flash-4bit`. Greedy
   decoding is deterministic, thus a wrong argmax at one step repeats on every
   run.
2. A number in the DeepSeek-V4 attention port. A single skipped step is what an
   off-by-one in the pooled-chunk path or in the cache offset looks like.

## How to tell the two apart

Run the same conversation through the Python reference (`ml-explore/mlx-lm`) on
the same checkpoint, greedy, and read the generated identifiers.

- The reference also omits 5406: the checkpoint is the cause, and the port is
  correct.
- The reference writes 5406: the port is the cause.

Capture the raw identifiers on the Swift side as well. `TokenIterator` gives
them directly, as `greedyFirstTokensMatchThePythonFixture` already does.

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process.
#deepseek-v4