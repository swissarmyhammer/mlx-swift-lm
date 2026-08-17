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
- actor: claude-code
  id: 01m062j3y5hprrnrtg4fygqzkk
  text: |
    ## Pre-registration: the deduction is VERIFIED, and E1 is INFEASIBLE as written

    ### 1. The indexer ranking is INERT at the failing step. VERIFIED by reading.

    `DeepSeekV4Indexer.callAsFunction` (Libraries/MLXLLM/Models/DeepSeekV4Indexer.swift,
    line 207) opens with:

    ```swift
    guard chunkCount > topK else {
        return broadcast(visible.expandedDimensions(axis: Self.headAxis), ...)
    }
    ```

    The published `config.json` of `mlx-community/DeepSeek-V4-Flash-4bit` states
    `index_topk` = 512 and `compress_ratios` = `[0, 0, 4, 128, 4, 128, ...]`.

    At absolute position 367 a ratio-4 layer holds `367 / 4` = 91 chunks and a
    ratio-128 layer holds 2. Both stand far below 512, thus `chunkCount > topK` is
    false, thus the guard returns EVERY visible chunk and the score path never runs.

    Three results follow:

    - The top-k ranking cannot cause this defect. Do NOT spend a run on
      `index_topk`.
    - The `wq_b`, `weights_proj` and `compressor` tensors of the indexer take NO
      part in the failing step. The whole `chunkScores` path is dead code for any
      prompt below 2048 positions.
    - The one measurement that DID run the ranking is
      `longPromptWithoutToolsRecallsAPlantedFact` at 3506 tokens, which holds 876
      chunks. That test PASSES, thus the ranking is measured good and the failing
      step does not use it.

    The compressed path at the failing step is thus: EVERY ratio-4 chunk of
    positions 0 through 363, EVERY ratio-128 chunk of positions 0 through 255, and
    the 128-token window of positions 240 through 367.

    ### 2. E1 cannot be built. The tool template is larger than the window.

    E1 asks for a tool prompt small enough that the `</｜DSML｜invoke>` syntax stays
    inside the 128-token window at the losing step. Measured against the published
    template:

    | quantity | tokens |
    | --- | ---: |
    | prompt of the reproduction | 328 |
    | last `</｜DSML｜invoke>` example ends at prompt index | 116 |
    | tail after that example | 211 |
    | decode steps before the losing step | 39 |
    | distance from the example to the losing step | 251 |
    | the window | 128 |

    The tail holds the schema and the user turn, which a test may shorten, AND the
    FIXED words of `TOOLS_TEMPLATE`, which a test may NOT shorten: the
    `</｜DSML｜tool_calls>` line, the `string="true|false"` paragraph, the
    `thinking_mode` paragraph, the `Otherwise, output directly` line, the
    `### Available Tool Schemas` heading and the `You MUST strictly follow` line.
    Those fixed words are about 131 tokens. Add the 39 decode steps and the least
    possible distance is about 170 positions, which is 42 past the window.

    Thus NO valid DeepSeek-V4 tool prompt, of any size, puts the closing-tag syntax
    inside the window at the step that loses the token. The compressed path is the
    only route the syntax has. E1 as written cannot be a control, thus it is
    replaced.

    ### 3. What replaces E1: the same question, asked with the prompt

    **E1' — put the syntax inside the window with a reminder in the user turn.**

    Hypothesis: the compressed path fails to deliver the closing-tag syntax, thus
    the identifier 5406 loses because the syntax reaches the step only through
    pooled chunks.

    Change: no code. The user turn gains a reminder that states the literal
    `</｜DSML｜invoke>` about 25 tokens before the end of the prompt, thus about 64
    positions before the losing step, thus INSIDE the window.

    - The model writes `oke` -> the syntax is reachable through raw keys and not
      through chunks. The compressed path is implicated.
    - The model still drops 5406 -> the compressed path is exonerated for THIS
      defect, because the syntax was in plain view and the model still refused it.

    ### 4. E2 is split in two, because the overlap decides the centre

    The card asks for the chunk position `c * ratio + ratio / 2`. Reading
    `DeepSeekV4Compressor.overlapped(_:padding:)` shows that this is right for one
    half of the layers and wrong for the other:

    - A ratio-4 layer POOLS WITH OVERLAP. Chunk `c` reads the tokens of chunk
      `c - 1` as well as its own, thus it covers the raw positions `4c - 4` through
      `4c + 3`, whose centre is `4c - 0.5`. The code writes `4c`. Thus on the
      overlapping layers the code ALREADY stands at the centre, and the comment of
      the source agrees with its own line.
    - A ratio-128 layer pools with NO overlap. Chunk `c` covers `128c` through
      `128c + 127`, whose centre is `128c + 63.5`. The code writes `128c`. Only
      here do the code and the comment disagree.

    Thus two runs, not one:

    - **E2a** `c * ratio + ratio / 2` on every compressed layer, which is the card's
      own wording.
    - **E2b** `c * ratio + ratio / 2` on the NON-overlapping layers alone, which is
      the one place the two authorities differ.

    Falsification for each: identifier 5406 comes back -> strong evidence. No change
    -> the chunk position is not the cause.

    ### 5. E3 keeps its place

    The rotary tables cast to bfloat16 at `DeepSeekV4MathHelpers.swift:221-222`.
    E3 does the rotation in float32 and casts the answer back. Falsification: 5406
    comes back -> the precision is the cause. No change -> it is not.

    ### 6. One load, five runs

    Each experiment is a run of the SAME loaded model with a different knob, thus
    one process loads the 141 GiB checkpoint ONE time and measures five
    configurations: `baseline`, `reminder` (E1'), `center-all` (E2a),
    `center-coarse` (E2b), `fp32-rope` (E3). Every knob is a temporary edit, and
    every temporary edit goes away before this card closes.
  timestamp: 2026-08-16T20:00:12.741612+00:00
- actor: claude-code
  id: 01m063ghbz3ywe7fwbr4x2146e
  text: |
    ## Results: one load, five configurations, plus one weight-free control

    Run of 2026-08-16, ONE load of the 141 GiB checkpoint, 243.7 s of test time.
    Log: `scratchpad/e1-e3.log`. Every knob was a temporary edit, and every
    temporary edit is now reverted.

    ### 0. The deduction is CONFIRMED by a print, not only by reading

    ```
    DSV4 EXP: indexer offset=0 queries=328 chunkWidth=4 chunkCount=82 topK=512 rankingRuns=false
    ```

    Six reports, one for each of the first six ratio-4 layers, all the same. The
    prefill of 328 tokens pools 82 chunks against a budget of 512, thus
    `chunkCount > topK` is FALSE and the guard returns every visible chunk. The
    score path of `DeepSeekV4Indexer` never runs at this prompt length. The top-k
    ranking cannot cause this defect, and no run was spent on `index_topk`.

    ### 1. The baseline reproduces the card exactly

    | identifier | logit |
    | --- | ---: |
    | `>\n` 1018 (the winner) | 28.25 |
    | `oke` 5406 (absent), rank 1 | 21.25 |
    | gap | 7.0 |

    The 47 generated identifiers match the run of 2026-08-16 in the earlier
    comment, identifier for identifier, in a NEW process. The defect is stable
    across processes as well as across runs.

    ### 2. E1' -- the compressed path is EXONERATED

    **Hypothesis**: the compressed path fails to deliver the closing-tag syntax,
    thus 5406 loses because that syntax reaches the step only through pooled
    chunks.

    **Change**: no code. The user turn gained a reminder that states the literal
    `</｜DSML｜invoke>`.

    **Measured, weight-free, by the new tokenizer probe**:

    ```
    DSV4 E1P: reminder prompt identifiers = 359
    DSV4 E1P: identifier 5406 stands at [62, 98, 103, 115, 344]
    DSV4 E1P: the losing step stands at 398
    DSV4 E1P: the window covers 271 through 398
    DSV4 E1P: the tail = [2728 "Ġ</", 128825 "｜DSML｜", 40148 "inv", 5406 "oke",
                          32 ">", 305 "Ġand", 270 "Ġthe", 5603 "Ġblock", ...]
    ```

    Thus the whole tag `</｜DSML｜invoke>` stands at prompt positions 342 to 346,
    which is 54 positions before the losing step and WELL INSIDE the 128-token
    window. Those five identifiers are RAW KEYS of the sliding window at the step
    that must write 5406.

    **Result**: the model wrote the SAME 47 identifiers as the baseline, byte for
    byte, and closed with `</｜DSML｜inv>` again.

    | identifier | logit |
    | --- | ---: |
    | `>\n` 1018 (the winner) | 33.75 |
    | `inv` 40148, rank 1 | 24.125 |
    | `oke` 5406, rank 2 | 22.25 |
    | gap | 11.5 |

    The gap did not close. It WIDENED from 7.0 to 11.5, and 5406 fell from rank 1
    to rank 2.

    **The hypothesis is REFUTED.** The model had the exact five identifiers of the
    correct tag as raw keys, 54 positions back, and refused them by a wider margin
    than before. A compressed path that failed to deliver the syntax cannot explain
    a step where the syntax was in plain view.

    ### 3. E2a and E2b -- the chunk position of this port is CORRECT

    Neither variant brings 5406 back. Both DEGRADE the answer, and the degradation
    rises with the size of the shift.

    | run | shift | the closing tags it wrote | 5406 at the losing step |
    | --- | --- | --- | --- |
    | baseline | none | `</｜DSML｜inv>` `</｜DSML｜tool_calls>` | 21.25, rank 1, gap 7.0 |
    | center-all | `+ ratio / 2` on every compressed layer | `</｜DSML｜inv>` `</｜DSML｜tool>` | 19.25, rank 5, gap 8.625 |
    | center-coarse | `+ 64` on the ratio-128 layers alone | `</｜DSML｜>` `</｜DSML｜>` | no closing tag at all |

    `center-coarse` shifts 2 chunks on 21 layers by 64 positions and the model then
    loses the whole tag name, not one piece of it. `center-all` also shifts every
    ratio-4 chunk by 2 and loses the name `tool_calls` as well.

    **Both hypotheses are REFUTED, and the refutation is POSITIVE evidence.** The
    first raw position is what the trained weights read, on the overlapping layers
    AND on the coarse layers. The comment "chunk centers" of the source is wrong;
    its code line is right, and this port follows the right one. The doc comment of
    `DeepSeekV4Compressor` that records this choice needs no change.

    ### 4. E3 -- not the cause, but it REFUTES the premise of the card

    **Change**: `rotatePairs` turns the pairs in float32 and casts the answer back,
    in place of casting the angle tables down to bfloat16 first.

    **Result**: the SAME 47 identifiers. The tag still reads `</｜DSML｜inv>`.

    | identifier | baseline | fp32-rope |
    | --- | ---: | ---: |
    | `>\n` 1018 | 28.25 | 30.0 |
    | `oke` 5406 | 21.25 | 24.625 |
    | gap | 7.0 | **5.375** |

    **The hypothesis is REFUTED as a cause.** But read the number. ONE cast of the
    rotary tables is worth 1.625 logits of the gap, and it moves the winner by 1.75
    logits as well.

    That number takes away the argument this card was built on. The card states:
    "The gap is 7.0 logits, and a quantization moves a confident logit by a fraction
    of one unit." The measurement says otherwise. If one bfloat16 cast of a
    positional table is worth 1.6 logits at this step, then 4-bit affine
    quantization of 141 GiB of weights across 43 layers is easily worth 7. The
    FIRST candidate of the card -- the checkpoint -- is back in play, and the
    reasoning that removed it does not hold.

    ### 5. What the loop leaves standing

    | candidate | state after this loop |
    | --- | --- |
    | the indexer top-k ranking | REFUTED. It never runs at this prompt length. |
    | the compressed path as the route of the syntax | REFUTED by E1'. |
    | the pooled-chunk rotary position | REFUTED by E2a and E2b, and shown CORRECT as it stands. |
    | the bfloat16 rotary tables | REFUTED as a cause. Worth 1.6 logits, thus a real numerics lead. |
    | the 4-bit quantization of the checkpoint | NOT REFUTED. The one argument against it is now measured false. |

    Nothing in the DeepSeek-V4 attention port is named by this loop. Four
    hypotheses are dead, one of them with positive evidence that the shipped code
    is right, and the quantization candidate is back.

    ### 6. What a person must decide next

    The loop reached the end of what a single machine with one checkpoint can
    answer. Two ways forward stay, and each needs a decision:

    1. **The reference run.** Install `mlx` and `mlx-lm` in Python and read whether
       the reference writes 5406 on the SAME 4-bit checkpoint. This is the one
       experiment that separates "the port" from "the checkpoint" completely, and
       it is the blocker this card already records.
    2. **A second checkpoint.** Read the same prompt on a checkpoint of a different
       quantization, e.g. 8-bit or bfloat16. If 5406 comes back, the 4-bit
       quantization is the cause and no port change can correct it. The published
       bfloat16 weights hold about 700 GiB, thus this may not be possible on this
       machine.

    Without one of the two, the card should be PARKED. Every cheap discriminator
    this branch can reach is now spent.
  timestamp: 2026-08-16T20:16:49.535879+00:00
- actor: claude-code
  id: 01m0659w3b1v5hgjt01my7g8bm
  text: |
    ### experiment loop — complete, nothing in the port is named

    - evidence: one load of the checkpoint, 243.7 s, five configurations, plus one
      weight-free tokenizer probe. Logs in the scratchpad: `e1-e3.log`, `e1p2.log`.
    - reverted: every experimental edit is gone. `git status` shows only the two
      kanban files of this card. `swift build` is green, the IntegrationTesting
      project builds, and `DeepseekV4ToolCallTokenDiagnosticTests` reports 1 test
      skipped in 0.001 s with no checkpoint load, thus the `.disabled` trait is
      back.
    - committed: NOTHING. No commit and no push, as the user directed.
    - outcome: four hypotheses REFUTED (the indexer ranking, the compressed path as
      the route of the syntax, the chunk rotary position, the bfloat16 rotary
      tables). One of the refutations is POSITIVE evidence that the shipped chunk
      position is right. The 4-bit quantization candidate is BACK, because E3
      measured that one bfloat16 cast is worth 1.6 of the 7.0 logits, which takes
      away the argument that removed the checkpoint.
    - next: a person decides between the Python reference run and a second
      checkpoint of a different quantization. Without one of the two, PARK the card.
  timestamp: 2026-08-16T20:48:08.299470+00:00
- actor: claude-code
  id: 01m07rdg52hdzk4zf0nayt3xg3
  text: |
    ### finish iteration 2 — stuck (infrastructure, not the work)

    - implement: no-change. The sub agent read the card and the comments, started to
      read `QuantizedLinear` and `Module.update`, and then stopped to make progress.
      Its transcript did not grow for approximately 30 minutes.
    - Measured, not assumed: `ps` showed NO `xctest` process, `vm.swapusage` showed
      0.00M used, and the system reported 98% memory free. Thus no checkpoint load
      was in progress and the machine was not in difficulty. The agent was wedged.
    - The orchestrator stopped the agent. `git status` is empty and the tree is
      clean, thus the wedged agent left nothing behind.
    - test: not run.
    - commit: none.
    - review: not run.

    ## The planned experiment stays valid

    It was never attempted. Record it here so the next agent does not design it
    again:

    **Hypothesis**: the 4-bit quantization of the checkpoint loses identifier 5406.

    **Method**: raise precision selectively inside the checkpoint already on disk,
    and read the gap at each step.
    1. Dequantize the output head to float32. Read the gap.
    2. Extend to the last N decoder layers, with N rising.
    3. Compute the memory BEFORE a full dequantization. The 4-bit checkpoint holds
       141 GiB and float32 is approximately 8 times that, thus a full float32 model
       does not fit in the 512 GB of this machine. Do not attempt it.

    **Falsification**: a gap that closes as precision rises confirms the
    quantization. A gap that does not move refutes it.

    This experiment needs NO Python, NO install and NO second checkpoint. The
    blocker this card recorded earlier is not correct.
  timestamp: 2026-08-17T11:41:24.514012+00:00
- actor: claude-code
  id: 01m07s47a4zz3g1t5bmcbh1bnm
  text: |-
    ## The card is rewritten. The new purpose: make tool calling WORK.

    The user decided that the "port or checkpoint" question does not need an answer,
    because no port change corrects it either way. The whole investigation stays as
    recorded history in the description and in this thread. No measurement is
    deleted.

    ## New evidence: I read the published tokenizer

    Source: `~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-4bit/`
    `snapshots/38c0bd20a6fba70f22c5ee2940ec0092b36ab936/tokenizer.json`.

    Identifier to piece:

    | identifier | piece |
    | ---: | --- |
    | 1718 | `</` |
    | 40148 | `inv` |
    | 5406 | `oke` |
    | 32 | `>` |
    | 1018 | `>\n` |
    | 41523 | `parameter` |
    | 72461 | `tool` |
    | 4941 | `_c` |
    | 12548 | `alls` |

    Which prefixes of each tag name the vocabulary holds:

    - `invoke`: `i` = 75, `in` = 261, `inv` = 40148. NO `invo`, NO `invok`, NO
      `invoke`.
    - `tool_calls`: `t` = 86, `to` = 1495, `too` = 56255, `tool` = 72461. Nothing
      longer.
    - `parameter`: `p` = 82, `pa` = 6256, `par` = 1789, `para` = 35638,
      `param` = 5494. Nothing longer, but the WHOLE word is one token, 41523.

    ## The rule I chose, and why

    **Accept a closing tag whose name is a prefix of the published name, cut at a
    token boundary of the published tokenization of that name.**

    - For `invoke` the tokenization is `inv` + `oke`, thus the rule gives exactly
      ONE extra literal: `</｜DSML｜inv>`. That is exactly the measured string. The
      two rules the work statement offered — "any non-empty prefix" and "the exact
      measured string" — agree here, because the token boundary is the only place
      the word can break.
    - The rule stops at the token boundary on purpose. `</｜DSML｜in>` is a legal
      output of a sampler, because `in` is token 261. It is NOT a lost piece of the
      published tokenization; it is a different word, and no run ever wrote it. The
      parser refuses it. The parser accepts TWO literals and no more.

    ## The sibling tags: the same rule, and it changes nothing

    - `parameter` is ONE token. The rule gives an empty set. The tag cannot lose a
      piece at a token boundary, and every measured run writes it whole.
    - `tool_calls` is three tokens, thus the rule gives `</｜DSML｜tool>` and
      `</｜DSML｜tool_c>`. The `center-all` experiment DID write `</｜DSML｜tool>`,
      thus the loss mode reaches this tag. But `extractToolCalls` never reads the
      block closing tag — it walks the invoke elements alone. The only reader is the
      `endTag` property, which `ToolCallProcessor` uses to flush its streaming
      buffer, and this card forbids loosening the processor. When the block closing
      tag is short, the processor holds the buffer to the end of generation and
      `processEOS` calls `parser.parseEOS(buffered)`, which recovers every call.
      Thus a short block closing tag already completes a round. A new weight-free
      test proves it end to end.

    ## What the disabled reproduction becomes

    `DeepseekV4ToolCallTokenDiagnosticTests.theGreedyRunWritesTheWholeInvokeClosingTag`
    asserts the model writes the WHOLE tag. That assertion is now known false and
    permanently so, thus it must not stay as an open defect. It becomes an ENABLED
    record of the accepted divergence: it keeps every measurement print, it keeps
    the determinism assertion, and it asserts that the generated text carries a
    closing tag the DSML parser ACCEPTS. It thus stays green whether the checkpoint
    writes the short tag or the whole one.

    Next: the tests first, RED, then the parser.
  timestamp: 2026-08-17T11:53:49.124499+00:00
- actor: claude-code
  id: 01m07sv9dm5778q4q52236rpj5
  text: |-
    ## The parser change is IN, and the weight-free suites are GREEN

    ### The tests came first, and they were RED

    Run of `MLXLMTests` before the parser change:

    ```
    ✘ Test "the short closing tag of the checkpoint gives the call of the whole tag"
      recorded an issue at DSMLToolCallParserTests.swift:309:25: Expectation failed:
      parser.parse(content: Self.stockCall(closingInvoke: Self.shortInvokeCloseTag), tools: nil)
    ✔ Test "a closing tag that no lost piece explains is refused" passed
    ✘ Test "a round completes although the model shortens both closing tags"
      failed after 0.001 seconds with 3 issues.
    ✘ Test run with 851 tests in 71 suites failed after 84.308 seconds with 4 issues.
    ```

    The refusal test passed before the change and passes after it, which is correct:
    the parser refused those tags then, and it refuses them now.

    ### The parser change

    `Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift`:

    - `shortInvokeClose` = `</｜DSML｜inv>`, a new private literal.
    - `invokeCloseTags` = `[invokeClose, shortInvokeClose]`, a table the lookup
      walks. It is a table and not an `if` chain, thus `data-driven` is satisfied.
    - `invokeCloseRange(in:from:)`, a new private helper. It reads each accepted tag
      and keeps the range that comes FIRST. No accepted tag is a substring of
      another — `</｜DSML｜invoke>` holds `inv` but never `inv>` — thus well-formed
      text always answers with `invokeClose` and the well-formed path is unchanged.
    - `extractToolCalls` calls the helper in place of the one literal search.

    Nothing else changed. `parameterClose` and `endTag` are untouched, no other
    tool-call format is touched, and no `ToolCallProcessor` path is touched.

    ### The WHY is in the code

    The type documentation of `DSMLToolCallParser` now carries a `## Tolerance`
    section that names card `^z5xrzg6`, states that the checkpoint drops identifier
    5406 deterministically with a 7.0 logit gap that repeats across runs AND
    processes, states that this is a DELIBERATE divergence from the published
    `parse_tool_calls` which accepts one syntax and raises on all others, and holds
    the token-boundary table for the three tag names. It ends with "Do NOT 'correct'
    it back before you read card `^z5xrzg6`."

    ### Four weight-free tests

    In `Tests/MLXLMTests/DSMLToolCallParserTests.swift`, section
    "The short closing tag of the checkpoint":

    1. `theShortInvokeCloseTagGivesTheCallOfTheWholeTag` — the short tag and the
       whole tag give equal `ToolCall`s, the same name, the same argument and the
       same `argumentsJSON`. The JSON text needs its own comparison, because
       `ToolCall` equality reads the name and the arguments alone.
    2. `aCallReadFromTheShortTagRendersAgainWithTheWholeTag` — a call read from the
       short tag renders again with the WHOLE tag. This locks the contract that the
       tolerance belongs to the READ side alone, which is what
       `inverse-operation-coverage` asks for.
    3. `aClosingTagThatNoLostPieceExplainsIsRefused` — `</｜DSML｜i>`,
       `</｜DSML｜in>`, `</｜DSML｜invok>`, `</｜DSML｜>` and no tag at all are each
       REFUSED.
    4. `aRoundCompletesAlthoughBothClosingTagsAreShort` — the streaming path
       completes a round when the model shortens BOTH tags. This proves the decision
       on the sibling tag `tool_calls`: the processor never sees `endTag`, thus it
       holds the buffer to the end of generation and `processEOS` calls
       `parser.parseEOS(buffered)`, which recovers the call.

    ### The reproduction suite is ENABLED again

    `IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift`:

    - The `.disabled(...)` trait is GONE.
    - The test is renamed `theGreedyRunWritesAClosingTagTheParserReads`.
    - The determinism assertion is unchanged.
    - The second assertion changed. It read "the model must write the whole closing
      tag", which is now known FALSE and permanently so, thus no edit could satisfy
      it. It now reads: the answer must carry a closing tag the DSML parser reads,
      measured by `DSMLToolCallParser().parseEOS(text, tools: nil)` and the name of
      the tool. That is NOT a weakened assertion under `no-test-cheating` check 4: it
      asserts the end-to-end property the round needs — one call, correctly named —
      in place of an exact identifier subsequence, and it holds whether the
      checkpoint writes the short tag or the whole one.
    - Every measurement `print` stays, and two new lines report which of the two tags
      the answer holds.
    - Two stale sentences of the file header are corrected. They stated that a gap of
      a few hundredths of a logit is what quantization does and a gap of several
      logits is not — the E3 experiment measured that one bfloat16 cast is worth 1.6
      logits, thus that reasoning does not hold.

    ### Two rules I read and answered rather than skipped

    - `case-sensitivity-coverage`: the diff does add a token match. The DSML tag
      names are case-free in practice, because the accepted literal is built from the
      vocabulary pieces `inv` (40148) and the marker (128825), which are lowercase
      entries. A different case is a DIFFERENT token sequence that no run wrote, and
      the tag matches of this parser were already case-sensitive before this change
      and stay so. Thus there is no case to cover.
    - `no-hard-code` check 4: the literal comes from the published `tokenizer.json`
      of the checkpoint, not from a test fixture. The carve-outs "Constants that are
      genuinely constant" and "Lookup tables that are correct for all inputs" hold.

    ### Evidence

    Both projects build:

    - `xcodebuild build-for-testing -skipPackagePluginValidation -scheme mlx-swift-lm-Package -destination 'platform=macOS'` -> TEST BUILD SUCCEEDED, no warning of this change.
    - `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` -> TEST BUILD SUCCEEDED.

    Five bundles, exit 0 for each:

    | bundle | result |
    | --- | --- |
    | `MLXLMTests` | 852 Swift Testing passed; 475 XCTest, 0 failures |
    | `MLXGuidedGenerationTests` | 70 passed |
    | `MLXFoundationModelsTests` | 160 passed |
    | `CXGrammarTests` | 7 passed |
    | `MLXHuggingFaceMacrosTests` | 5 passed, 0 failures |

    `swift-format` ran over the three changed files. It also wanted to reformat eight
    files this card never touched (the MiniMax-M3 files, `DeepSeekV4Attention.swift`
    and `ToolTests.swift`); those are reverted, because an unrelated reformat is not
    this card's work.

    Next: the real-weights runs. One load for each process.
  timestamp: 2026-08-17T12:06:24.948785+00:00
- actor: claude-code
  id: 01m07vemv4mf9td4v6gy5xydrn
  text: |-
    ## The real weights: ONE TOOL ROUND NOW COMPLETES

    ### Run 1 — the criterion that matters. One load, 65.8 s, PASSED

    `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' "-only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests/aShortToolPromptEmitsOneDSMLToolCall()"`

    ```
    Tool round calls: [ToolCall(function: Function(
        name: "get_stock_level",
        arguments: ["bay": .string("bay 7")],
        argumentsJSON: Optional("{\"bay\": \"bay 7\"}")),
        id: Optional("call_6411b241fd97461aaef6d4ec8d31fb1d"))]
    ✔ Test aShortToolPromptEmitsOneDSMLToolCall() passed after 65.823 seconds.
    ** TEST SUCCEEDED **
    ```

    The model wrote the short closing tag, the parser read it, and the round
    completed with the right tool name and the right argument. That criterion was
    open since card `^2dvj1g6` handed it over. It is closed.

    Note the filter: `-only-testing` needs the `()` of the Swift Testing identifier.
    Without it the run reported "0 tests in 1 suite passed" and exit 0 — a silent
    no-op that looks like a pass. Record that for the next agent.

    ### Run 2 — the re-enabled record. One load, 272.8 s, PASSED

    `-only-testing:IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests`

    ```
    DSV4 TOKEN: prompt identifiers = 328
    DSV4 TOKEN: closing tag identifiers = [1718, 128825, 40148, 5406, 32]
    DSV4 TOKEN: the closing tag opens at step 36
    DSV4 TOKEN: step 39 of run 1: chose 1018; candidates [1018 at 28.25, 5406 at 21.25,
      19 at 20.5, 32 at 20.25, 1 at 19.5]; tracked identifier at logit 21.25, rank 1,
      7.0 below the winner
    DSV4 TOKEN: step 39 of run 2: (identical)
    DSV4 TOKEN: the answer holds </｜DSML｜invoke>: false
    DSV4 TOKEN: the answer holds </｜DSML｜inv>: true
    ✔ Test theGreedyRunWritesAClosingTagTheParserReads() passed after 272.848 seconds.
    ```

    This is a THIRD process, and it writes the same 47 identifiers and the same 7.0
    gap. The divergence is now recorded by an ENABLED test rather than hidden behind
    a `.disabled` trait.

    ### Run 3 — the two prompt-cache tool rounds. One load, 1034 s

    Both got PAST `#require(roundOne.toolCalls.first)`, which is the criterion this
    card carried:

    ```
    DSV4 CACHE: chat round 1 tool calls = ["get_stock_level"]
    DSV4 CACHE: thinking round 1 tool calls = ["get_stock_level"]
    ```

    Both then FAIL on the prompt-cache assertions that stand after that point, which
    NO run could reach before today:

    | quantity | chat | thinking |
    | --- | ---: | ---: |
    | tool round rendered / fed tokens | 3906 / 3906 | 3918 / 3918 |
    | tool round tokens skipped by reuse | 0 | 0 |
    | tool round prefill s / cold control s | 15.71 / 15.63 | 18.14 / 21.39 |
    | follow-up round tokens skipped by reuse | 3951 | 0 |

    Fact (a) PASSES in both modes — `tool round extends round 1 = true`, common
    prefix 3814 of 3814 — thus the seam is right and `ExtendCachedPrefixRule` is
    reachable. Fact (b) fails: the tool round feeds every token and takes as long as
    a cold session.

    That is NEW work and NOT the tolerance this card owns. Card `^v7z7v99`
    ("DeepSeek-V4: the tool round reuses NO prompt cache, although its render
    extends round 1 whole") holds it, with every number above and the one clue the
    numbers carry: chat mode's FOLLOW-UP round DOES reuse 3951 of 3967 tokens in
    2.69 s, thus the reuse machinery works and only the tool round resets it.

    These two tests were already failing before this card, at the `#require`. This
    change moved the failure later, from "no tool call at all" to "the cache saves no
    work". Neither test is in the five bundles and neither runs in CI.

    ### One load for each process

    Four processes, four loads, never two in one process. `sysctl vm.swapusage`
    reported 0.00M used before the first run and no `xctest` was live.
  timestamp: 2026-08-17T12:34:27.812984+00:00
- actor: claude-code
  id: 01m07vf8zfyreq185j4zfgxscb
  text: |-
    ### implement — changed
    - evidence: 3 source files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Libraries/MLXLMCommon/Tool/Parsers/DSMLToolCallParser.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/Tests/MLXLMTests/DSMLToolCallParserTests.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests.swift. Tests written FIRST and proved RED (2 of 3 failed before the parser change; the refusal test passed then and passes now). Both projects build. Five bundles green: MLXLMTests 852 Swift Testing + 475 XCTest/0 failures, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 160, CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. Real weights, one load for each of four processes: `aShortToolPromptEmitsOneDSMLToolCall` PASSED in 65.8 s with the call `get_stock_level(bay: "bay 7")`; `theGreedyRunWritesAClosingTagTheParserReads` PASSED in 272.8 s and records the 7.0 gap and `the answer holds </｜DSML｜inv>: true`; the two prompt-cache tool rounds both got past `#require(roundOne.toolCalls.first)` with `round 1 tool calls = ["get_stock_level"]`. Every acceptance criterion of the card is checked. `swift-format lint` clean on the three files.
    - next: /review. New work found and carded: `^v7z7v99` holds the prompt-cache failure the completed tool round newly exposes.
  timestamp: 2026-08-17T12:34:48.431111+00:00
position_column: doing
position_ordinal: '80'
title: 'Make DeepSeek-V4 tool calling work: the DSML parser accepts the short closing tag `</｜DSML｜inv>`'
---
## The purpose

Make DeepSeek-V4 tool calling WORK, although the checkpoint drops one token.

This card started as a question: is the port or the checkpoint at fault? The
user decided that the question does not need an answer, because NO port change
corrects it either way. The user's words: "we need tool calling before we even
try to use this model." The investigation stays below as recorded history,
because it is what justifies this decision.

## The defect, in one line

With the real weights and greedy decoding the model deterministically writes the
closing tag `</｜DSML｜inv>` where the DSML syntax states `</｜DSML｜invoke>`.
Identifier 5406 (`oke`) is never generated. `DSMLToolCallParser` refused the
payload, thus no tool round completed.

```
<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_stock_level">
<｜DSML｜parameter name="bay" string="true">bay 7</｜DSML｜parameter>
</｜DSML｜inv>
</｜DSML｜tool_calls>
```

## What the parser accepts, and why

The rule: **the parser accepts a closing tag whose name is a prefix of the
published name, cut at a token boundary of the published tokenization of that
name.** Nothing else. The parser does NOT accept arbitrary text.

The published `tokenizer.json` of `mlx-community/DeepSeek-V4-Flash-4bit` gives
the pieces of each DSML tag name:

| tag name | published pieces | prefixes at a token boundary |
| --- | --- | --- |
| `invoke` | `inv` (40148), `oke` (5406) | `inv` |
| `tool_calls` | `tool` (72461), `_c` (4941), `alls` (12548) | `tool`, `tool_c` |
| `parameter` | `parameter` (41523) | none |

Thus for `invoke` the rule gives ONE extra literal, `</｜DSML｜inv>`, which is
exactly the string the real weights write. The two defensible rules the work
statement named — "any non-empty prefix" and "only the exact measured string" —
give the same answer here, because the token boundary is the only place the word
can break.

The rule stops at the token boundary on purpose. The vocabulary also holds `i`
(75) and `in` (261), thus `</｜DSML｜in>` is a legal output of a sampler. It is
NOT a lost piece of the published tokenization; it is a different word. No run
ever wrote it. The parser refuses it.

## The sibling tags

The same rule applies to `</｜DSML｜parameter>` and `</｜DSML｜tool_calls>`. It
changes nothing for either one, for two different measured reasons:

- **`parameter`**: the name is ONE token (41523). The rule gives an empty set,
  thus the tag cannot lose a piece at a token boundary. The measured stream
  writes it whole in every run: `1718, 128825, 41523, 1018`.
- **`tool_calls`**: the rule gives `</｜DSML｜tool>` and `</｜DSML｜tool_c>`, and
  the `center-all` experiment DID measure `</｜DSML｜tool>` under a perturbed
  rotary position, thus the loss mode reaches this tag. But the parser never
  READS this tag when it walks the calls: `extractToolCalls` finds each invoke
  element on its own. The only reader of the block closing tag is the `endTag`
  property, which `ToolCallProcessor` uses to know when to flush its buffer
  during streaming, and this card scopes the change to the parser. When the
  block closing tag is short, the processor holds the buffer to the end of
  generation and `parseEOS` recovers every call from it. Thus a short block
  closing tag ALREADY completes a round, and
  `aRoundCompletesAlthoughBothClosingTagsAreShort` proves it.

## Acceptance criteria

- [x] The DSML parser accepts `</｜DSML｜inv>` and gives the same `ToolCall` as
      `</｜DSML｜invoke>`
- [x] The whole tag `</｜DSML｜invoke>` parses exactly as it does today
- [x] A closing tag that is not one of the accepted literals is REFUSED
- [x] The tolerance is in `DSMLToolCallParser` alone. No other tool-call format
      and no general `ToolCallProcessor` path is loosened
- [x] The code states WHY: card `^z5xrzg6`, identifier 5406, the 7.0 logit gap,
      and the deliberate divergence from the published `parse_tool_calls`, which
      accepts one syntax and raises on all others
- [x] Make one tool round complete:
      `DeepseekV4IntegrationTests.aShortToolPromptEmitsOneDSMLToolCall` passes on
      the real weights
- [x] `DeepseekV4ToolCallTokenDiagnosticTests` is ENABLED again and records the
      accepted divergence rather than reporting it as an open defect
- [x] `chatModeToolRoundReusesThePromptCache` and
      `thinkingModeToolRoundReusesThePromptCache` both get past
      `#require(roundOne.toolCalls.first)`. Both report
      `round 1 tool calls = ["get_stock_level"]`. The prompt-cache assertions
      that stand AFTER that point are now reachable and they fail; that is new
      work, and card `^v7z7v99` holds it.

## Recorded history: what the investigation measured

Every measurement below is ESTABLISHED and must NOT be measured again. The
comment thread of this card holds the full reports.

- The loss is deterministic across runs AND across processes. Three processes now
  report the same 47 identifiers.
- At the losing step the winner `>\n` (1018) stands at logit 28.25 and `oke`
  (5406) at 21.25 — a gap of 7.0, rank 1.
- The opening tag `<｜DSML｜invoke name="...">` is written WHOLE, thus the model
  writes the word `invoke` correctly when it opens the element.
- `</｜DSML｜tool_calls>` and `</｜DSML｜parameter>` are written WHOLE in every
  run measured.
- The text path of this library is proven clean:
  `DeepSeekV4TokenizerIntegrationTests.aToolCallStreamsBackWholeAcrossItsNewlines`
  streams the correct answer back byte for byte, `NaiveStreamingDetokenizer`
  cannot delete a tail, `StopStringFilter` is off, `ToolCallProcessor` gives the
  parser a contiguous slice, and the multi-token-prediction head never loads.
  Thus identifier 5406 never entered the stream.

### Four hypotheses are REFUTED

| hypothesis | how it was refuted |
| --- | --- |
| the indexer top-k ranking | It never runs at this prompt length. 82 chunks against a budget of 512, thus the guard returns every visible chunk. |
| the compressed path as the route of the syntax | E1': a reminder put the whole tag as RAW keys 54 positions before the losing step. The model refused 5406 by a WIDER margin (11.5). |
| the pooled-chunk rotary position | E2a and E2b: both shifts DEGRADE the answer. This is positive evidence that the shipped first-raw-position choice is right. |
| the bfloat16 rotary tables | E3: float32 rotation gives the same 47 identifiers. It moves the gap from 7.0 to 5.375, thus one cast is worth 1.6 logits — a real numerics lead, and the reason the "quantization moves a logit by a fraction of one unit" argument does not hold. |

The port matches its only reference. Nothing in the DeepSeek-V4 attention port
is named by the investigation.

## Memory

The checkpoint holds 141 GiB. Run ONE real-weights test for each process.
#deepseek-v4