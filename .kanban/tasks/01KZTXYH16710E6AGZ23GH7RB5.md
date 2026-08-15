---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzxjab9ss39p61pc09fzy61t
  text: |-
    Picked up. Profile complete. Root cause found and a fix measured.

    ## The profile (M3 Ultra, 512 GiB, release build, `mlx-community/DeepSeek-V4-Flash-4bit`, 141 GB on disk, 43 layers)

    Method: load the checkpoint straight through `LLMTypeRegistry` + `loadWeights` (no tokenizer), then time one decode step and each stage of it. MLX is lazy, thus every stage measurement adds its result into the returned array; a sweep that only overwrote a variable measured the last layer alone and gave wrong numbers.

    One decode step, 64-token context, steady state: **2.06-2.30 s**. This matches the 2.4-3.3 s/token the card records.

    Stage split of one decode step, all 43 layers, one `eval`:

    | stage | time | share |
    |---|---|---|
    | routed experts (`ffn.switch_mlp`, gathered mxfp4 over 256 experts) | **2039 ms** | **99%** |
    | hyper-connections (`hc_attn` + `hc_ffn`, 20 Sinkhorn steps each) | 21.8 ms | 1% |
    | attention | 14.0 ms | <1% |
    | shared expert | 2.4 ms | <1% |

    Two control measurements name the cause:

    - The SAME layer chained 43 times, which has the same operation count and the same graph depth but only one layer's weights: **68 ms**, that is 30 times faster than 43 different layers (2060 ms). Thus the cost is not the number of operations.
    - One layer alone, 43 separate evals: 2.2 ms each, 96 ms in total.

    Arithmetic and memory bandwidth cannot explain 2 s. One token reads about 6.5 GB of 4-bit weights, which is 8 ms at 800 GB/s. The measured 2039 ms over the 147 GB the routed-expert tensors hold gives 72 GB/s, and 85% of the run is SYSTEM time.

    ## Root cause

    MLX gives its Metal residency set a capacity of ZERO unless the process raises the wired limit (`wired_limit_{0}` in `mlx/backend/metal/allocator.h`), and it takes only buffers of 1 MB or less from the one heap that set holds (`heap_size_ = 1 << 20`). Every weight tensor of this checkpoint is larger than 1 MB, and each routed-expert tensor is about 1.07 GB. Thus every weight buffer sits OUTSIDE the residency set, and Metal makes all 141 GB resident again for each command buffer. One command buffer is one decode step, thus each token pays for the whole checkpoint. The 284B-total MoE shape makes the effect large, because 147 GB of the 141 GB total is routed-expert weight that one token barely reads (6 of 256 experts).

    ## The fix, measured

    Raise the wired limit BEFORE the weights are allocated:

    - default limit: **2087, 2103, 2087, 2095, 2055, 2126, 2061, 2096 ms** per decode step
    - limit raised before the load: **73.7, 69.6, 68.2, 68.1, 67.3, 70.0, 66.0, 67.7 ms** per decode step

    That is **31 times faster**, about 68 ms per token.

    The ORDER matters. A limit raised AFTER the load changes nothing (measured 2040-2357 ms per step), because a buffer joins the residency set when it is made. The ticket must also never end, because a limit that falls again empties the set.

    ## What this means for the >12k test

    At 68 ms/token, 12,400 tokens take about 14 minutes, plus 22 s of load. The 240-minute per-test limit holds with a wide margin. Criterion 2 is met by a fix, not by a decision to drop the test.

    Note for the next agent: the Python reference (mlx-lm PR 1189) is slow for the same reason. `mlx_lm` does not call `mx.set_wired_limit` either. The cause is not in the Swift port.
  timestamp: 2026-08-13T12:42:25.466003+00:00
- actor: claude-code
  id: 01kzxjxe5rq56x2g6fd4bky74k
  text: |-
    Fix landed and verified end to end.

    ## The change

    - `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift` — a new `raiseWiredMemoryLimit()` starts a `WiredMemoryTicket` sized to `deepseekV4RequiredMemoryBytes` (clamped to `GPU.maxRecommendedWorkingSetBytes()`) INSIDE the shared load task, before `LLMModelFactory.shared.loadContainer`. The ticket never ends. The doc comment states the mechanism and both measurements.
    - `docs/deepseek-v4-support.md` — a new "Decode performance" section holds the stage split, the two control measurements, and the fix. Deferred item 5 now says NOT to start a decode-speed task with the fused mxfp4 kernels, because the profile shows the arithmetic is not the cost.

    ## Measured, before and after

    The suite reported `DeepSeek-V4 wired limit: asked 171798691840 bytes, applied 171798691840 bytes`, thus the limit takes hold.

    | test | before (card) | after |
    |---|---|---|
    | `chatAndThinkingModesBothGenerate` (2 x 48 tokens) | 232 s | 36.8 s |

    Whole suite, minus the >12k test, one process, six tests, ALL PASS in 72.3 s:

    - `loadsTheRealCheckpointEndToEnd` 7.3 s (this test carries the weight load)
    - `greedyFirstTokensMatchThePythonFixture` 19.2 s — the 32-token Python parity still matches token for token
    - `chatAndThinkingModesBothGenerate` 36.8 s
    - `twoRoundConversationRecallsTheFirstRound` 9.0 s for 120 tokens, that is **74.8 ms/token** in the steady state
    - the two encoder cache tests 0.001 s each

    Command:

    ```
    xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
      -scheme IntegrationTesting -destination 'platform=macOS' \
      -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests \
      -skip-testing:"IntegrationTestingTests/DeepseekV4IntegrationTests/longGenerationPastTwelveThousandTokensCompletes()"
    ```

    Also green: `swift test --filter DeepSeekV4` — 111 tests in 13 suites, which includes `DeepSeekV4DocsTests` against the changed document. The integration target builds with no warning.

    ## Criterion 2, answered

    At the measured 74.8 ms/token through the full `generate` path, 12,400 tokens take about **15.5 minutes**, plus the load. The per-test limit is 240 minutes, thus `longGenerationPastTwelveThousandTokensCompletes` now fits with a wide margin, and the test needs no change of its own. The user skipped the endurance run itself for this card, thus it is not measured here.

    ## The >12k run was NOT started

    The user skipped that criterion. The projection above rests on the 74.8 ms/token of the 120-token two-round test and on the 68 ms/token of eight consecutive direct decode steps. Attention over a 12k context adds about 1.4 ms/token, and the KV cache of the run holds about 1.1 GB, which stays inside the 160 GB wired limit beside the 141 GB of weights.

    ## New work raised

    Card `^4kv2h0q` — "Wired memory: a ticket taken at generate time is too late, thus every large model pays a per-token residency cost". The library gap is real and it is a design decision for a person: `MLXLMCommon.generate(... wiredMemoryTicket:)` takes the ticket AFTER the load, and the measurement shows such a ticket gives no residency benefit. No load path of this repository raises the limit, thus every consumer of every large model pays this cost. I did NOT change the library here, because that changes global process state for every consumer.

    ## Dead ends, so that the next agent does not repeat them

    - The fused mxfp4 kernels of `docs/deepseek-v4-support.md` item 5 are NOT the answer for decode. The routed-expert gather over all 43 layers costs 1.04 s of arithmetic when the buffers are hot, and 2.04 s when they are cold.
    - `MLX.Memory.withWiredLimit(_:_:)` in its SYNCHRONOUS form is a documented no-op. The first run of this experiment used it and measured no change. Use the `async` overload, or a `WiredMemoryTicket` directly.
    - MLX is lazy. A stage sweep that writes `value = stage(...)` in a loop measures the LAST stage alone. Each result must join the returned array (`value = value + stage(...)`).

    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/docs/deepseek-v4-support.md. Decode 2.10 s/token -> 0.068 s/token (31x). Suite minus the >12k test: 6 tests pass in 72.3 s. `swift test --filter DeepSeekV4`: 111 tests pass.
    - next: /review
  timestamp: 2026-08-13T12:52:51.000582+00:00
- actor: claude-code
  id: 01kzxkqqb81msbaegqthcrat18
  text: |
    ### review — findings
    - scope: commit f389c22, range HEAD~1..HEAD
    - engine: `review sha HEAD~1..HEAD` gave "Nothing in scope to review", counts findings 0 / attempted 0. The engine does not read test files or Markdown, and this commit changes only those two kinds. Thus the result is not clean, because nobody looked.
    - mutation proof: put a type error at DeepseekV4IntegrationTests.swift:137. `swift build --build-tests` gave exit 0 (blind). `xcodebuild build-for-testing -scheme IntegrationTesting` gave exit 65 with "cannot convert value of type 'String' to specified type 'Int'" (caught). Put the file back with `git checkout --`; the same xcodebuild gave exit 0, "TEST BUILD SUCCEEDED". The tree is clean.
    - evidence: 7 findings — DeepseekV4IntegrationTests.swift:122, :127-128, :134, :134-140, :135, :136, :138-139; docs/deepseek-v4-support.md:134
    - next: the task stays in `review`. Do the 7 findings, then review again.
  timestamp: 2026-08-13T13:07:12.360020+00:00
- actor: claude-code
  id: 01kzxkrqbcw8b76y4faesm7w3z
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 2 files. Decode step 2.10 s to 0.068 s (31x). The cause is Metal residency, not arithmetic: the routed experts took 2039 ms of 2060 ms, but the same layer 43 times took 68 ms. The fix raises the wired limit before the weight load.
    - test: green — `swift test`, 535 tests, 0 failures, 2 skips that were there before.
    - commit: f389c22 fix(mlx-lm): raise the Metal wired limit before DeepSeek-V4 weight load
    - review: findings — the engine attempted 0 files, thus it is not a clean verdict. A mutation proof shows `swift build --build-tests` does not compile IntegrationTesting/ (exit 0 with a type error in it), but `xcodebuild build-for-testing -scheme IntegrationTesting` catches it (exit 65). 7 findings: DeepseekV4IntegrationTests.swift:122, :127-128, :134, :134-140, :135, :136, :138-139; docs/deepseek-v4-support.md:134.
  timestamp: 2026-08-13T13:07:45.132182+00:00
- actor: claude-code
  id: 01kzxpfzw787rv89mdby2jgxqz
  text: |
    All 7 review findings closed. Two measurements changed what the card says.

    ## Finding by finding

    1. `:134` build evidence — the file header now records that `swift test` is blind to this directory, with the measured proof (a type error gives exit 0 from `swift build --build-tests` and exit 65 from `xcodebuild build-for-testing`), and it names the xcodebuild command as the compile evidence for this file. `docs/deepseek-v4-support.md` records the same.
    2. `:134-140` no assertion — two new tests. `wiredMemoryLimitCoversTheWholeCheckpoint` asserts the applied limit and that the limit covers the whole checkpoint. `decodeStepStaysInsideTheLongGenerationBudget` measures 16 steady decode steps and asserts the median against a bound taken from the per-test time limit.
    3. `:138-139` unchecked `applied` — `raiseWiredMemoryLimit()` now returns a `WiredMemoryOutcome` with `requestedBytes`, `appliedBytes` and `isFullyApplied`. The function prints a warning when the apply is short, and the new test fails on it.
    4. `:135` silent guard — the guard prints why the limit was not raised.
    5. `:136` the skip gate reused — the wired limit has its own constants: `deepseekV4CheckpointBytes` + `deepseekV4WiredHeadroomBytes` = `deepseekV4WiredMemoryBytes` (168,662,344,796 bytes). `deepseekV4RequiredMemoryBytes` stays the skip gate, and its doc says so.
    6. `:127-128` "147 GB of the 141 GB total" — the two numbers were decimal GB against GiB. Measured on the local snapshot: the weight files hold 151,482,475,612 bytes (141 GiB), and the routed experts hold 147,169,738,752 bytes (137 GiB), which is 97%. The file now says "137 GiB of the 141 GiB total".
    7. `:122` and `docs:134` "141 GB" against "91 GB" — the 91 GB figure was wrong. The measured size is 141 GiB. Every size statement of both files now uses that one measurement, and the header states the byte count beside it.

    ## Two measurements that correct the card

    - The steady decode step through this suite is **0.593 s**, not 0.068 s. The 0.068 s came from a release build with a direct model call. The suite builds for debug and decodes through `TokenIterator`. Both numbers are now in the file and in the document, each with the build it came from.
    - The earlier "74.8 ms/token" of the two-round test assumed 120 generated tokens. `generate` stops at the end-of-sentence token, thus that turn generated far fewer tokens and the per-token number was too low.

    Thus the 12,400-token run projects to about **123 minutes**, not 15.5 minutes. It is still inside the 240-minute per-test limit, and the criterion holds.

    ## Mutation proof of the new tests

    The production line `let wiredMemory = await raiseWiredMemoryLimit()` was replaced by `let wiredMemory: WiredMemoryOutcome? = nil`, which is the change the finding names.

    - RED: both new tests failed. `wiredMemoryLimitCoversTheWholeCheckpoint` recorded "the load must raise the Metal wired limit before it allocates a weight". `decodeStepStaysInsideTheLongGenerationBudget` measured a median of 2.1247 s against the 0.8710 s bound. Exit 65.
    - The line was put back. GREEN: both tests pass, wired limit asked 168,662,344,796 bytes and applied the same, median decode step 0.5917 s. Exit 0.

    ## Notes for the next agent

    - Do NOT assert the mean of the first steps. The first `next()` call runs the whole prompt; the test drops it and takes the median of the 16 steps that follow.
    - Do NOT put a value that holds the model container inside `#require`. `try #require(load.wiredMemory, ...)` printed the whole `ModelContext` on failure, which is 8 MB of output. The test uses `guard ... else { Issue.record(...) }` instead.
    - Do not pipe an xcodebuild run into `grep` when you need the output later. Redirect to a file.
  timestamp: 2026-08-13T13:55:24.679528+00:00
- actor: claude-code
  id: 01kzxpg7ja7fhgcwemp99enp40
  text: |
    ### implement — changed
    - evidence: 2 files — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift, /Users/wballard/github/swissarmyhammer/mlx-swift-lm/docs/deepseek-v4-support.md. All 7 review findings closed. `xcodebuild build-for-testing -scheme IntegrationTesting`: exit 0, no warning. `xcodebuild test -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests -skip-testing:.../longGenerationPastTwelveThousandTokensCompletes()`: 8 tests pass in 100.4 s. Mutation proof of the two new tests: exit 65 with the wired-limit call removed (median step 2.1247 s against the 0.8710 s bound), exit 0 with it back (median 0.5917 s). `swift test`: 850 tests in 4 runs, 0 failures.
    - next: /review
  timestamp: 2026-08-13T13:55:32.554646+00:00
- actor: claude-code
  id: 01kzxq7tny1e4dp1sxy933vv8m
  text: |-
    ### review — clean

    Scope: commit 2701212, range `HEAD~1..HEAD`.

    - evidence: The review engine gave `findings: 0, confirmed: 0, refuted: 0, attempted: 0, failed: 0` and the text "Nothing in scope to review." `attempted: 0` means the engine opened no file. The commit changes only a test file and a Markdown file, and the engine reads neither. Thus the engine gave no verdict. This verdict is the reading of the reviewer, plus the recorded mutation proofs.
    - compile evidence: `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` gave exit 0, "TEST BUILD SUCCEEDED", 0 errors and 0 warnings, on the committed tree with no local change.
    - mutation proof of the round before: a type error in the changed function left `swift build --build-tests` at exit 0, and gave `xcodebuild build-for-testing` exit 65.
    - mutation proof of this round: removal of the wired-limit call makes both new tests fail. The median step went to 2.1247 s against the 0.8710 s budget, exit 65. With the call put back, exit 0 and a median of 0.5917 s.

    #### The seven findings of 2026-08-13 08:06 are closed

    1. Compile evidence. The file header now records that `swift test` is blind to this file, and it names the `xcodebuild build-for-testing` command as the compile evidence. `docs/deepseek-v4-support.md` records the same. CLOSED.
    2. No assertion covers the fix. Two tests now guard it: `wiredMemoryLimitCoversTheWholeCheckpoint` and `decodeStepStaysInsideTheLongGenerationBudget`. CLOSED.
    3. The function threw away its result. `raiseWiredMemoryLimit()` now gives back a `WiredMemoryOutcome?` that holds `requestedBytes` and `appliedBytes`, and `isFullyApplied` compares them. The test asserts it. CLOSED.
    4. The guard gave no message. The guard now prints why it stopped, in the manner of the other gates of the file. CLOSED.
    5. The wired limit rode on the skip gate. The limit now has `deepseekV4WiredMemoryBytes`, built from `deepseekV4CheckpointBytes` plus `deepseekV4WiredHeadroomBytes`. The doc of the skip gate says which question each constant answers. CLOSED.
    6. "147 GB of the 141 GB total". The text now says the routed experts hold 137 GiB of the 141 GiB total, which is 97%. No "147" is left in the file. CLOSED.
    7. The 91 GB and 141 GB disagreement. One value set is now used everywhere: 151,482,475,612 bytes, which is 141 GiB, with routed experts at 137 GiB. No "91 GB" is left in the test file or in `docs/deepseek-v4-support.md`. CLOSED.

    #### Checks made against the new code

    - The applied-bytes contract holds. `WiredMemoryTicket.start()` gives back the applied limit, and it gives back the baseline when the backend is not supported. `WiredFixedPolicy.limit()` gives back the asked bytes with no clamp. Thus `appliedBytes >= requestedBytes` truly detects a short apply.
    - The step measurement is true. `TokenIterator.next()` calls `asyncEval` on the new token and gives back the previous token through `.item()`, which forces the previous step. Thus each measured interval is one decode step. The first call absorbs the prompt, and the test drops it.
    - The token count is exact. `maxTokens` is 17, the test drops one step and measures 16, and `TokenIterator` honours `maxTokens`.
    - The budget arithmetic holds. 240 min x 60 x 0.75 / 12,400 = 0.8710 s per step. At the measured 0.593 s the 12,400-token run takes 123 minutes, inside the 240-minute limit.
    - The skip gate admits no machine that the new assertion then fails. Measured on this M3 Ultra, the recommended working set is 464 GiB of 512 GiB physical, a ratio of 0.906. A machine at the 160 GiB gate thus reports about 145 GiB, above the 141.08 GiB checkpoint, thus `requestedBytes >= deepseekV4CheckpointBytes` holds at the gate.

    - next: the card goes to `done`.
  timestamp: 2026-08-13T14:08:25.790236+00:00
- actor: claude-code
  id: 01kzxq946m9mg552n91e8y6ma2
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 2 files. All 7 findings closed. `raiseWiredMemoryLimit()` now returns a `WiredMemoryOutcome`, the wired limit has its own constants, and all size values agree (151,482,475,612 bytes = 141 GiB; routed experts 137 GiB). Two guard tests added.
    - test: green — `swift test`, 0 failures, 2 skips that were there before. `xcodebuild build-for-testing` for IntegrationTesting: TEST BUILD SUCCEEDED, 0 errors, 0 warnings.
    - commit: 2701212 fix(mlx-lm): close review findings on the DeepSeek-V4 wired-limit test
    - review: clean — the engine attempted 0 files again, thus the verdict comes from a direct reading plus two mutation proofs. Behaviour proof: with the wired-limit call removed the two new tests fail (median step 2.1247 s against the 0.8710 s budget, exit 65); with it back they pass (median 0.5917 s, exit 0). The card moved to `done`.
  timestamp: 2026-08-13T14:09:08.308205+00:00
position_column: done
position_ordinal: f180
title: 'DeepSeek-V4 decode performance: about 2.4 s/token blocks the >12k-token test'
---
## What

Measured 2026-08-12 on the M3 Ultra (512 GiB), `mlx-community/DeepSeek-V4-Flash-4bit`, card ^e7b24ws:

- 64 greedy tokens in ~177-210 s inside the parity test (load excluded) — about 2.4-3.3 s/token.
- The chat/thinking test took 232 s for two 48-token generations.
- At this speed the 12,400-token issue-1662 regression test (`longGenerationPastTwelveThousandTokensCompletes`) needs about 7 hours. The suite's per-test limit is 240 minutes, thus the test cannot complete. A run was killed after 83 minutes in progress.

For contrast, the Python reference (mlx-lm PR 1189) also decodes slowly on this checkpoint (~3.7 s/token for its 64-token fixture run), thus the cause is likely in the model's compute shape (284B-total MoE gathers), not only in the Swift port. Profile the Swift decode step, find the dominant cost (switch_mlp gather, hyper-connection, attention), and compare with the Python reference throughput.

## Acceptance Criteria

- [x] A profile names the dominant cost of one decode step with numbers. — The routed experts (`ffn.switch_mlp`) carry 2039 ms of a 2060 ms decode step, which is 99%. The cost is Metal residency of the routed-expert weight buffers, not arithmetic: the same layer chained 43 times, with the same operation count, takes 68 ms.
- [x] A decision or a fix: either decode gets fast enough that 12,400 tokens fit inside the 240-minute test limit, or the card records why not and what the >12k test should do instead. — FIXED. A `WiredMemoryTicket` that starts before the weight load takes one decode step from 2.124 s to 0.593 s in the debug test build, thus 12,400 tokens take about 123 minutes, inside the 240-minute limit. A release build with a direct model call measures 2.10 s and 0.068 s for the same two steps.

#deepseek-v4

## Review Findings (2026-08-13 08:06)

Scope: commit f389c22, range `HEAD~1..HEAD`.

The review engine gave this result word for word:

> ## Review Findings (2026-08-13 08:01)
>
> Nothing in scope to review.

The counts were `findings: 0, confirmed: 0, refuted: 0, attempted: 0, failed: 0`. `attempted: 0` means the engine opened no file. The commit changes only a test file and a Markdown file, and the engine does not read those. Thus this is not a clean result. Nobody looked.

### Mutation proof

A mutation proof replaced the missing engine pass.

1. Put a type error in the changed function `raiseWiredMemoryLimit()`, at `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:137`.
2. `swift build --build-tests` gave **exit 0, "Build complete!"**. The error stayed unseen. `swift package describe` shows no SwiftPM target below `IntegrationTesting/`.
3. `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'` gave **exit 65, "TEST BUILD FAILED"**, with `DeepseekV4IntegrationTests.swift:137:36: error: cannot convert value of type 'String' to specified type 'Int'`.
4. `git checkout --` put the file back. The same `xcodebuild` command then gave **exit 0, "TEST BUILD SUCCEEDED"**. `git status` shows the file clean.

Result: the Xcode scheme compiles the changed code and finds a mutation in it. The SwiftPM build does not. No test asserts the effect of the change.

The real-weights suite did not run. It loads 141 GB and needs many minutes, and the >12k-token test needs hours.

### Findings

- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:134` — The commit message gives "swift test — 535 tests, 0 failures" as the evidence for this change. No SwiftPM target holds `IntegrationTesting/`, thus that command never compiles the changed code. The mutation proof above shows this: a type error in the changed function left `swift build --build-tests` at exit 0. Give evidence that covers the changed code, or record which build command covers it.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:134-140` — No assertion covers the fix. The suite has no `#expect` for the applied wired limit, and none for decode speed. Line 388 measures `elapsed`, and line 391 only prints it. If a later change removes the call at line 154, or gives a wrong limit, the code still builds and all assertions still pass. The only symptom is the >12k test that hits the 240-minute limit after hours. Add an assertion that fails quickly when the limit is not raised.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:138-139` — The function throws away its own result. The value `applied` goes only to `print`. If `start()` applies fewer bytes than the request, the suite continues at the slow speed and reports no failure. Compare `applied` with `bytes`.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:135` — The `guard let recommended = GPU.maxRecommendedWorkingSetBytes() else { return }` goes back with no message. Every other gate in this file prints the reason when it stops (lines 224, 231, 241). Print the reason here also.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:136` — The wired limit uses `deepseekV4RequiredMemoryBytes`. Line 43 documents that constant as the least physical memory the machine must have, which is a skip gate. The two values answer different questions. A later change to the skip gate changes the wired limit with no warning, and can remove this fix. Give the wired limit its own constant with its own name.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:127-128` — The text says "The routed experts hold 147 GB of the 141 GB total". A part cannot be larger than the whole. Correct the two numbers.
- [x] `IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift:122` and `docs/deepseek-v4-support.md:134` — The new text says Metal makes "all 141 GB" resident. Lines 11, 43, 145 and 234 of the same file, and `docs/deepseek-v4-support.md:69`, say the checkpoint is about 91 GB. The text does not say why the two numbers are different. Make them agree, or say what each number measures. The correction belongs to the new text, because this commit made the disagreement.