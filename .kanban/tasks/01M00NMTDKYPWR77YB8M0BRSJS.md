---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m00vs5rw1gecea8j8pjp5jea
  text: |
    ### The stack sample this card waited for CANNOT exist. I withdraw the request.

    I asked the reporting sessions for a stack sample taken during the hang, to
    show the tool frame above the `perform` frame. That request was wrong, and
    this comment withdraws it.

    A suspended Swift async function holds NO OS thread. Its continuation lives
    on the heap, and `sample` and `spindump` walk thread stacks alone. Thus a
    tool body that waits for the container is invisible to both, and so is a
    `respond` that waits inside `perform`. The Multitool session ran
    `sample <pid> 3` during a hang, took 251 KB, and found zero matches for
    `perform`, `respond`, `call(arguments`, `SerialAccess` or `generate`. That
    is the expected answer, not a failed measurement.

    The parked workers, the idle run loop and the MLX scheduler on a condition
    variable are the same fact from the other side: every async frame is off
    the threads by construction.

    Nobody must hold this card open for that artifact.

    ### What this repository can confirm on its own

    `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, read today:

    - Line 1016: `try await container.perform(nonSendable: messages) { context, messages in`
    - Line 1232: `for call in result.toolCalls {` — INSIDE that block
    - Line 1234: `await Self.emitToolCall(` — INSIDE that block
    - Line 1475: the block closes

    Thus this repository emits the tool call to the consumer WHILE it holds the
    container. That is the half of the ordering that only this code can state,
    and it is now stated with line numbers.

    ### What is left, and how to settle it

    The one open link is whether the SDK runs the tool body while the `respond`
    call is still suspended inside `perform`. Three ways to settle it, cheapest
    first:

    1. **A unit test in this repository.** A stub tool whose body calls the same
       container. No 30B model, no download, seconds to run. If it hangs, the
       case closes with no consumer evidence at all. This is the cheapest by a
       wide margin and it belongs here, not with a consumer.
    2. **LLDB with Swift concurrency support** — `language swift task backtrace`,
       or `thread backtrace --extended`. That reads suspended async frames, which
       `sample` cannot. The Multitool session offered this on a next hang, which
       costs them a revert of their workaround.
    3. **The recorded transcripts they already hold.** In the hung run the outer
       `standard` session held one line and no recorded response, thus its turn
       had not completed, and a forked session carrying the selection grammar
       existed. Their tool body creates that fork. Thus the tool body ran while
       the outer turn was in flight.

    Point 3 and the line numbers above are already a strong case. Point 1 makes
    it conclusive.

    ### Scope

    The correction lives in `MLXFoundationModels`, which is neither DeepSeek-V4
    nor MiniMax. The `catch-up-upstream` branch holds its difference against
    `main` to those two alone, thus even the unit test of point 1 is a scope
    decision for the user, and it is surfaced to them.
  timestamp: 2026-08-14T19:25:31.804630+00:00
- actor: claude-code
  id: 01m00w05jctqwptqn79fzhzv8t
  text: |
    ### The user decided: correct it in this fork

    > I think we need to fix it here and leave a clear comment to pay attention
    > when we merge — I'm assuming this will get fixed upstream eventually, but
    > if not we can supply it as a standalone PR.

    Thus:

    - The correction lands in this repository now. It is NOT held for upstream.
    - The scope rule of the `catch-up-upstream` branch bends for this one card.
      The user said so directly, thus `MLXFoundationModels` may change.
    - The correction carries a comment that a person reads at merge time. It
      states that this is a local correction of a defect of the upstream
      `MLXLanguageModel`, that upstream may correct it later, and that a merge
      must not drop it silently.
    - If upstream does not correct it, this work becomes a stand-alone pull
      request to upstream.

    ### The direction

    Direction 1 of the three: release the container across a tool round, and
    take it again for the continuation. The lock then covers generation, not
    the whole turn.

    Direction 3 (document and fail loudly) is not enough on its own here. The
    user asked for a correction, and a host that generates inside a tool body
    does a normal thing.
  timestamp: 2026-08-14T19:29:20.972578+00:00
- actor: claude-code
  id: 01m00wycqvakpcphv48z65zxft
  text: |
    ### The premise is DISPROVED on this SDK. No production code was changed.

    The card names one open link:

    > That the SDK invokes a tool body while the `respond` executor call is still
    > suspended inside `perform`, rather than after it returns.

    Point 1 of the earlier comment asked for a unit test in this repository to
    settle it. That test is written, and it settles it the other way: **the SDK
    invokes the tool body AFTER `respond` returns.**

    ### The test

    `Tests/MLXFoundationModelsTests/ToolBodyContainerReentryTests.swift`

    No model, no download, no weights, no GPU shaders. The doubles are:

    - `ScriptedByteTokenizer` — one token for each byte, thus a script is its
      own UTF-8 bytes. End-of-text is byte `0x03`, which is one UTF-8 byte, thus
      `convertTokenToId` resolves it and the generation loop stops.
    - `ScriptedLanguageModel` — a `Module` / `MLXLMCommon.LanguageModel` that
      replays one scripted token sequence for each generation round. It counts a
      round at `prepare(_:cache:state:prefill:)`.
    - `FixedPromptInputProcessor` — returns three token IDs for every input.
    - A real `ModelContainer` over those doubles, given to a real
      `MLXLanguageModel` through its injected `load` closure.

    The shape is the host's own shape:

    - A real `LanguageModelSession` with a real `Tool`.
    - The tool body opens a SECOND `LanguageModelSession` on the SAME
      `MLXLanguageModel` and generates. That is the forked selection tier of the
      `FoundationModelsMultitool` report.
    - The executor runs its real `.allowed` tool-calling path over the doubles.

    The three scripts, in the order the model serves them:

    1. `<tool_call>{"name":"probe","arguments":{"query":"weather"}}</tool_call>`
    2. `Candidate one ranks highest.`  (the tool body's own generation)
    3. `The probe answered.`          (the outer turn's continuation)

    ### What the test did BEFORE any change: it PASSED, in 0.077 s

    ```
    ✔ Test "A tool body may generate on the same model while its turn is in flight"
      passed after 0.077 seconds.
    ✔ Test run with 1 test in 1 suite passed after 0.077 seconds.
    ```

    Three assertions passed, and each one is load-bearing:

    1. `log.emittedToolCallNames == ["probe"]` — recorded through the
       `Executor.generationObserver` task-local, which fires inside
       `Self.emitToolCall`, which is INSIDE the `container.perform` block. Thus
       the test truly ran the defective code path. It is not a vacuous pass.
    2. `log.generatedText == rounds[1]` — the tool body's nested `respond`
       received script 2, not script 1. The round counter lives in the
       `ScriptedLanguageModel` INSTANCE. Thus the nested session drove the SAME
       model instance, thus the SAME `ModelContainer`, thus the SAME
       `SerialAccessContainer` lock. This is the proof of container identity.
    3. `content == rounds[2]` — the outer turn's continuation round ran and the
       turn completed.

    Together: the executor emitted its tool call from inside the lock, AND the
    tool body then took that same lock and generated, AND the turn finished. The
    tool body did not wait: all three generations completed in 0.077 s.

    Thus `channel.send(.toolCalls(...))` returns, the `perform` block exits, and
    `respond` returns, BEFORE the session runs the tool body. There is no
    deadlock from the container lock.

    ### Proof the test cannot hang and cannot pass falsely

    The timeout was set to `Duration.nanoseconds(1)` one time, and the test
    FAILED with the recorded message instead of hanging:

    ```
    ✘ Test ... recorded an issue at ToolBodyContainerReentryTests.swift:302:25
      ↳ The turn did not complete within 1e-09 seconds. ...
    ✘ Test ... failed after 0.003 seconds with 1 issue.
    ```

    The timeout was then set back to `Duration.seconds(30)`. The turn runs in its
    own task and reports through an `AsyncStream`, because a task parked on
    `AsyncMutex` cannot be cancelled; the timeout never awaits that task
    directly, thus the task group always finishes.

    ### What stays true, and why it is not enough

    The line numbers of the earlier comment stay correct. `respond` DOES emit the
    tool call while it holds the container. But an emit inside a lock is only
    half of a deadlock. The other half — the consumer running the tool body
    before the producer releases — does not happen on this SDK. Thus the
    correction the user asked for would remove a latent hazard, not the reported
    hang, and shipping it as "the fix for the 15-minute park" would be false.

    Measured against: FoundationModels 2.0.62.1, MacOSX27.0.sdk, Xcode-beta,
    `swift test`, arm64e-apple-macos14.0.

    ### The hang has another cause. Leads, marked as hypotheses.

    The card says to record this rather than close it silently, so here is what
    the evidence now points at. None of these is verified.

    1. **Not a Swift lock at all.** The sample shows "MLX scheduler on a
       condition variable" at 0.0% CPU. A parked `AsyncMutex` leaves the threads
       idle in the run loop, which matches; but an MLX scheduler thread on a
       condition variable is MLX's own stream/eval scheduler waiting for work
       that never arrives. That is a stronger fit for an MLX-level wait than for
       a `SerialAccessContainer` wait.
    2. **Concurrent, not nested, generation on one model.** `ModelContainer.generate`
       (`Libraries/MLXLMCommon/ModelContainer.swift`) builds the generation
       stream INSIDE `context.read` but the stream keeps generating AFTER `read`
       returns. Its own comment states this is on purpose ("only visiting the
       model exclusively for the pre-fill time"). Two callers on that path
       therefore run concurrent MLX work on one model. `MLXLanguageModel.Executor`
       does not use that path, but a host that reaches `ModelContainer` directly
       does.
    3. **The forked tier ran under a grammar.** The recorded transcript stops at a
       fork "carrying the selection grammar". That is the guided path
       (`GuidedGenerationLoop`, xgrammar), not the plain tool path. A park there
       would be inside the xgrammar shim, and the shared per-model caches
       (`makeXGTokenizer`, `makeConstraint`, `makeTokenizerBias`) are the shared
       state that disappears when the two slots name different models — which is
       exactly the reported workaround signature. This lead fits "give the two
       slots different models and the sharing disappears" as well as the
       container lock does, and it was never examined.

    A next step that costs little: reproduce with the tool body running a
    GUIDED generation (a schema or `.required` tool mode) on the same model,
    instead of the plain text generation this test uses.

    ### State

    - `Tests/MLXFoundationModelsTests/ToolBodyContainerReentryTests.swift` is
      ADDED and kept. It satisfies acceptance criterion 1 ("a test that generates
      from inside a tool body on the same container either completes or fails
      loudly; it does not hang") and it guards the behaviour from here on.
    - `Libraries/MLXFoundationModels/MLXLanguageModel.swift` is UNCHANGED. The
      user's instruction was explicit: do not change production code on a premise
      that the test disproves.
    - Acceptance criterion 3 ("If the premise is wrong ... that is recorded here
      with what the real cause is") is met for the first half. The real cause is
      not known; the leads above are the honest state of it.
    - Acceptance criterion 2 (state the constraint in `MLXLanguageModel`'s own
      documentation) is NOT met, because the constraint the card names does not
      exist on this SDK. A person must decide what, if anything, to document.

    ### The decision a person must make

    1. Land the Direction 1 correction anyway, as hardening against a future SDK
       that consumes the channel differently? It is a real latent hazard and the
       test above would keep guarding it. But it does not fix the reported hang.
    2. Or re-open the investigation on the leads above, and keep
       `MLXFoundationModels` untouched on this branch as its scope rule says?

    The user's earlier decision assumed the premise. The premise is now
    disproved, thus the decision needs the user again.
  timestamp: 2026-08-14T19:45:51.355880+00:00
- actor: claude-code
  id: 01m00wynnvq75sgm0zqkggp4rb
  text: |
    ### implement — stuck

    - evidence: 1 file ADDED — `Tests/MLXFoundationModelsTests/ToolBodyContainerReentryTests.swift`.
      `Libraries/MLXFoundationModels/MLXLanguageModel.swift` is UNCHANGED.
      BEFORE the fix the reproduction PASSED in 0.077 s, not hung: the executor
      emitted its tool call from inside `container.perform` (proved by the
      `generationObserver` mirror), the tool body then generated on the SAME
      container (proved by the round counter of the shared model instance), and
      the turn completed. Thus the card's one unproven link is false on this SDK:
      the session runs the tool body AFTER `respond` returns.
      The timeout guard was proved by setting it to 1 ns one time: the test
      FAILED with its message in 0.003 s and did not hang; it was then set back
      to 30 s.
      `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test --filter
      MLXFoundationModelsTests` — 140 tests in 22 suites passed, 0 failures.
    - next: a person must decide — land Direction 1 as hardening against a future
      SDK (it removes a latent hazard, not the reported hang), or re-open the
      investigation on the three leads recorded above and leave
      `MLXFoundationModels` untouched. The earlier decision assumed a premise the
      test now disproves.
  timestamp: 2026-08-14T19:46:00.507390+00:00
- actor: claude-code
  id: 01m05ez91sq67tc60k4hfezchz
  text: |
    ### real-weights evidence from ^5ddpy4s — a tool body that generates does NOT deadlock here

    `^5ddpy4s` added `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/ToolBodyContainerReentryRealModelTests.swift`.
    It has the shape of `ToolBodyContainerReentryTests`, but it runs over real
    weights and thus does real MLX evaluation.

    **Model.** `mlx-community/Llama-3.2-1B-Instruct-4bit`, 680 MB, from the local
    hub cache.

    **Result.** PASSED after 3.697 seconds. It did NOT hang.

    ```
    ✔ Test "A tool body may generate on the same real model while its turn is in flight" passed after 3.697 seconds.
    ✔ Test run with 2 tests in 1 suite passed after 3.722 seconds.
    ```

    **What the turn does.** A `LanguageModelSession` runs a tool under
    `.toolCallingMode(.required)`, thus the tool call is grammar-constrained and
    always happens. The tool BODY opens a second `LanguageModelSession` on the SAME
    `MLXLanguageModel` and awaits `respond`, thus a nested generation runs on the
    one resident container while the outer turn is in flight. The outer turn then
    completes on a third generation.

    Three real forward passes, one resident container, no park.

    **The test can fail.** Two mutations prove it:

    1. `turnTimeout` set to 1 ms — the test failed at the timeout arm, thus the arm
       that catches a park does fire.
    2. The tool body replaced by a constant, with no nested generation — the test
       failed on `log.callCount → 0` and `log.generatedText → nil`, thus the
       nested-generation assertions are load-bearing.

    **What this closes.** This card stays closed on evidence that covers real
    weights, not a stub. It adds nothing about the grammar-constrained nested
    decode: this turn's nested round is unconstrained. The cause of the reported
    hang is in `FoundationModelsRouter` — a per-container `AsyncSemaphore(value: 1)`
    that `beginTurn()` holds for the complete turn — and not in this repository.
  timestamp: 2026-08-16T14:17:52.441976+00:00
- actor: claude-code
  id: 01m05fsdnedv5rdttptfnx3kkr
  text: |
    ### The premise is FALSE. I checked the code again, independently.

    I read `MLXLanguageModel.swift` on the current tree. I did not take the earlier
    comments on trust. The line numbers of the card moved since the card was filed:
    the card says 1017 / 1232 / 1475, and the current tree says 1081 / 1298 / 1531.
    Each item below gives the symbol and the current line.

    **What the card gets RIGHT.** The executor emits the tool-call delta while it
    holds the container. Two paths do this:

    - `.allowed` path: `Self.emitToolCall` at line 1303.
    - `.required` path: `emitRequiredToolCallEvent` at line 1479.

    Both are inside the block that `try await container.perform(nonSendable:)`
    opens at line 1081.

    **What the card gets WRONG.** That block holds NO tool round. Both exits are
    short:

    - Lines 1334-1335: `Stream.gpu.synchronize()`, then `return`. That `return` sits
      at the indent of the `perform` closure body. An `if` block is not a function
      body, thus the `return` leaves the closure itself. The tool-call emit at line
      1303 is the last work the `.allowed` path does before it releases.
    - Line 1531: the `perform` block closes. The `.required` path reaches it just
      after its emit at 1479 and its usage events.

    There is NO loop over tool rounds in that block. The comment at lines 1198-1201
    gives the design: "Tool path, entered on every round while tools are enabled --
    fresh turns and continuations alike." Each continuation round is a NEW `respond`
    call from the session. Each `respond` call opens its own `perform`. Thus the
    lock is taken one time and released one time for each round. It is not held
    across rounds.

    **The lock releases when the block returns.** The chain has three steps:

    - `ModelContainer.perform(nonSendable:_:)` calls `context.read`
      (`Libraries/MLXLMCommon/ModelContainer.swift:116`).
    - `SerialAccessContainer.read` calls `lock.withLock`
      (`Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift:55`).
    - `AsyncMutex.withLock` has `defer { unlock() }`
      (`Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift:34`).

    Thus the container is free as soon as the closure returns. The session then runs
    the tool body against a free container.

    **One call site only.** `container.perform` occurs ONE time in the complete
    `MLXFoundationModels` library: `MLXLanguageModel.swift:1081`. No other path of
    that library takes the container, thus no other path can hold it across a tool
    round.

    ### The tests agree, and they pass on the current tree

    Both branches of the tool code now have a test.

    - `Tests/MLXFoundationModelsTests/ToolBodyContainerReentryTests.swift` covers the
      `.allowed` path with scripted doubles. I ran it today. The test "A tool body
      may generate on the same model while its turn is in flight" PASSED after 0.043
      seconds. The complete bundle gave 155 tests in 24 suites with 0 failures.
    - `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/ToolBodyContainerReentryRealModelTests.swift`
      covers the `.required` path with real weights on
      `mlx-community/Llama-3.2-1B-Instruct-4bit`. Card ^5ddpy4s recorded a PASS after
      3.697 seconds. This is the grammar-constrained path that an earlier comment
      listed as an untested lead. It is tested now.

    ### The cause is OUTSIDE this repository

    The `FoundationModelsRouter` session traced the hang it saw to a per-container
    `AsyncSemaphore(value: 1)`. Its `beginTurn()` holds that semaphore for the
    complete turn together with the tool rounds of that turn. That code is in
    `FoundationModelsRouter`, which is a DIFFERENT repository.

    I checked this repository for the same construct: `Libraries/` holds no
    `AsyncSemaphore` and no `beginTurn`. The construct that causes the hang does not
    exist here.

    Note that an earlier comment on this card ruled the router out. That conclusion
    was too early: it examined only `container.perform` call sites of the router,
    and the router's lock is a semaphore of its own, not a container lock.

    ### The acceptance criteria

    - Criterion 1 is MET. Two tests generate from inside a tool body on the same
      container. Both complete. Both fail loudly and do not hang: each has a timeout
      arm, and each timeout arm was proved by a mutation (1 ns on the stub test, 1 ms
      on the real-model test).
    - Criterion 3 is MET. The premise is wrong, and this comment records why, with a
      file and a line for each item of evidence. The real cause is the router
      semaphore, outside this repository.
    - Criterion 2 does NOT apply, and it stays unticked. Its condition is "Whichever
      direction is chosen". No direction is chosen, because there is no defect here
      to correct. The constraint that criterion 2 names -- a tool body must not
      generate on its own container -- is FALSE on this code. To write it in the
      documentation would put a false statement in the documentation. A person must
      decide if `MLXLanguageModel` instead gets the OPPOSITE statement: that a tool
      body MAY generate on the same container, because two tests now guard that
      behaviour. That decision is surfaced to the user, not made here.

    ### No production code changed

    `Libraries/MLXFoundationModels/MLXLanguageModel.swift` is UNCHANGED. This card
    alone changed. No fix was invented for a defect that does not exist here.
  timestamp: 2026-08-16T14:32:09.134736+00:00
- actor: claude-code
  id: 01m05ftgfp7v62825p3c6qd360
  text: |
    ### implement — no-change

    - evidence: NO source file changed. `Libraries/MLXFoundationModels/MLXLanguageModel.swift`
      is unchanged; `git status` shows only kanban files. The card description and
      this comment thread are the whole change.
      I checked the premise independently and it is FALSE: `container.perform` opens
      at `MLXLanguageModel.swift:1081` and closes at `:1531`, and it holds NO tool
      round. The `.allowed` path emits its tool call at `:1303` and then `return`s
      out of the closure at `:1335`. The `.required` path emits at `:1479` and
      reaches the close at `:1531`. `AsyncMutex.withLock` releases on `defer`
      (`SerialAccessContainer.swift:34`), thus the container is free while the
      session runs the tool body. `container.perform` has exactly ONE call site in
      the complete `MLXFoundationModels` library.
      Test evidence on the current tree:
      `xcodebuild build-for-testing -skipPackagePluginValidation -scheme
      mlx-swift-lm-Package -destination 'platform=macOS'` then `xcrun xctest
      MLXFoundationModelsTests.xctest` — 155 tests in 24 suites passed, 0 failures.
      "A tool body may generate on the same model while its turn is in flight"
      PASSED after 0.043 s. The five bundles were NOT all run, because no source
      file changed.
      Criteria 1 and 3 are ticked. Criterion 2 does not apply and stays unticked,
      with the reason written into the description.
    - next: a person decides one question only — does `MLXLanguageModel` get a
      documentation sentence that states the POSITIVE guarantee (a tool body may
      generate on the same container)? Two independent sessions read this code and
      both concluded a deadlock exists, thus the code misleads a reader. Nothing
      else on this card is open.
  timestamp: 2026-08-16T14:32:44.790306+00:00
- actor: claude-code
  id: 01m05gghcqt3r2vmbzfqqbv6k7
  text: |
    ### The guarantee is documented. Criterion 2 is met.

    The user decided to write the documentation:

    > yes, write the documentation. I am deciding this, not escalating it. Two
    > independent sessions read this code and both concluded a deadlock exists. That
    > is a defect in what the code communicates, even though the behavior is
    > correct.

    ### Where it went, and why there

    `Libraries/MLXFoundationModels/MLXLanguageModel.swift`, at the entry of the tool
    path, immediately above `if !enabledToolDefinitions.isEmpty`. It extends the
    comment block that is already there.

    That position is correct for three reasons:

    1. It is the entry of BOTH tool branches. The `.allowed` branch and the
       `.required` branch each begin below it, thus one block covers both emit sites
       and no text is duplicated.
    2. It sits with the emit sites, not at the top of the file. A reader who hunts
       for the deadlock reads the tool path, thus the reader meets the answer at the
       place of the question.
    3. The block above it already describes the tool path. The new text continues
       that description instead of starting a competing one.

    ### The text

        // A tool body can start a second generation on this same
        // model. This executor releases the model container BEFORE
        // the SDK runs a tool body. This `perform` closure sends the
        // tool-call delta, then it returns. Each continuation round
        // opens its own `perform`. Thus the container lock covers
        // one generation. It does not cover a complete turn with
        // the tool rounds of that turn.
        // `ToolBodyContainerReentryTests` and
        // `ToolBodyContainerReentryRealModelTests` guard this
        // guarantee.

    It states the guarantee, it names the mechanism, and it names the two tests. It
    does not repeat the investigation; this card holds that.

    `//` is correct and `///` is not: the comment is inside a method body, thus it
    documents no declaration. The `swift/documentation` rule that requires `///`
    governs documentation comments on declarations.

    ### The rules were read first

    `dump validators` gave 65 rules across 9 validators for this file. The rules that
    touch a comment were checked one at a time:

    - `code-hygiene/no-commented-code` — the block is English prose, thus it is not
      commented-out code at any length.
    - `swift/documentation`, "Describe what/why, not how" — the text states the
      guarantee and why it holds.
    - `swift/concurrency` — this rule ASKS for a comment that states a
      synchronization invariant. The new text is what the rule wants.
    - No rule forbids a test name, a file name, or an issue link inside a comment.
    - No line-length rule exists in the set, and comment-only lines do not count
      toward the function-length gate.

    ### Verification

    - `swift-format format --in-place` then `swift-format lint` -- CLEAN. The
      formatter changed nothing in the new text. The file was clean before the edit
      too, thus the result is not accidental.
    - `xcodebuild build-for-testing -skipPackagePluginValidation -scheme
      mlx-swift-lm-Package -destination 'platform=macOS'` -- TEST BUILD SUCCEEDED,
      0 compiler warnings.
    - All FIVE bundles ran, and each exited 0 with 0 failure lines:

      | bundle | result |
      |---|---|
      | MLXLMTests | 844 tests in 71 suites + 475 XCTest, 0 failures |
      | MLXGuidedGenerationTests | 70 tests in 14 suites, 0 failures |
      | MLXFoundationModelsTests | 155 tests in 24 suites, 0 failures |
      | CXGrammarTests | 7 tests in 5 suites, 0 failures |
      | MLXHuggingFaceMacrosTests | 5 XCTest, 0 failures |

      Total: 1076 Swift Testing tests and 480 XCTest tests. Nothing was skipped.

    - The guard test still passes: "A tool body may generate on the same model while
      its turn is in flight" PASSED after 0.045 seconds.

    ### A note for the merge

    This comment describes behaviour of the UPSTREAM `MLXLanguageModel`, and it
    corrects no code. Upstream can move this code without knowing that the comment
    exists. A person who merges upstream must check that the comment still sits at
    the entry of the tool path, and that the two branch names below it are unchanged.
  timestamp: 2026-08-16T14:44:46.615206+00:00
- actor: claude-code
  id: 01m05ggrwryhh5082ynhrq8tn1
  text: |
    ### implement — changed

    - evidence: 1 source file — `Libraries/MLXFoundationModels/MLXLanguageModel.swift`
      (+11 lines, comment only, at the entry of the tool path above
      `if !enabledToolDefinitions.isEmpty`). No executable code changed.
      `swift-format lint` CLEAN, and the formatter changed nothing in the new text.
      `xcodebuild build-for-testing` — TEST BUILD SUCCEEDED, 0 compiler warnings.
      All five bundles, each exit 0 with 0 failure lines: MLXLMTests 844 + 475
      XCTest; MLXGuidedGenerationTests 70; MLXFoundationModelsTests 155;
      CXGrammarTests 7; MLXHuggingFaceMacrosTests 5 XCTest. 1076 Swift Testing and
      480 XCTest, nothing skipped.
      Acceptance criteria 1, 2 and 3 are ALL ticked now. Criterion 2 is met by the
      new comment; the direction taken is recorded in the description.
    - next: ready for `/review`. Nothing is open on this card. One item for a future
      merge is recorded: the comment describes upstream code that upstream can move,
      so a merge must check that it still sits at the entry of the tool path.
  timestamp: 2026-08-16T14:44:54.296991+00:00
position_column: doing
position_ordinal: '8380'
title: One turn holds the container lock across tool rounds, so a tool that generates deadlocks
---
Filed by the `FoundationModelsMultitool` session, with `FoundationModelsRouter` concurring after independently reading the same code.

**RESOLVED: the premise is false for this repository. See the comment of
2026-08-16.** The tool rounds do NOT run inside `container.perform`. The block
emits the tool-call delta and then returns, thus the container is free while the
session runs the tool body. The reported hang comes from a per-container
`AsyncSemaphore(value: 1)` in `FoundationModelsRouter`, a different repository.
`MLXLanguageModel` now documents the true guarantee at its tool path. The text
below is the original report and is kept as filed.

## The defect

`MLXLanguageModel.respond(to:model:streamingInto:)` (`Libraries/MLXFoundationModels/MLXLanguageModel.swift:940`) opens `container.perform(nonSendable: messages) { context, messages in …` at **:1017**, and that block does not close until roughly **:1475**. The tool-calling continuation rounds run *inside* it — `:1232` iterates `result.toolCalls`, and the comment at `:1160` says continuation rounds "run the tool path like fresh turns".

So one turn holds the `SerialAccessContainer` lock for its whole length, tool rounds included. **Any tool whose body generates on the same container can never acquire it, and both sides park forever.**

This is not a hypothetical. It is a guaranteed deadlock for a host whose tool generates — and generating inside a tool is a normal thing for a host to do: a discovery tool that ranks candidates with a small model, a router that summarizes, a tool that asks a model to pick from a catalog.

## The evidence

`FoundationModelsMultitool`'s `searchTools` runs a selection tier: given a task string, a model picks which catalog entries match, under a grammar. It runs from inside the outer turn's tool call.

When both slots named the same model — one `ModelRef`, therefore one resident container — a gated scenario produced this:

- 15 minutes with no completion, at **0.0% CPU**
- 18.8 GB resident, **98% system memory free, zero swap** (so not memory pressure)
- `sample` showed every thread parked: MLX scheduler on a condition variable, thread pool idle, main thread in the run loop
- the recorded transcript stops exactly at the forked selection-tier generation: a `standard` session with one line and nothing generated, then a fork carrying the selection grammar, then silence

Give the two slots different models and the sharing disappears. We are running that now as a workaround, and it is only a workaround: it makes the deadlock unreachable for our configuration rather than fixing it. Any host that points two roles at one model hits this, and pointing two roles at one model is the *efficient* configuration — it loads the weights once.

## What was ruled out

- **Not the router.** `FoundationModelsRouter` has exactly one `container.perform` in its whole `Sources` tree, in `embed(texts:)`. Generation never goes through it — the router drives a native `LanguageModelSession` through `MLXFoundationModelsSessionBackend` and takes no container lock. Nothing on the router side needs re-entrancy handling.
  **CORRECTION (2026-08-16): this conclusion was wrong.** It examined only the
  `container.perform` call sites of the router. The router's lock is an
  `AsyncSemaphore(value: 1)` of its own, held by `beginTurn()` for the complete
  turn with its tool rounds. That is the real cause.
- **Not memory, not the model, not the grammar.** See the numbers above.

## What is not proven

That the SDK invokes a tool body while the `respond` executor call is still suspended inside `perform`, rather than after it returns. Reading alone did not settle it. The hang is the strongest evidence that it does, since a deadlock requires exactly that ordering.

**SETTLED (2026-08-16): the SDK invokes the tool body AFTER `respond` returns.**
Two tests prove it, one with scripted doubles and one with real weights.

If it turns out the SDK does call tools after `respond` returns, then this card is wrong and the hang has another cause — please say so on the card rather than closing it silently, because the consumer-side evidence is real and would then need a different explanation.

## Directions worth considering

Not prescriptive — the fork owns this call:

- release the container across a tool round and re-acquire for the continuation, so the lock covers generation rather than the whole turn;
- or make the lock re-entrant for the same task tree;
- or document plainly that a tool body must not generate on its own container, and give hosts a way to detect it rather than hang. A clear error beats a silent park.

A silent park is the worst of the three. A host cannot tell it from a slow model, and a host that has deliberately removed its timeouts — ours has, for discovery — waits forever.

**THE DIRECTION TAKEN (2026-08-16).** None of the three above is taken, because
the code already behaves correctly: it releases the container across a tool round
by itself. The direction taken is the documentation half of direction 3, with the
statement INVERTED. The code now states the true guarantee at its tool path: a
tool body CAN generate on the same container. Two independent sessions read this
code and both concluded that a deadlock exists, thus the code misled a reader.
The comment removes that trap. It costs nothing at run time.

## Acceptance Criteria

- [x] A test that generates from inside a tool body on the same container either completes or fails loudly; it does not hang
- [x] Whichever direction is chosen, the constraint on tool bodies is stated in `MLXLanguageModel`'s own documentation, where a host implementer will read it
- [x] If the premise is wrong (tools are invoked after `respond` returns), that is recorded here with what the real cause is

Criterion 2 is met by the comment at the tool path of
`Libraries/MLXFoundationModels/MLXLanguageModel.swift`, above
`if !enabledToolDefinitions.isEmpty`. It is the entry of BOTH tool branches
(`.allowed` and `.required`), thus it sits with the two emit sites that a reader
who hunts for the deadlock will examine. What it states is the guarantee, not a
constraint: a tool body can start a second generation on the same model, because
the `perform` closure sends the tool-call delta and then returns, and because
each continuation round opens its own `perform`. It names the two tests that
guard the guarantee.

#eventplan