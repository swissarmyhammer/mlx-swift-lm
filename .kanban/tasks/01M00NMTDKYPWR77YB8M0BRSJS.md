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
position_column: doing
position_ordinal: '8380'
title: One turn holds the container lock across tool rounds, so a tool that generates deadlocks
---
Filed by the `FoundationModelsMultitool` session, with `FoundationModelsRouter` concurring after independently reading the same code.

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
- **Not memory, not the model, not the grammar.** See the numbers above.

## What is not proven

That the SDK invokes a tool body while the `respond` executor call is still suspended inside `perform`, rather than after it returns. Reading alone did not settle it. The hang is the strongest evidence that it does, since a deadlock requires exactly that ordering.

If it turns out the SDK does call tools after `respond` returns, then this card is wrong and the hang has another cause — please say so on the card rather than closing it silently, because the consumer-side evidence is real and would then need a different explanation.

## Directions worth considering

Not prescriptive — the fork owns this call:

- release the container across a tool round and re-acquire for the continuation, so the lock covers generation rather than the whole turn;
- or make the lock re-entrant for the same task tree;
- or document plainly that a tool body must not generate on its own container, and give hosts a way to detect it rather than hang. A clear error beats a silent park.

A silent park is the worst of the three. A host cannot tell it from a slow model, and a host that has deliberately removed its timeouts — ours has, for discovery — waits forever.

## Acceptance Criteria

- [ ] A test that generates from inside a tool body on the same container either completes or fails loudly; it does not hang
- [ ] Whichever direction is chosen, the constraint on tool bodies is stated in `MLXLanguageModel`'s own documentation, where a host implementer will read it
- [ ] If the premise is wrong (tools are invoked after `respond` returns), that is recorded here with what the real cause is

#eventplan