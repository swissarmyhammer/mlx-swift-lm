---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m05awytexcbeg1w3hnt1rtn8
  text: |-
    ### The hypothesis is narrower now — test THIS

    The `FoundationModelsMultitool` session sent its instrumented diagnosis on
    2026-08-16. It replaces the hypothesis this card was filed with.

    **Nested GRAMMAR-CONSTRAINED decode on one resident container deadlocks.** Not
    nested generation of any kind. The guided path alone.

    ### The trace that names it

    `os_log` spans, enter and exit on each call, both slots on
    `mlx-community/Muse-Glimmer-30B-4bit`:

    ```
    07:41:30.650 enter SearchToolsTool.call #1        task=Get current temperature...
    07:41:30.650 enter SearchToolsTool.search #2      limit=1
    07:41:30.650 enter SearchToolsTool.makeSelectionSession #3
    07:41:30.651 exit  SearchToolsTool.makeSelectionSession #3   returned
    07:41:30.651 enter AgentSession.fork #4           role=selection
    07:41:30.651 exit  AgentSession.fork #4           returned
    07:41:30.651 enter AgentSession.respond #5        role=selection promptCharacters=49
    ```

    Nothing follows. After 4 minutes 32 seconds: 0.0% CPU, 19.7 GiB resident, and
    the trace still holds those nine lines.

    ### What the trace rules OUT

    - **The fork is not the hang.** `fork #4` entered and returned inside 1 ms.
    - **Session construction is not the hang.** `makeSelectionSession` entered and
      returned inside 1 ms, thus no grammar COMPILE blocks.
    - **The snippet path never ran.** `MultiTool.call`, `RunBinding.invoke` and
      `WaitTool.call` were never entered.

    ### What it names

    `AgentSession.respond` on the forked selection session, which reaches the MLX
    constrained decode. `promptCharacters=49` states that prompt assembly finished,
    thus the block is inside the decode.

    The selection tier is the ONLY decode under a grammar in that run, it runs on
    the container the outer turn is mid-generation on, and it is the one call that
    never returns.

    ### The test this card now asks for

    `ToolBodyContainerReentryTests`, with two changes:

    1. the nested call is constrained by a GRAMMAR (the outer call need not be), and
    2. a real model doing real MLX evaluation.

    **Try the stub with a grammar FIRST.** The peer states it plainly: if a stub
    under a grammar also hangs, the reproduction needs no weights at all, and that
    is a better test than any real-weights one.

    ### The lead this joins

    `^m0brsjs` recorded a third lead that nobody pursued: the guided/xgrammar path
    holds shared per-model caches. That hazard leaves the moment two slots name
    different models — which is exactly the workaround that makes the hang go away.
    Two independent observations now point at the same place.

    ### What each outcome means

    - It hangs → this repository holds the defect, with a deterministic
      reproduction, and `^m0brsjs` reopens.
    - It passes → the hypothesis is refuted at the MLX layer, `^m0brsjs` stays
      closed on evidence that covers a grammar and real weights, and the peer keeps
      looking on their side.

    The peer is instrumenting in parallel and will send the result whichever way it
    points. They changed nothing in this tree.
  timestamp: 2026-08-16T13:06:42.126705+00:00
- actor: claude-code
  id: 01m05bkc3ez4p7e012jknaabqb
  text: |
    ### hypothesis refuted — from the peer session `foundationmodelsmultitool-00`

    The deadlock is NOT in this repository. `FoundationModelsRouter` found the cause.

    - Router holds a per-container `generationGate`, an `AsyncSemaphore(value: 1)`.
      `beginTurn()` takes the permit and releases it only after the COMPLETE turn,
      with all tool rounds. A nested `respond` on that container waits for a permit
      that cannot become free until the tool returns, and the tool cannot return
      until the nested `respond` returns.
    - The wait is a bare `withCheckedContinuation` with no cancellation handler.
      This is why it is silent, uses no CPU, and continues after cancellation.
    - Router confirms the gate is a throughput device, not a safety device.

    This repository is correct. `MLXLanguageModel.swift:1238-1274` sends the
    tool-call delta and RETURNS OUT of `container.perform`, thus the SDK runs the
    tool body while the container is free. `ToolBodyContainerReentryTests` asserts
    this behavior.

    Both earlier peer hypotheses are dead:
    - "nested generation on the same model deadlocks" — no.
    - "nested grammar-constrained decode deadlocks" — no; the grammar is not related.

    ## What stays

    One true gap remains, and it is the only remaining scope of this card: no test
    in this repository drives generation from inside a tool call against a REAL
    model. `ToolBodyContainerReentryTests` uses a stub.
  timestamp: 2026-08-16T13:18:56.622901+00:00
- actor: claude-code
  id: 01m05e9px65atcn6hrf6bw3zjj
  text: |
    ### research — what the new test must look like

    Read the card, both comments, `Tests/MLXFoundationModelsTests/ToolBodyContainerReentryTests.swift`,
    and the `IntegrationTesting` conventions.

    **Where the file goes.** `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/`.
    The xcodeproj uses `PBXFileSystemSynchronizedRootGroup` (4 of them), thus a new
    file needs no project edit.

    **The model.** `TestFixtures.llamaModelID` = `mlx-community/Llama-3.2-1B-Instruct-4bit`.
    It is in the local hub cache and it is 680 MB, thus it is the smallest of the
    three the card names. Its chat template renders tools (`tools_in_user_message`),
    an assistant `tool_calls` message, and a `tool`/`ipython` message, thus the
    tool round and the continuation round both render.

    **How the tool call becomes deterministic.** A 1B model does not reliably choose
    to call a tool. `StructuredToolOutputSessionTests` shows the pattern that removes
    the doubt: a `LanguageModelSession.DynamicProfile` with a `@SessionPropertyEntry`
    counter. Round one uses `.toolCallingMode(.required)`, which drives the
    grammar-constrained tool-call path and thus always emits a call; `.onToolCall`
    increments the counter; round two uses `.toolCallingMode(.disallowed)`, which
    stops the loop.

    **How the timeout works.** `.timeLimit` alone is not enough. A task parked on the
    container lock cannot be cancelled, thus the test must never await that task.
    The unit test's shape is correct and this test copies it: the turn runs in its
    own `Task`, it reports through an `AsyncStream`, and a task group races the
    stream against `Task.sleep`. Both children of the group are cancellable, thus
    the group always finishes.

    **Confirmed from the macOS 27 SDK interface** (`FoundationModels.swiftmodule/arm64e-apple-macos.swiftinterface`):
    `LanguageModelSession.respond(to:options:)`, `GenerationOptions(samplingMode:temperature:maximumResponseTokens:toolCallingMode:)`,
    `DynamicProfile.model(_:)`, `.toolCallingMode(_:)`, `.onToolCall(perform:)`, and
    the `SessionPropertyEntry` macro all exist as the plan uses them.
  timestamp: 2026-08-16T14:06:05.734316+00:00
- actor: claude-code
  id: 01m05eywas3erz8dk9n9q97qym
  text: |
    ### the test passes on real weights — model, runtime, and the two mutation proofs

    **New file.** `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/ToolBodyContainerReentryRealModelTests.swift`

    **Model.** `mlx-community/Llama-3.2-1B-Instruct-4bit` (`TestFixtures.llamaModelID`),
    680 MB in the local hub cache. This is the smallest of the three the card names.
    The DeepSeek-V4 checkpoint was NOT loaded.

    **Result.** PASSED. It did NOT hang.

    ```
    ✔ Test "Setup: release GPU state from prior suites" passed after 0.023 seconds.
    ✔ Test "A tool body may generate on the same real model while its turn is in flight" passed after 3.697 seconds.
    ✔ Test run with 2 tests in 1 suite passed after 3.722 seconds.
    ** TEST EXECUTE SUCCEEDED **
    ```

    The complete `xcodebuild test-without-building` step is 9.6 s of wall time.
    Three real MLX generations run inside those 3.697 s: the tool-call round of the
    outer turn, the nested round the tool body starts, and the round that concludes
    the turn.

    **Shape.** The outer session uses a `LanguageModelSession.DynamicProfile` with a
    `@SessionPropertyEntry` counter. Round one sets `.toolCallingMode(.required)`,
    thus the grammar constrains generation to a real tool call and a 1B model cannot
    decide to skip it. `.onToolCall` increments the counter, thus each round after
    the tool output uses `.toolCallingMode(.disallowed)` and the turn ends. The tool
    body opens a second `LanguageModelSession` on the SAME `MLXLanguageModel` and
    awaits `respond`, thus it generates on the one resident container while the
    outer turn is in flight.

    **Timeout.** The turn runs in a task of its own and reports through an
    `AsyncStream`. A task group races that stream against `Task.sleep(for: 300 s)`.
    Both children are cancellable, thus a park fails the test instead of blocking.
    The suite also carries `.timeLimit(.minutes(10))`.

    ### The two mutations that prove the test can fail

    1. **`turnTimeout` set to 1 ms.** The test FAILED at the timeout arm with the
       park message — the arm that catches a deadlock does fire.
       `✘ … recorded an issue at ToolBodyContainerReentryRealModelTests.swift:209:25`
    2. **Tool body replaced by `return "MUTATION \(topic)"`, with no nested
       generation.** The test FAILED with two issues: `log.callCount → 0` and
       `log.generatedText → nil`. The nested-generation assertions are load-bearing.

    Both mutations were reverted, and the test is green again.

    ### Other checks

    - `swift-format format --in-place --configuration .swift-format` — no change.
    - `swiftlint 0.65.0` over the file, with `cyclomatic_complexity` 15,
      `function_body_length` 250, `closure_body_length` 250,
      `no_magic_numbers` (`allowed_numbers: [0, 1, -1, 100]`) and `missing_docs`
      (`warning: [open, public]`) — 0 findings.
    - The five package bundles — `MLXLMTests` (844 Swift Testing + 475 XCTest),
      `MLXGuidedGenerationTests` (70), `MLXFoundationModelsTests` (155),
      `CXGrammarTests` (7), `MLXHuggingFaceMacrosTests` (5) — all pass, 0 failures.
      The stub `ToolBodyContainerReentryTests` still passes beside the new one.
  timestamp: 2026-08-16T14:17:39.417890+00:00
- actor: claude-code
  id: 01m05ezejsp97669w84x0b4bk6
  text: |
    ### implement — changed
    - evidence: 1 file — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/ToolBodyContainerReentryRealModelTests.swift; new suite passes in 3.697 s on mlx-community/Llama-3.2-1B-Instruct-4bit; five package bundles pass with 0 failures; swiftlint 0.65.0 gives 0 findings
    - next: /review
  timestamp: 2026-08-16T14:17:58.105041+00:00
- actor: claude-code
  id: 01m05fgt8bez5m9ttf3sgx5fkg
  text: |
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (9a987e2), backend local — 0 findings, 0 confirmed, 0 refuted; 9 validator passes attempted, 0 failed, 0 skipped; 1 file reviewed (IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/ToolCalling/ToolBodyContainerReentryRealModelTests.swift), 0 not reviewed
    - next: none — the card moves to done
  timestamp: 2026-08-16T14:27:27.115009+00:00
- actor: claude-code
  id: 01m05fh71m7tn8eyhghv1vdb3d
  text: |
    ### finish iteration 1 — clean
    - implement: changed — 1 new file, `ToolBodyContainerReentryRealModelTests.swift`
    - test: green — the new test passes after 3.697 s on `mlx-community/Llama-3.2-1B-Instruct-4bit`; five package bundles 0 failures
    - commit: 9a987e2
    - review: clean — 0 findings, 9 validator passes, 1 file
    - next: the card is in `done`
  timestamp: 2026-08-16T14:27:40.212378+00:00
position_column: done
position_ordinal: ff8880
title: Add a tool-body re-entrancy test that uses a real model
---
## Scope

`ToolBodyContainerReentryTests` proves that the executor releases the model
container across the tool call. It drives a scripted tokenizer and a stub
model, thus it does no MLX evaluation.

No test in this repository drives generation from inside a tool call against a
real model. This card closes that gap.

## Background — the deadlock hypothesis is dead

An earlier version of this card kept a hypothesis that a nested generation on
the same model deadlocks in MLX. It is refuted. The peer session found the
cause in `FoundationModelsRouter`: a per-container `AsyncSemaphore(value: 1)`
that `beginTurn()` holds for the complete turn. See the comment ledger.

This repository is correct. `MLXLanguageModel.swift:1238-1274` sends the
tool-call delta and returns out of `container.perform` before the SDK runs the
tool body.

## The test

Give `ToolBodyContainerReentryTests` the same shape, but with a real cached
model that does real MLX evaluation. Use the smallest cached model available.
Put it in `IntegrationTesting`, because it needs weights.

## Acceptance Criteria

- [x] A re-entrancy test exists that drives real MLX evaluation, not a stub
- [x] The test passes, and its record names the model and the runtime
- [x] `^m0brsjs` is annotated with that evidence

#eventplan