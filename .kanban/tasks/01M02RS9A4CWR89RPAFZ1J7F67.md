---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m05bnnyrfyjm1w6vveh08r9m
  text: |
    ### Research, and the design decision it forced

    **The seam of `^mscrreq` is avoidable, and the way to avoid it is the ledger.**

    `ExtendCachedPrefixRule` compares a new render against the ledger. `ChatSession`
    writes `render + generated tokens` into that ledger, and a later render of the
    same turn need not tokenize into the identifiers the model sampled. That is the
    one-token break `^mscrreq` measured, and no rule can repair it.

    This card therefore writes a DIFFERENT ledger: the rendered PROMPT of the turn,
    and nothing else. After generation the caches rewind to the prompt length, thus
    the ledger and the caches agree. Round 2 then compares render against render,
    which is exactly the comparison `^mscrreq` recorded as `follow-up extends round
    1 = true`.

    The price is the rewind. `rewindPromptCache` verifies that every cache landed on
    the requested position and answers `false` when one did not. A rotating cache
    past its window answers `false`, thus a sliding-window model gets no entry and
    its next turn starts cold with a reported count of 0. That is correction "C" of
    `^mscrreq` for those models, and it is the safe answer: a wrong reuse gives a
    silently wrong generation.

    **Reuse of the existing machinery, not a new policy.**

    `PromptCacheReusePolicy`, `ExtendCachedPrefixRule` and `RewindToCommonPrefixRule`
    are `internal` to `MLXLMCommon`, thus `MLXFoundationModels` cannot name them. Two
    `package` functions in `PromptCacheReusePolicy.swift` wrap them instead of
    widening six types:

    - `reusablePromptPrefix(promptTokens:cachedTokens:caches:)` asks the standard
      policy, applies a rewind when the decision asks for one, and answers with the
      number of leading tokens the caches hold.
    - `rewindPromptCache(_:to:)` trims and verifies.

    No protocol rule is consulted. Those rules (`HarmonyToolRestartRule`,
    `OnyxToolRestartRule`) serve a ledger that holds generated tokens; this ledger
    holds a render.

    **Session identity.**

    The framework hands the executor no session identifier. `Transcript.Entry.id` is
    stable, thus the key is `(modelID, first transcript entry id)`. A
    `LanguageModelSession` keeps its entries and appends to them, thus every turn of
    one session names the same key and two sessions never collide. An empty
    transcript gets no key and carries no cache.

    **What is NOT wired, and why.**

    `GuidedGenerationLoop.run` builds its own `KVCacheStorage` and takes no cache
    from a caller. The schema path and the required-tool path therefore carry
    nothing, and `ExecutorPromptCacheSlot.carriesNoCache()` says so at those two
    sites. Their reported count is a runtime value that is 0 because 0 is the
    measured truth for a pass that owns its cache. Giving that loop a carried cache
    changes a public API of a second module (`MLXGuidedGeneration`) and belongs to
    its own card.
  timestamp: 2026-08-16T13:20:12.248403+00:00
- actor: claude-code
  id: 01m05bp8sd0rqhspw2aq3ka24b
  text: |
    ### implement — changed

    **RED first, on all three fronts.** The two new `package` functions and
    `ExecutorPromptCacheStore.checkIn` were reduced to their no-reuse answers, and
    the suites were run against that state:

    - `MLXLMTests`: `✘ Test run with 844 tests in 71 suites failed after 81.776
      seconds with 5 issues.` -- 5 expectations of `PromptCachePrefixReuseTests`
      (`prefix == 2`, `rewindPromptCache(caches, to: 6)`, ...).
    - `MLXFoundationModelsTests`: `✘ Test run with 155 tests in 24 suites failed
      after 0.160 seconds with 6 issues.` -- 6 expectations of
      `ExecutorPromptCacheTests` (check-in, isolation, the bound, eviction).
    - `IntegrationTestingTests/PromptCacheReuseChannelTests` with the real weights of
      `mlx-community/Llama-3.2-1B-Instruct-4bit`:

      ```
      round 1: prompt 45, cached 0
      round 2: prompt 63, cached 0, fed 63
      cold control: prompt 63, cached 0
      ✘ Expectation failed: secondTurn.cachedTokenCount > 0
      ✘ Expectation failed: warmFedTokenCount < coldTurn.promptTokenCount
      ```

    **GREEN after the implementations landed.** The same real-weights test:

    ```
    round 1: prompt 45, cached 0
    round 2: prompt 63, cached 45, fed 18
    cold control: prompt 63, cached 0
    ✔ Test run with 1 test in 1 suite passed after 2.081 seconds.
    ```

    The second turn feeds 18 tokens where a cold turn feeds 63, the reused count is
    the whole 45-token render of round 1, and `secondTurn.text == coldTurn.text`
    holds under greedy sampling, thus the answer did not change.

    **The five bundles, 0 failures.**

    | Bundle | XCTest | Swift Testing |
    | --- | --- | --- |
    | `MLXLMTests` | 475 | 844 |
    | `MLXGuidedGenerationTests` | 0 | 70 |
    | `MLXFoundationModelsTests` | 0 | 155 |
    | `CXGrammarTests` | 0 | 7 |
    | `MLXHuggingFaceMacrosTests` | 5 | 0 |

    480 XCTest + 1076 Swift Testing = 1556, against a baseline of 1530. The 26 new
    tests are 12 in `PromptCachePrefixReuseTests` and 14 in
    `ExecutorPromptCacheTests`. `TurboQuantIntegrationTests` did NOT fail on this
    run.

    **Regression cover for the paths that now carry a cache.** Each ran against real
    weights and passed:

    - `UpdateUsageEmissionTests` -- unconstrained, guided and tool-calling usage.
    - `PlainChatGenerationTests`.
    - `MultiTurnToolCallingTests` -- 5 tests, both the Gemma-4 and the Qwen3 family,
      which is the cross-turn tool path the carried cache touches most.
    - `ReasoningIntegrationTests` and `StreamingDeltaTests` -- 10 tests.

    **Builds.** `swift build` complete with no warning in a changed file.
    `xcodebuild build-for-testing` on the package and on `IntegrationTesting` each
    report `TEST BUILD SUCCEEDED`. `swift-format lint` is clean on every changed
    file.

    - evidence: 7 files -- 3 new
      (`Libraries/MLXFoundationModels/ExecutorPromptCache.swift`,
      `Tests/MLXLMTests/PromptCachePrefixReuseTests.swift`,
      `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`,
      `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseChannelTests.swift`),
      4 changed (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`,
      `Libraries/MLXFoundationModels/MLXLanguageModel.swift`,
      `IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/Support/FMTestHelpers.swift`).
    - next: `/review`.
  timestamp: 2026-08-16T13:20:31.533464+00:00
- actor: claude-code
  id: 01m05bpnk4rsfwsq9vyrjh4g9y
  text: |
    Correction to the evidence line above: the counts are swapped. The change is 4
    NEW files and 3 CHANGED files. The file list itself is complete and correct.
  timestamp: 2026-08-16T13:20:44.644753+00:00
position_column: doing
position_ordinal: '8580'
title: Reuse the prompt cache across turns inside MLXLanguageModel.Executor
---
## What

Card `^z2996cp` restored the usage channel send and gave it a test. It could not
give `cachedTokenCount` a real value, because the executor reuses nothing. A
person split that feature into this card on 2026-08-15.

## The measurement that made this card

Taken on 2026-08-15, on branch `catch-up-upstream`:

| The question | The measurement |
| --- | --- |
| Does `MLXFoundationModels` hold cache-reuse machinery? | `grep -r "KVCache\|cachedTokens\|PromptCacheReuse\|ChatSession" Libraries/MLXFoundationModels/` gives **0** matches |
| Does the executor pass a cache? | Every generation entry point takes the default `cache: nil` |
| Does the executor reuse a prompt? | `respond(...)` renders the whole transcript again on each call |
| Does `GenerateCompletionInfo` hold the count? | It holds no reuse field |

Thus every turn pays a full prefill, and
`Libraries/MLXFoundationModels/MLXLanguageModel.swift`
`Executor.reusedPromptTokenCount` is 0 because 0 is true, not because a value is
missing.

## What the consumers see

`FoundationModelsRouter` holds two assertions that stay red until this card
lands, in `LanguageModelSessionBackendTests.swift`:

- `:563` `turn2Usage.input.cachedTokenCount > 0`
- `:574` `abs(turn2Usage.input.cachedTokenCount - turn1ProcessedTokenCount) <= tolerance`

A consumer that starts a second turn on the same session pays the whole prompt
again. On a long agentic transcript that cost is the largest part of a turn.

## Where the parts already are

`MLXLMCommon` holds the machinery, and `ChatSession` is the only consumer of it:

- `PromptCacheReusePolicy`, `ExtendCachedPrefixRule` and `RewindToCommonPrefixRule`
  decide what a new render may reuse.
- The reused count is computed inside that policy, and the value dies in its
  `switch`.
- `KVCache.isTrimmable` states whether a layer can rewind.

## What to watch out for

Card `^mscrreq` holds a measured reason a good prefix is still not sufficient:
`ExtendCachedPrefixRule` compares a new render against the LEDGER, which is the
earlier render PLUS the tokens that turn generated. An encoder that primes a
generation region breaks that prefix by one token. Read `^mscrreq` before you
design this.

Memory of this project also records: template priming breaks the prompt cache,
thus prove reuse with a real-weights two-round test and not with a unit test
alone.

## What the implementation does

The ledger holds the rendered PROMPT of the last turn, and nothing else. After
generation the caches rewind to the prompt length, thus the next turn compares
its own render against a render. This removes the seam `^mscrreq` measured: a
ledger that holds generated tokens can never be a prefix of a later render.

A cache that cannot rewind -- a sliding-window cache past its window -- gives no
entry, and the next turn of that session starts cold and reports 0.

## Acceptance Criteria

- [x] `MLXLanguageModel.Executor` holds a key/value cache across the turns of one
      session, and invalidates it when the transcript is not an extension.
- [x] `emitUsage` reports the true number of reused tokens at each of its five
      call sites, and `Executor.reusedPromptTokenCount` leaves.
- [x] A test proves reuse through the CHANNEL, the way
      `Tests/MLXFoundationModelsTests/UsageChannelSendTests.swift` does.
- [x] A real-weights two-round test shows the second turn feeds fewer tokens than
      the first, and that its answer is unchanged.
- [x] `swift build` is clean, and the five test bundles are green with no new
      warning. NOTE: `swift test` is stale on this card -- it dies at the first
      GPU test on the metallib. `CLAUDE.md` holds the correct procedure.
- [x] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
      gives `TEST BUILD SUCCEEDED`.

#eventplan