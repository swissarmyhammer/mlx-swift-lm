---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m05q885vqg76fqjfhmzq5vjp
  text: |-
    Picked up. Research done, and the premise of the card HOLDS.

    Read of `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`: `ExecutorPromptCachePlan.committed()` is `guard rewindPromptCache(caches, to: promptTokens.count) else { return nil }`. `rewindPromptCache` (`Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`) answers false when a trim does not land, and a `RotatingKVCache` past its window drops no positions. `Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift` already measures that a DeepSeek-V4 cache of 200 tokens answers `canTrimPromptCache == false`. Thus each executor turn of such a model ends with `committed() == nil`.

    How `ChatSession` succeeds (`Libraries/MLXLMCommon/ChatSession.swift`, `Conversation.record`):

        let cachedTokenCount = cachedTokens.count
        if processedTokenCount >= cachedTokenCount,
            processedTokenCount - cachedTokenCount <= generatedTokens.count
        {
            let committedGeneratedTokenCount = processedTokenCount - cachedTokenCount
            cachedTokens.append(
                contentsOf: generatedTokens.prefix(committedGeneratedTokenCount))
            uncommittedTokens = Array(
                generatedTokens.dropFirst(committedGeneratedTokenCount))
        } else {
            cachedTokens.removeAll()
            uncommittedTokens.removeAll()
        }

    It WIDENS the ledger to the cache. The executor TRIMS the cache to the ledger. They are opposite directions on the same seam, and only one of them needs a rewind. `ChatSession` calls `rewindPromptCache` nowhere.

    Two facts that make the widened ledger safe:

    1. `TokenIterator.next()` calls `step(previous: previousY)` BEFORE it returns `previousY`, thus every token the iterator returns is already in the cache. `generateLoopTask` records each of those tokens (`tokenCollector.record(token)`), and `discardGeneratedToken()` is telemetry alone. The recorded list is therefore exactly the tokens the caches took.
    2. A widened ledger cannot regress a model that DOES rewind. When the next render breaks the prefix at the seam, `RewindToCommonPrefixRule` trims back to the common prefix, which is at least the render this turn -- the same position the eager rewind reaches today. When the render extends the ledger, `ExtendCachedPrefixRule` reuses MORE than today.

    Plan: `committed(generatedTokens:)` keeps the ledger of the render plus the generated tokens the caches took, and no rewind. The three executor generation paths must supply the recorded tokens.
  timestamp: 2026-08-16T16:42:35.067041+00:00
- actor: claude-code
  id: 01m05rxgfhnb7tj0cygb4yevxa
  text: |-
    Implementation landed. What changed, and the RED-then-GREEN evidence.

    ## The correction

    `ExecutorPromptCachePlan.committed(generatedTokens:)` no longer rewinds. It
    names the position the caches really stand at and widens the ledger to it, which
    is what `ChatSession.Conversation.record` does:

        guard let position = caches.first?.offset,
            caches.allSatisfy({ $0.offset == position }),
            position >= promptTokens.count,
            position - promptTokens.count <= generatedTokens.count
        else { return nil }
        let committedGeneratedTokenCount = position - promptTokens.count
        return ExecutorPromptCacheEntry(
            caches: caches,
            tokens: promptTokens + generatedTokens.prefix(committedGeneratedTokenCount))

    The three generation paths of the executor now hand it the tokens they
    generated:

    * `generateProtocolTokensTask` (MLXLMCommon/Evaluate.swift) answers
      `Task<[Int], Never>` in place of `Task<Void, Never>`. It already recorded
      nothing; it now carries `RecordingGeneratedTokens()`. Both callers --
      `runAllowedToolGeneration` and `runReasoning` -- read the value.
    * A new `package func generateTaskRecordingTokens(input:cache:state:...)`
      answers `(AsyncStream<Generation>, Task<[Int], Never>)`. `runUnconstrained`
      takes it: the text stream alone carries no token identifiers, thus that path
      could leave no ledger at all without it.

    Why the recorded list and not the `.token` events of the loop: the recorder runs
    INSIDE the generation task, thus it holds every token the iterator fed, even the
    ones the loop never drained after a `break generationLoop`. A count read from the
    loop would be short of the cache and the guard would answer nil, which is a
    COLD turn. The recorder keeps the count exact.

    ## RED, then GREEN

    Unit, no weights (`Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift`,
    5 new tests). With the rewind still in place, 4 of the 5 failed:

        ✘ "a cache past its sliding window still carries a ledger to the next turn"
        ✘ "a cache that rewinds carries the same ledger"
        ✘ "a token the caches did not take stays out of the ledger"
        ✘ "caches that hold more than the pass generated leave the session cold"

    Real weights, through the executor, on `mlx-community/gemma-3-270m-it-4bit`
    (`IntegrationTesting/.../TextGeneration/PromptCacheReuseChannelTests.swift`).
    The first turn renders 3512 tokens, past the 512-token window, thus the cache
    cannot rewind.

        BEFORE: sliding-window round 1: prompt 3512, cached 0
                sliding-window round 2: prompt 3553, cached 0, fed 3553   FAILED
        AFTER:  sliding-window round 1: prompt 3512, cached 0
                sliding-window round 2: prompt 3553, cached 3536, fed 17  PASSED

    3536 stands 24 tokens PAST the 3512-token render, which is the ledger carrying
    the generated tokens. Compare card ^mscrreq on `ChatSession`: 3509 of 3525.

    ## No regression on a cache that DOES rewind

    The Llama round of the same suite, which was already green:

        round 1: prompt 45, cached 0
        round 2: prompt 63, cached 47, fed 16
        cold control: prompt 63, cached 0

    47 stands 2 tokens past the 45-token render. The widened ledger reaches further
    than the old rewind, and it cannot reach less: when the next render breaks the
    prefix at the seam, `RewindToCommonPrefixRule` trims to the common prefix, which
    is at least the render.

    ## Tests

    Five bundles, all green, no warnings from the changed files:
    MLXLMTests 847 + 475, MLXGuidedGenerationTests 70, MLXFoundationModelsTests 160,
    CXGrammarTests 7, MLXHuggingFaceMacrosTests 5. The IntegrationTesting project
    builds.
  timestamp: 2026-08-16T17:11:40.273563+00:00
- actor: claude-code
  id: 01m05rxq3c7b434vgnnjkf4f4n
  text: |-
    ### implement — changed
    - evidence: 6 files — Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Libraries/MLXLMCommon/Evaluate.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift, Tests/MLXLMTests/DeepSeekV4PromptCacheRewindTests.swift, IntegrationTesting/IntegrationTestingTests/MLXFoundationModelsIntegration/TextGeneration/PromptCacheReuseChannelTests.swift. Five bundles green (847+475, 70, 160, 7, 5). IntegrationTesting builds. Real weights: sliding-window round 2 went from `cached 0, fed 3553` to `cached 3536, fed 17`.
    - next: /review. The user must read the DEVIATION section of the description: the real-weights round ran on `mlx-community/gemma-3-270m-it-4bit`, not on `mlx-community/DeepSeek-V4-Flash-4bit`.
  timestamp: 2026-08-16T17:11:47.052163+00:00
- actor: claude-code
  id: 01m05s1yvpvda3cqtjbrqc9548
  text: |-
    ### commit — changed
    - evidence: f661489 fix(foundation-models): widen the prompt-cache ledger instead of rewinding it. Files: ExecutorPromptCache.swift, MLXLanguageModel.swift, MLXLMCommon/Evaluate.swift, ExecutorPromptCacheTests.swift, DeepSeekV4PromptCacheRewindTests.swift, PromptCacheReuseChannelTests.swift, plus this card's kanban state.
    - A second, unrelated board-only commit closed out card ^mscrreq (its code had already landed as 10b350e): 29b4b55 chore(kanban): close ^mscrreq after a clean review.
    - next: `/review` for this card.
  timestamp: 2026-08-16T17:14:06.070632+00:00
- actor: claude-code
  id: 01m05t16scm75s54gswqp21pw5
  text: |-
    ### review — findings
    - evidence: 5 findings (5 confirmed, 1 refuted, 9 attempted) — `review sha f661489~1..f661489` backend local — MLXLanguageModel.swift:1674, 1914, 1918, 2071, 2074
    - next: remove the duplicated task-completion and cache-commit code from the whole file, then run /review again.
  timestamp: 2026-08-16T17:31:09.996244+00:00
position_column: review
position_ordinal: '80'
title: The executor prompt cache gives a sliding-window model no reuse
---
`Libraries/MLXFoundationModels/ExecutorPromptCache.swift` landed with card
`^z1j7f67`. It held its ledger to the rendered PROMPT of the last turn alone,
and it paid for that with a REWIND after each generation:

```swift
func committed() -> ExecutorPromptCacheEntry? {
    guard rewindPromptCache(caches, to: promptTokens.count) else { return nil }
    return ExecutorPromptCacheEntry(caches: caches, tokens: promptTokens)
}
```

`rewindPromptCache` answers false when a cache does not land on the position it
asks for. Its own documentation states the outcome: "A cache past its sliding
window cannot rewind, thus a sliding-window model takes the nil answer."

## Why this is a defect

Every sliding-window model of this repository takes that nil answer -- DeepSeek-V4,
Gemma 3/4, GPT-OSS, Exaone4 and Mistral3 all take a `RotatingKVCache`. Each turn
of such a model therefore ends with `committed() == nil`, and the next turn of the
same session starts cold. The executor gives them NO prompt-cache reuse at all.

Card `^mscrreq` measured what the OTHER path does for the same model on the same
day. `MLXLMCommon.ChatSession` keeps a ledger of the render PLUS the tokens the
turn generated, thus `ExtendCachedPrefixRule` fires with NO rewind:

```
DSV4 SEAM: decision = appendSuffix(suffixStart: 3509, ...);
    prompt = 3525; ledger = 3509; processed = 3509;
    aligned = true; trimmable = false; seam = 3509
```

Read `trimmable = false` beside `seam = 3509`. The cache does not rewind and it
does not have to, because the tokens it already holds ARE a prefix of the next
render. That measurement is the evidence that a rewind is not the only way to
carry a cache from one turn to the next.

## The work

- [x] Prove the gap with the synthetic checkpoint and NO weights: build
      `DeepSeekV4Model.newCache`, feed more tokens than the sliding window, and
      read `rewindPromptCache(caches, to:)`
- [x] Choose how the executor carries a cache a sliding-window model cannot
      rewind. The `ChatSession` answer -- a ledger of the render plus the
      committed generated tokens -- needs no rewind
- [x] Measure the correction with the real weights, through the
      `MLXFoundationModels` executor and not through `ChatSession`. SEE THE
      DEVIATION BELOW: the measurement ran on
      `mlx-community/gemma-3-270m-it-4bit`, not on
      `mlx-community/DeepSeek-V4-Flash-4bit`

## Deviation the user must read

The card names `mlx-community/DeepSeek-V4-Flash-4bit` for the real-weights
measurement. The measurement ran on `mlx-community/gemma-3-270m-it-4bit`
instead, which is the smallest model whose cache rotates. Two reasons:

1. The dispatching agent directed it: "run ONE test per process on the smallest
   model that has a rotating cache".
2. A committed DeepSeek-V4 executor test is unsafe in the shape this repository
   has. `IntegrationTestingTests/DeepseekV4SharedCheckpoint.swift` holds ONE
   shared 141 GiB load that `DeepseekV4IntegrationTests` and
   `DeepseekV4AgenticPromptCacheAssessmentTests` await. The executor holds its
   OWN model container, thus an executor test of the same checkpoint is a
   SECOND 141 GiB load in the same test process.

The mechanism under test is the same: Gemma 3 gives a `RotatingKVCache` to five
of each six layers, which is the cache class DeepSeek-V4 gives to a layer with
no compressor.

## Memory

The DeepSeek-V4 checkpoint holds 141 GiB. Run ONE real-weights test for each
process, or the machine runs out of memory.
#deepseek-v4 #performance

## Review Findings (2026-08-16 12:14)

> Scope: `review sha f661489~1..f661489` — reviewed the diffs only — lines this change added or modified. 6 file(s) reviewed, 0 not reviewed.

- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1674` `duplication/duplication` — Identical method call `promptCache.commit(plan, generatedTokens: await task.value)` appears in three functions (lines 1674, 1918, 2074); cache commitment is duplicated. Extract a shared helper function `commitPromptCache(plan:tokens:)` or pass the task result through a common completion path rather than duplicating the commit call.
- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1914` `duplication/duplication` — Identical statement `_ = await task.value` appears across three generation functions; this line duplicates lines 1670 and 2071. Extract task completion logic into a helper function to avoid maintaining three identical error-handling paths.
- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:1918` `duplication/duplication` — Identical method call `promptCache.commit(plan, generatedTokens: await task.value)` appears across three functions; this line duplicates lines 1674 and 2074. Extract cache commitment into a shared helper to prevent divergence and reduce maintenance burden across three generation paths.
- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:2071` `duplication/duplication` — Identical statement `_ = await task.value` appears across three generation functions; this line duplicates lines 1670 and 1914. Extract task completion handling into a helper function to centralize error-path logic and reduce duplication.
- [ ] `Libraries/MLXFoundationModels/MLXLanguageModel.swift:2074` `duplication/duplication` — Identical method call `promptCache.commit(plan, generatedTokens: await task.value)` appears across three functions; this line duplicates lines 1674 and 1918. Extract cache commitment into a shared helper to ensure consistency and simplify maintenance across runReasoning, runUnconstrained, and runAllowedToolGeneration.
