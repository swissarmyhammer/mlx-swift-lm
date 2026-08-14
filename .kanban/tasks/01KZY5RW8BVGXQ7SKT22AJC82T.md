---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzyeb0fypva52czkpd1vtg36
  text: |-
    ## Assessment: does the official upstream path cache Qwen 3.6 on its own?

    Answer: **NO**. Both facts fail. Measured with real weights on
    `mlx-community/Qwen3.6-27B-mxfp4` through upstream's own `MLXLMCommon.ChatSession`
    only. No `MLXFoundationModels.PromptCache` type is touched.

    New test: `IntegrationTesting/IntegrationTestingTests/Qwen36UpstreamPromptCacheAssessmentTests.swift`

    ### The numbers

    ```
    QWEN36 CACHE: round 1 rendered prompt tokens = 4705
    QWEN36 CACHE: round 1 fed prompt tokens = 4705
    QWEN36 CACHE: round 1 prefill seconds = 13.35
    QWEN36 CACHE: round 1 generated tokens = 24
    QWEN36 CACHE: round 2 rendered prompt tokens = 4748
    QWEN36 CACHE: round 2 extends round 1 = false
    QWEN36 CACHE: round 2 common prefix with round 1 = 4703 of 4705
    QWEN36 CACHE: round 2 fed prompt tokens = 4748
    QWEN36 CACHE: round 2 prefill seconds = 11.60
    QWEN36 CACHE: round 2 tokens skipped by reuse = 0
    QWEN36 CACHE: control fed prompt tokens = 4748
    QWEN36 CACHE: control prefill seconds = 11.61
    ```

    ### (a) Prefix extension -- FAILS

    Round 2's render is not a prefix extension of round 1's. The two share 4703 of
    round 1's 4705 tokens, so the template rewrites the last 2 tokens of round 1's
    prompt -- the generation-priming tail. This is the same trap
    `PromptCacheHybridReuseTests` records.

    ### (b) Skipping the reprocessing -- FAILS

    Round 2 fed all 4748 of its rendered tokens and skipped 0. Its prefill took
    11.60 s against the cold control's 11.61 s for the same prompt -- the same work,
    to within 0.1 percent. Upstream reprocessed everything.

    (b) fails for a second, independent reason. With a 4703-token common prefix,
    `RewindToCommonPrefixRule` would decide `.trimToCommonPrefix`, but that rule
    requires `PromptCacheState.isTrimmable`, which `ChatSession` fills from
    `canTrimPromptCache(kvCache.cache)` = every layer `isTrimmable`. A hybrid
    Mamba/attention stack has recurrent layers that cannot rewind, so the policy
    falls through to `.rebuild`. Even a near-perfect prefix cannot be salvaged by
    upstream's machinery on this architecture.

    ### Structural findings behind the numbers

    - Upstream's `Libraries/MLXFoundationModels/MLXLanguageModel.swift` has NO cache
      across `respond()` calls at all. `ModelCache` holds weights, tokenizers,
      grammar templates and logit biases only. Every generation call site omits the
      `cache:` parameter, so `TokenIterator` calls `model.newCache(...)` fresh each
      turn. `cachedTokenCount` is the literal `0` at all five usage emission sites.
      The only prompt-cache mention is `supportsPromptCacheReuse`, called from tests
      alone.
    - All of upstream's reuse machinery lives in `MLXLMCommon.ChatSession`
      (`Conversation.cachedTokens` ledger + `PromptCacheReusePolicy`). Even there it
      does not fire for Qwen 3.6, per the numbers above.
    - `ChatSession.Conversation.record` appends `.assistant(content, toolCalls:)` and
      never sets `Chat.Message.reasoning`, so upstream does not replay preserved
      thinking into history renders. Our fork added exactly that (kanban 05zt40g)
      to make the prefix stable.

    ### What a port must restore

    1. A cross-turn cache in the FoundationModels `Executor` -- upstream has none.
    2. Transcript-stable-boundary checkpointing, or preserved-thinking history
       replay, to survive the 2-token priming divergence at the turn seam.
    3. Hybrid-checkpoint reuse for a non-trimmable Mamba/attention stack. Upstream's
       trim-based rewind cannot serve this architecture.

    Verdict: we cannot drop `PromptCache.swift` / `PromptCacheChunks.swift`. The port
    this card describes is still needed.
  timestamp: 2026-08-13T20:52:07.294814+00:00
- actor: claude-code
  id: 01kzyekn1xgnc342780gyqyed9
  text: |-
    ### Divergence, decoded

    The new test now decodes the two renders at the first divergent token. This
    names the exact seam a port must close:

    ```
    QWEN36 CACHE: round 1 divergent tail = <<<<think>
    QWEN36 CACHE: round 2 divergent tail = <<<Here's a thinking process:
    ```

    Round 1's prompt ends with the generation-priming block `<think>\n`. Round 2's
    history re-render writes the assistant reply text at that same offset. The two
    renders therefore part company 2 tokens before the end of round 1's prompt.

    The cause is upstream's own transcript accounting:
    `ChatSession.Conversation.record` appends `.assistant(content, toolCalls:)` and
    never fills `Chat.Message.reasoning`, so no history render can put the priming
    block back. Our fork's `preserve_thinking` + `reasoning_content` replay (kanban
    05zt40g) exists exactly to close this gap.

    The suite is committed green as a BASELINE, not as a wish: it asserts the
    measured non-reuse. If a later change makes upstream reuse the cache, the
    assertions fail on purpose, and their messages say to invert them and revisit
    this card.
  timestamp: 2026-08-13T20:56:50.493029+00:00
- actor: claude-code
  id: 01kzyem6hdk21vr7tw8whc8r2j
  text: |-
    ### implement — changed
    - evidence: 1 file — /Users/wballard/github/swissarmyhammer/mlx-swift-lm/IntegrationTesting/IntegrationTestingTests/Qwen36UpstreamPromptCacheAssessmentTests.swift. Commit f85fc50 (not pushed). Test passes: 1 test in 1 suite, 40.9 s, zero warnings in the build.
    - next: the card stays open. The assessment says NO, so the port this card describes is still needed. Do not drop PromptCache.swift or PromptCacheChunks.swift.
  timestamp: 2026-08-13T20:57:08.397811+00:00
position_column: todo
position_ordinal: 9a80
title: Wire the prompt cache into the upstream MLXLanguageModel
---
The upstream catch-up merge (`-X theirs`) replaced our `Libraries/MLXFoundationModels/MLXLanguageModel.swift` (5086 lines) with the official one (2229 lines). Both sides had created that file after the merge base, so the merge took the upstream file whole.

Result: `PromptCache.swift` (1371 lines) and `PromptCacheChunks.swift` (732 lines) still exist, but `MLXLanguageModel.swift` has **zero** references to them. The prompt cache is dead code.

Only `supportsPromptCacheReuse` was put back, so the tree builds.

## What is missing

The pre-merge file held about 15 prompt-cache members, which the tag `pre-upstream-merge-2026-08-13` still holds:

- `private static let promptCache = PromptCache()`
- `resolvePromptCache`, `storePromptCache`, `removePromptCache`
- `setPromptCacheChunkSize`, `setPromptCacheByteBudget`
- `populatePromptCacheChunks`
- `PromptCacheSlot`, `resolvePromptCacheIfTextOnly`, `makePromptCacheSlot`
- `prefillPromptCache`, and two `commitPromptCache` overloads
- the calls to all of these in the `Executor` generation path

## Why it is not a simple re-apply

Upstream rewrote the `Executor`. The prefill and commit points the old code hooked no longer look the same, so this needs a real port against the new generation loop, not a patch.

## Done when

- `MLXLanguageModel` reuses the prompt cache again for both mechanisms: chunk reuse and hybrid checkpoint reuse.
- The three `PromptCacheHybrid*` suites pass with real weights.
- A second round reports a non-zero input `cachedTokenCount`.

Recover the old code with:
`git show pre-upstream-merge-2026-08-13:Libraries/MLXFoundationModels/MLXLanguageModel.swift` #upstream-catch-up-prompt-cache