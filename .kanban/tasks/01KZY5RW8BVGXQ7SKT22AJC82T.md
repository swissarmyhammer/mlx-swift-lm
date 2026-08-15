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
- actor: claude-code
  id: 01kzyw57m2hgfg52bhm47c1x6j
  text: |-
    ### Two properties a port must keep — from cards ^3z5pydx and ^qsv1415

    Both cards closed on 2026-08-14, and each leaves one requirement for this port.

    **1. The chunk key must stay a chain (from `^3z5pydx`).**

    `PromptCacheChunks.chunkKey(parentKey:tokens:)` hashes the PARENT key together with this chunk's tokens, thus each key holds the whole token prefix from position zero. That is what makes it safe to share chunks across the conversations of one model: a conversation resolves another conversation's chunk only when it presents the same tokens, and then the state is its own state.

    A port that hashes only a chunk's own tokens — to save work, or to share more — lets a session take KV state for tokens it never rendered. `^3z5pydx` asked whether a session could act as though a tool call had already happened. With the chained key the answer is no. Without it the answer becomes yes.

    Do not "simplify" this key.

    **2. Clearing must not need an eviction (from `^qsv1415`).**

    `PromptCache.swift` holds `remove(modelID:)` at line 677 and `evictAll()` at line 650, and **each member of that file is internal**. Before the catch-up the only public lever that cleared a prompt cache was `MLXLanguageModel.evictAll()`, which also drops the weights, the tokenizers and the constraint templates. A consumer that wanted one clean cache paid a full model reload — minutes for a 27B model.

    Give this port a public `clearPromptCache(modelID:)`, or make `remove(modelID:)` public. Without it the port rebuilds the wall that `^qsv1415` was filed against.
  timestamp: 2026-08-14T00:53:38.050381+00:00
- actor: claude-code
  id: 01kzywcbt3y7ppqy8749m5avce
  text: |-
    ### Abandoned on 2026-08-14 — Qwen is written off for the time being

    The user chose to drop this port and to put the effort into DeepSeek-V4 instead, with working caching for agentic and multi-tool use.

    **Nothing was ported.** Upstream's `MLXLanguageModel` still keeps no cross-turn prompt cache, thus each `respond()` builds a cache of its own and drops it. Qwen 3.6 pays a full prefill on each turn: 11.59 s for round two against a cold control of 11.60 s, 0 tokens skipped of 4748.

    **What stays in the tree, dead:**

    - `Libraries/MLXFoundationModels/PromptCache.swift`, 1371 lines
    - `Libraries/MLXFoundationModels/PromptCacheChunks.swift`, 732 lines
    - `Tests/MLXFoundationModelsTests/PromptCacheHybridExecutorTests.swift.disabled`, the 9 tests of card `^tbyb0dy`

    They build, and no caller names them. They are kept, not deleted, because the user said "for the time being": a later reader can wire them in without recovering anything. The tag `pre-upstream-merge-2026-08-13` also holds each one.

    **What a later port must know**, so that this work is not learned twice:

    1. Upstream's FoundationModels `Executor` holds no cross-turn cache. This is not a wiring gap; there is nothing to wire to.
    2. Qwen 3.6 breaks prefix extension at the turn seam by 2 tokens. Round one ends `<think>`, round two ends `Here's a thinking process:`, because `ChatSession.Conversation.record` appends `.assistant(content, toolCalls:)` and never fills `Chat.Message.reasoning`.
    3. A hybrid Mamba/attention stack is not trimmable, thus `RewindToCommonPrefixRule` answers `.rebuild` even for the 4703-token common prefix that does survive.
    4. Keep `chunkKey` a chain of hashes. Card `^3z5pydx` says why.
    5. Make `remove(modelID:)` public. Card `^qsv1415` says why.

    The measurement that gives these numbers is `IntegrationTesting/IntegrationTestingTests/Qwen36UpstreamPromptCacheAssessmentTests.swift`, commit `f85fc50`. It runs, and it will answer the same question for another model with a change of model id.
  timestamp: 2026-08-14T00:57:31.715098+00:00
position_column: done
position_ordinal: fb80
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