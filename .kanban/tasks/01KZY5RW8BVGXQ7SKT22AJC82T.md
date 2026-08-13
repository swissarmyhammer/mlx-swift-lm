---
assignees:
- claude-code
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