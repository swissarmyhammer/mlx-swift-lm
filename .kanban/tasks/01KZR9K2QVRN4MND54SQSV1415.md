---
comments:
- actor: claude-code
  id: 01kzyw1nk6y3cjcj8ayggaw6ch
  text: |-
    ### Closed outright — the merge took away the thing this card asked to change

    The user closed this card on 2026-08-14.

    **The API this card names is gone.** `MLXLanguageModel.removePromptCache(modelID:)` does not exist anywhere in the tree now. The catch-up to `ml-explore/mlx-swift-lm` replaced `Libraries/MLXFoundationModels/MLXLanguageModel.swift` whole, because the two sides had no common ancestor. The line numbers this card gives, `:843` and `:857`, name a file that is no longer here.

    **The pain is gone too, and not because somebody fixed it.** Upstream's `ModelCache` holds containers, loading tasks, suppressed load ids, grammar tokenizers, constraint templates, tokenizer biases and last errors. It holds NO prompt cache. Each `respond()` builds a cache of its own and drops it, thus each test case already starts clean and no model reload is needed for that.

    `evictAll()` is still there, at `:472`. It no longer has a prompt cache to clear.

    **The knowledge this card holds, so that it is not lost:**

    The constraint comes back the moment somebody does card `^2ajc82t` and wires a cross-turn prompt cache into the upstream `MLXLanguageModel`. Our `Libraries/MLXFoundationModels/PromptCache.swift` already holds `remove(modelID:)` at line 677 and `evictAll()` at line 650, and **each member of that file is internal**. A port that does not make `remove(modelID:)` public builds the same wall this card was filed against. Whoever takes `^2ajc82t` should read this paragraph first.

    The card came from a consumer of this fork, `FoundationModelsRouter` (`^pw807cp` AC#5). That consumer gets clean caches for nothing now, and gets no reuse at all.
  timestamp: 2026-08-14T00:51:41.286825+00:00
position_column: done
position_ordinal: f980
title: Make prompt-cache clearing available without evicting model weights
---
Filed from FoundationModelsRouter (^pw807cp AC#5). Not fixable there — this is fork API.

`MLXLanguageModel.removePromptCache(modelID:)` (Libraries/MLXFoundationModels/MLXLanguageModel.swift:843) is **not public**. The only public lever that clears a prompt cache is `evictAll()` (:857), which also drops every cached model, tokenizer and constraint template, freeing the GPU memory held by model weights.

Consequence for consumers: a test suite that wants each case to start from a clean KV/prompt cache must pay a full model reload — for a 27B model that is the difference between seconds and minutes per case, which makes per-suite cache isolation impractical.

## What is wanted

A public way to clear a model's prompt cache while leaving its weights resident. `removePromptCache(modelID:)` already does exactly this internally; the ask is to expose it (or an equivalent `clearPromptCache(modelID:)`) as public API.

## Acceptance criteria
- [ ] A public API clears one model's prompt cache without evicting its weights
- [ ] Documented as such, including that it does not affect the container/tokenizer caches
- [ ] Existing `evictAll()` semantics unchanged