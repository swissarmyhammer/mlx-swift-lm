---
position_column: todo
position_ordinal: '9280'
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