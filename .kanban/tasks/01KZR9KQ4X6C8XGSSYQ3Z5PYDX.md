---
position_column: todo
position_ordinal: '9380'
title: 'Question: can a prefix-matched prompt-cache chunk chain leave a session acting as though a tool call already happened?'
---
Filed from FoundationModelsRouter (^pw807cp AC#5) as a **question with evidence**, not an asserted defect. Not investigable there — this is PromptCache internals.

## The mechanism, as documented in this repo

`PromptCache` stores each completed round's KV state as content-addressed chunks **shared across every conversation on that model** (SGLang RadixAttention-style). `MLXLanguageModel.removePromptCache(modelID:)` (MLXLanguageModel.swift:843) exists precisely because an entry can go bad — its own doc says it is used when "a round's actual generated content can't be reconciled with `cache`'s own `offset` (see `Executor.commitPromptCache`) — the entry is untrustworthy, so the next round rebuilds instead of risking a stale reuse."

So this repo already models the idea of a cache entry that must not be reused. The question is whether that guard is sufficient.

## The observation that prompts the question

A consumer running `mlx-community/Qwen3.6-27B-mxfp4` with two tools mounted, on a fixed prompt with sampling pinned to `.greedy` (temperature 0, argmax), saw a model **answer with identifiers for tools it never called**:

- expected marker prefix `MARKER-7F3A-`; the model answered `MARKER-9B2C-ONE` / `MARKER-9B2C-TWO`
- in another run it dispatched **zero** tools and answered `12345` / `67890`

The consumer verified its own side is clean: one tool output per announced call, every output resolving to its call, completed-id set equal to called-id set, and each surface's answer equal to its own transcript's final response entry.

## The question

Can a prefix-matched chunk chain place a fresh conversation into KV state carried over from a *different* conversation in which a tool call had already occurred — such that the model continues as though it had results it never received?

**This is explicitly not asserted.** The same symptom is fully explained by the model simply being unreliable at tool dispatch, and the consumer's evidence cannot separate the two. Filing so that someone with visibility into chunk resolution can rule it in or out.

## Suggested first step
Determine whether a session that has never dispatched a tool can resolve chunks contributed by a session that did, and if so whether `commitPromptCache`'s offset reconciliation would catch the mismatch.

## Acceptance criteria
- [ ] Answered yes or no, with the resolution path that makes it so
- [ ] If yes: a reproduction, and whether the offset reconciliation is expected to catch it
- [ ] If no: recorded here so the consumer stops suspecting it