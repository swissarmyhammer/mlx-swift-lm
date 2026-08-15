---
comments:
- actor: claude-code
  id: 01kzyw4vq5r5q0g2x7vgsqjzsj
  text: |-
    ### Answered: NO. A prefix-matched chunk chain cannot carry a tool call into a session that never made one.

    The reason is the shape of the key. `PromptCacheChunks.swift:91`:

    ```swift
    internal nonisolated static func chunkKey(parentKey: Int, tokens: [Int]) -> Int {
        var hasher = Hasher()
        hasher.combine(parentKey)
        for token in tokens {
            hasher.combine(token)
        }
        return hasher.finalize()
    }
    ```

    The key takes the PARENT key and this chunk's tokens. The parent key came from the same function, and so did its parent, back to `rootChunkKey`. Each key thus holds the whole token prefix from position zero. This is a chain of hashes, not a hash of one chunk.

    **What that gives:** for a fresh conversation to resolve a chunk that a tool-calling conversation made, the fresh conversation must present the same tokens from position zero. That means the tool call, its arguments and its results stand in its OWN prompt. To read the KV state at that point is correct, and it is not a leak. No path lets a session take state for tokens it never rendered.

    The chunks are shared across the conversations of one model, and that sharing is on purpose — it is the RadixAttention design. It shares on identical content, which is the condition that makes sharing safe.

    The offset reconciliation of `commitPromptCache` and `removePromptCache` are a second guard against an entry that went bad. They are not what stops this.

    **The strength of this answer, stated plainly:** it is a reading of the key derivation, not a reproduction. Short of a hash collision there is no resolution path, thus the argument is structural. Nobody ran an experiment.

    **Two facts make the suspicion weaker still:**

    1. The consumer ran `mlx-community/Qwen3.6-27B-mxfp4`. On 2026-08-13 this repository measured that exact model getting ZERO cache reuse — round 2 fed each of its 4748 tokens and skipped 0, and its prefill took 11.59 s against a cold control of 11.60 s. See card `^2ajc82t`.
    2. After the catch-up to `ml-explore/mlx-swift-lm`, the upstream `MLXLanguageModel` holds no cross-turn cache at all, thus no chunk chain runs today.

    That leaves the reading the card itself gives: greedy sampling does not make a model good at tool dispatch. `MARKER-9B2C-ONE` and `12345`/`67890` read like a model that made up believable identifiers.

    **The consumer can stop suspecting the prompt cache.**

    **One thing to carry to `^2ajc82t`:** the chained key is a correctness property, not an implementation detail. A port that hashes only a chunk's own tokens, to save work or to share more, builds the very defect this card asked about.
  timestamp: 2026-08-14T00:53:25.861420+00:00
position_column: done
position_ordinal: fa80
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