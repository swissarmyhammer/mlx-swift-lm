// Copyright © 2026 Apple Inc.
//
// Shared fixtures for the `PromptCache` actor-level test suites
// (`PromptCacheResolveTests`, `PromptCacheSlotPoolTests`,
// `PromptCacheMultiSessionTests`, `PromptCacheConcurrencyTests`): a minimal
// `LanguageModel` probe, cache constructors positioned as if real tokens had
// been fed, and a `resolve()` convenience that hides the single-use
// `SendableBox` plumbing. Internal (not `private`) so every suite shares one
// definition instead of re-declaring per-file copies.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Minimal `LanguageModel` stand-in whose only job is to hand
/// `PromptCache.resolve`/`applyDecision` a fresh, distinguishable
/// `[KVCache]` on every `newCache(parameters:)` call (via
/// `KVCacheDimensionProvider`'s default implementation) -- `prepare`/
/// `callAsFunction` are never invoked by `PromptCache` and simply trap if
/// they somehow were.
///
/// `@unchecked Sendable` so concurrent test tasks can capture it directly:
/// it holds no mutable state (`kvHeads` is computed, `newCache` allocates
/// fresh instances), unlike a real model, which is why production code
/// threads models through `SendableBox` instead.
final class PromptCacheProbeModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    /// Number of KV-attention heads per layer, one entry per layer.
    ///
    /// A single layer with one head is the minimal shape `KVCacheDimensionProvider`
    /// needs to build a real `[KVCache]` via `newCache(parameters:)` -- this probe
    /// never runs actual attention, so the specific count is otherwise arbitrary.
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        fatalError("not exercised by PromptCache tests")
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised by PromptCache tests")
    }
}

/// A trimmable cache positioned as if `tokenCount` tokens had been fed through it.
///
/// This is the state `Executor.commitPromptCache` stores alongside a slot's
/// tokens, where `trimAndVerify`'s offset arithmetic holds.
func makeSlotCache(tokenCount: Int) -> KVCacheSimple {
    let cache = KVCacheSimple()
    cache.offset = tokenCount
    return cache
}

/// A NON-trimmable cache positioned at `tokenCount`.
///
/// `RotatingKVCache` reports `isTrimmable == false` once `offset >=
/// maxCacheSize` (its window has rotated, so earlier positions are gone and
/// a prefix trim is meaningless). Used to drive `decide`/`applyDecision`
/// down their rebuild-instead-of-trim fallbacks.
func makeNonTrimmableSlotCache(tokenCount: Int) -> RotatingKVCache {
    let cache = RotatingKVCache(maxSize: tokenCount)
    cache.offset = tokenCount
    return cache
}

/// One `resolve()` round against a cache.
///
/// Hides the single-use `SendableBox` plumbing (`consume()` traps on a
/// second call, so each resolve needs a fresh box around the model and must
/// consume the result exactly once).
func resolveOnce(
    cache: PromptCache, modelID: String, newTokens: [Int],
    model: any MLXLMCommon.LanguageModel
) async -> (cache: [KVCache], tokensToFeed: [Int]) {
    await cache.resolve(
        modelID: modelID, newTokens: newTokens, model: SendableBox(model), parameters: nil
    ).consume()
}

/// Stores well-formed slots of tokens backed by a trimmable cache.
///
/// The cache's offset matches `tokens.count` -- the well-formed slot shape
/// every reuse path assumes -- and the stored INSTANCE (not just its
/// identity) is returned so a later `resolve()` can prove (or disprove)
/// reuse of this exact instance.
///
/// Callers must keep the returned instance alive until their last identity
/// assertion: once `PromptCache` evicts a slot, the actor drops its only
/// other reference, and a deallocated cache's address can be recycled for a
/// later `newCache()` allocation -- an `ObjectIdentifier` collision that
/// makes a genuine rebuild look like a (forbidden) reuse. Comparing against
/// `ObjectIdentifier(<live instance>)` at assertion time rules that out,
/// because the use itself keeps the instance alive to that point.
@discardableResult
func storeWellFormedSlot(
    cache: PromptCache, modelID: String, tokens: [Int]
) async -> KVCacheSimple {
    let slotCache = makeSlotCache(tokenCount: tokens.count)
    await cache.store(modelID: modelID, tokens: tokens, cache: SendableBox([slotCache]))
    return slotCache
}

/// Identity of the single layer cache in a resolved result.
///
/// The probe model is single-layer (`kvHeads == [1]`), so every result
/// carries exactly one cache. `KVCache` isn't statically class-constrained,
/// but every conformer in this codebase is a class.
func cacheIdentity(resolved: (cache: [KVCache], tokensToFeed: [Int])) -> ObjectIdentifier {
    ObjectIdentifier(resolved.cache[0] as AnyObject)
}

#endif
