// Copyright © 2026 Apple Inc.
//
// Shared fixtures for the `PromptCache` actor-level test suites
// (`PromptCacheChunkCutoverTests`, `PromptCacheResolveTests`,
// `PromptCacheEvictionScopeTests`, `PromptCacheConcurrencyTests`): a minimal
// `LanguageModel` probe, cache constructors mirroring the two shapes
// `store()`/`sliceChunks` actually reason about, and a `resolve()`
// convenience that hides the single-use `SendableBox` plumbing. Internal
// (not `private`) so every suite shares one definition instead of
// re-declaring per-file copies.
//
// `makeUnchunkableCache` only positions a cache's `offset` (no real tensor
// content) since `sliceChunks` rejects a `RotatingKVCache` layer on its type
// check alone, before ever inspecting `state` -- so this fixture needs no
// GPU bootstrap of its own. `makeChunkableCache` populates real `MLXArray`
// content, because `store()` routes through `sliceChunks`, which requires a
// genuine `KVCacheSimple.state` (not just `offset`) to slice -- callers of
// that fixture force a real GPU-device eval under plain `swift test`; see
// `TestBootstrap.swift`'s `MetalLibraryTestBootstrap` (kanban 23ff1zx,
// memory note `swiftpm-test-gpu-metallib-limit`) for why those suites'
// `init()` bootstrap call is required.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Minimal `LanguageModel` stand-in whose only job is to hand
/// `PromptCache.resolve` a fresh, distinguishable
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

    /// Protocol conformance stub; not invoked by PromptCache and will trap if called.
    func prepare(_ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?) throws -> PrepareResult {
        fatalError("not exercised by PromptCache tests")
    }

    /// Protocol conformance stub; not invoked by PromptCache and will trap if called.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("not exercised by PromptCache tests")
    }
}

/// A NON-chunkable cache positioned at `tokenCount`.
///
/// `RotatingKVCache` reports `isTrimmable == false` once `offset >=
/// maxCacheSize` (its window has rotated, so earlier positions are gone),
/// and -- more to the point for the chunk store -- `sliceChunks` rejects it
/// on its dynamic-type check alone, before ever inspecting `state`: only a
/// verified `KVCacheSimple` layer can be sliced, so `store()` silently drops
/// a round backed by this cache.
///
/// - Parameter tokenCount: How many sequence positions to position the cache's `offset` at.
/// - Returns: A `RotatingKVCache` whose `offset` is set to `tokenCount` and whose
///   `state` is left empty -- `sliceChunks` rejects it by dynamic type alone, so
///   `store()` silently drops any round backed by this cache.
func makeUnchunkableCache(tokenCount: Int) -> RotatingKVCache {
    let cache = RotatingKVCache(maxSize: tokenCount)
    cache.offset = tokenCount
    return cache
}

/// A fully-packed, chunkable `KVCacheSimple` layer covering exactly
/// `tokenCount` sequence positions, with distinct, position-derived key/value
/// content (not zeros) -- mirroring `PromptCacheChunkTests.makeCache`.
///
/// Setting `state` directly (rather than feeding through
/// `update(keys:values:)`) leaves `offset == keys.dim(2) == tokenCount`
/// exactly, the "verified" shape `sliceChunks` (and therefore `store()`)
/// requires. Real content (rather than placeholder zeros) lets a caller
/// verify slice/assembly correctness by tensor value, not just shape, should
/// a test need that; most `store()`-only callers only care that the shape is
/// genuine.
///
/// - Parameters:
///   - tokenCount: How many sequence positions this layer covers.
///   - headDim: Per-token feature width for both keys and values.
///   - valueOffset: Shifts every element's value upward -- lets a multi-layer
///     test build layers with non-overlapping content ranges to verify
///     per-layer independence.
/// - Returns: A `KVCacheSimple` with `state` set to distinct key/value
///   `MLXArray`s of shape `[1, 1, tokenCount, headDim]`, so `offset` reports
///   exactly `tokenCount` and `sliceChunks`/`store()` see a genuine, sliceable layer.
func makeChunkableCache(tokenCount: Int, headDim: Int = 4, valueOffset: Int = 0) -> KVCacheSimple {
    let cache = KVCacheSimple()
    let count = tokenCount * headDim
    let keys = MLXArray(
        Int32(valueOffset) ..< Int32(valueOffset + count), [1, 1, tokenCount, headDim])
    let values = MLXArray(
        Int32(valueOffset + 10_000) ..< Int32(valueOffset + 10_000 + count),
        [1, 1, tokenCount, headDim])
    cache.state = [keys, values]
    return cache
}

/// A `PromptCache.HybridCheckpoint` with placeholder (never content-inspected)
/// tensors -- for tests exercising the hybrid checkpoint store's
/// longest-prefix-match/eviction bookkeeping, which never reads `layers`'
/// tensor content, only `tokens`/`byteSize`/`lastUsed`. Shared by
/// `PromptCacheHybridArchitectureTests` (matching/LRU semantics) and
/// `PromptCacheHybridEvictionScopeTests` (kanban `vgznm16`'s eviction-scope
/// and cross-conversation isolation coverage).
///
/// - Parameters:
///   - tokens: The full token prefix the checkpoint claims to cover.
///   - byteSize: The byte footprint to account for this checkpoint --
///     explicit (not derived from the placeholder tensors) so byte-budget
///     tests can drive eviction pressure precisely.
/// - Returns: A two-layer (`.mamba`, `.simple`) checkpoint with placeholder
///   tensor state.
func makeHybridCheckpoint(
    tokens: [Int], byteSize: Int = 64
) -> PromptCache.HybridCheckpoint {
    let dummy = MLXArray([Int32(0)])
    return PromptCache.HybridCheckpoint(
        tokens: tokens,
        layers: [
            (kind: .mamba, state: [dummy, dummy]),
            (kind: .simple, state: [dummy, dummy]),
        ],
        byteSize: byteSize, lastUsed: 0)
}

/// One `resolve()` round against a cache.
///
/// Hides the single-use `SendableBox` plumbing (`consume()` traps on a
/// second call, so each resolve needs a fresh box around the model and must
/// consume the result exactly once).
///
/// - Parameters:
///   - cache: The `PromptCache` actor to resolve against.
///   - modelID: The model identity the round is stored/resolved under.
///   - newTokens: The full prompt token sequence for this round.
///   - model: The language model whose `newCache(parameters:)` builds a fresh `[KVCache]`
///     when no stored prefix matches.
/// - Returns: `cache`, the `[KVCache]` to generate this round with (either the
///   matched-and-assembled prefix cache, or a freshly-built one when nothing
///   matched), paired with `tokensToFeed`, the subsequence of `newTokens` still
///   needing to be fed through the model -- a suffix when a prefix matched, or
///   all of `newTokens` otherwise.
func resolveOnce(
    cache: PromptCache, modelID: String, newTokens: [Int],
    model: any MLXLMCommon.LanguageModel
) async -> (cache: [KVCache], tokensToFeed: [Int]) {
    await cache.resolve(
        modelID: modelID, newTokens: newTokens, model: SendableBox(model), parameters: nil
    ).consume()
}

/// Identity of the single layer cache in a resolved result.
///
/// The probe model is single-layer (`kvHeads == [1]`), so every result
/// carries exactly one cache. `KVCache` isn't statically class-constrained,
/// but every conformer in this codebase is a class.
///
/// - Parameter resolved: A `resolve()` round's result, as returned by `resolveOnce`.
/// - Returns: The `ObjectIdentifier` of `resolved.cache`'s single layer instance,
///   for comparing whether two rounds returned the same or distinct cache instances.
func cacheIdentity(resolved: (cache: [KVCache], tokensToFeed: [Int])) -> ObjectIdentifier {
    ObjectIdentifier(resolved.cache[0] as AnyObject)
}

/// Reaches PAST the Swift `MLXArray` wrapper to the actual underlying
/// C++ buffer address, via the public ``MLXArray/asData(access:)`` API's
/// ``MLXArray/AccessMethod/noCopy`` mode -- which wraps
/// `mlx_array_data_uint8(ctx)`'s pointer directly, with no copy (see
/// `MLXArray+Bytes.swift` in the vendored `mlx-swift` package).
///
/// This is NOT the same thing `ObjectIdentifier` comparison checks:
/// `MLXArray.concatenated(...)` always constructs a brand-new Swift
/// wrapper object for its result, regardless of whether the underlying
/// MLX C++ buffer is shared with an input -- so comparing
/// `ObjectIdentifier`s can never distinguish "genuinely fresh buffer"
/// from "same C++ buffer, new Swift wrapper" (confirmed empirically: a
/// prior version of `PromptCacheAssembleTests` compared `ObjectIdentifier`s
/// and passed even with `ownedCopy(of:)` temporarily removed from
/// `assemble`'s single-chunk path). Comparing raw buffer addresses closes
/// that gap: two arrays with the SAME address are backed by the identical
/// C++ buffer no matter how many distinct Swift wrapper objects point at it.
///
/// Shared by `PromptCacheAssembleTests` (single-chunk `assemble()`
/// buffer-ownership proof) and `PromptCacheMultiSessionTests` (concurrent-
/// session buffer isolation proof) -- both need the same past-the-wrapper
/// check, so this lives here rather than as a per-file private duplicate.
///
/// - Parameter array: The array whose underlying buffer address to read.
/// - Returns: The raw base address of `array`'s backing C++ buffer, or
///   `nil` if the buffer is empty.
func rawBufferAddress(of array: MLXArray) -> UInt? {
    array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress) }
}

/// Writes directly into `array`'s underlying buffer, in place, at its
/// first raw byte -- bypasses MLX's functional update path (and
/// `KVCacheSimple.update()`'s functional replace) entirely, reaching
/// through ``MLXArray/asData(access:)``'s ``MLXArray/AccessMethod/noCopy``
/// mode -- which wraps the actual backing pointer with no copy -- and
/// writing through it directly, so any OTHER array that genuinely shares
/// this same underlying buffer would observe the change.
///
/// - Parameter array: The array to mutate in place.
func mutateFirstElementInPlace(of array: MLXArray) {
    array.asData(access: .noCopy).data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        UnsafeMutableRawPointer(mutating: raw.baseAddress!)
            .storeBytes(of: Int32(-1), as: Int32.self)
    }
}

#endif
