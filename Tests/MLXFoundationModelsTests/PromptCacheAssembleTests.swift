// Copyright © 2026 Apple Inc.
//
// Tests for `PromptCache.assemble` (kanban mwwk2bs): reconstructing a fresh,
// PRIVATE `[KVCacheSimple]` stack from previously stored chunks (see
// `PromptCacheChunkTests` for `sliceChunks`, `PromptCacheChunkStoreTests` for
// the chunk store's `insert`/`lookupLongestPrefix`). `assemble` is the final,
// still-unwired "Assembly" step those two tasks deferred: nothing in
// `MLXLanguageModel.swift`'s `Executor` calls it yet (a later wiring task
// does), so these tests exercise it directly, independent of `resolve()`.
//
// Like `PromptCacheChunkTests`, these fixtures populate real tensor CONTENT
// (not just placeholder arrays) so byte-exact equality and post-assembly
// `update()` behavior can be verified against actual values -- this forces a
// real GPU-device eval under plain `swift test`; see `TestBootstrap.swift`'s
// `MetalLibraryTestBootstrap` (kanban 23ff1zx) for why the `init()` bootstrap
// call below is required.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("PromptCache chunk assembly")
struct PromptCacheAssembleTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixtures

    /// Mirrors `PromptCacheChunkTests.makeCache`: a fully-packed `KVCacheSimple`
    /// layer covering exactly `tokenCount` positions with distinct,
    /// position-derived content (not zeros), so assembled content can be
    /// checked byte-for-byte against the source.
    private func makeCache(tokenCount: Int, headDim: Int = 4, valueOffset: Int = 0)
        -> KVCacheSimple
    {
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

    // MARK: - Content correctness

    @Test(
        "assembled keys/values are element-equal to the source cache's state sliced to the matched token count"
    )
    func assembledContentMatchesSourceSlice() {
        let chunkSize = 4
        let chunkCount = 3
        let tokenCount = chunkSize * chunkCount
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let matchedChunkCount = 2
        let matched = Array(chunks.prefix(matchedChunkCount))
        let matchedTokenCount = matchedChunkCount * chunkSize

        let assembled = PromptCache.assemble(chunks: matched, layerCount: 1)

        #expect(assembled.count == 1)
        let expectedKeys = cache.state[0][.ellipsis, ..<matchedTokenCount, 0...]
        let expectedValues = cache.state[1][.ellipsis, ..<matchedTokenCount, 0...]
        #expect((assembled[0].state[0] .== expectedKeys).all().item())
        #expect((assembled[0].state[1] .== expectedValues).all().item())
    }

    @Test("assembly preserves per-layer content independently across a multi-layer stack")
    func multiLayerAssemblyPreservesPerLayerContent() {
        let chunkSize = 4
        let chunkCount = 2
        let tokenCount = chunkSize * chunkCount
        let cache0 = makeCache(tokenCount: tokenCount, valueOffset: 0)
        let cache1 = makeCache(tokenCount: tokenCount, valueOffset: 500)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [cache0, cache1], chunkSize: chunkSize)!

        let assembled = PromptCache.assemble(chunks: chunks, layerCount: 2)

        #expect(assembled.count == 2)
        #expect((assembled[0].state[0] .== cache0.state[0]).all().item())
        #expect((assembled[0].state[1] .== cache0.state[1]).all().item())
        #expect((assembled[1].state[0] .== cache1.state[0]).all().item())
        #expect((assembled[1].state[1] .== cache1.state[1]).all().item())
    }

    @Test("assembling zero chunks yields an empty cache stack")
    func assemblingZeroChunksYieldsEmptyStack() {
        let assembled = PromptCache.assemble(chunks: [], layerCount: 3)

        #expect(assembled.isEmpty)
    }

    /// Mirrors `sliceChunks`'s own defensive convention (returns `nil` rather
    /// than guess at a shape it can't safely reason about -- see
    /// `PromptCacheChunkTests`'s "Non-chunkable degradation" section):
    /// `assemble` must not trust a caller-supplied `layerCount` that doesn't
    /// match every chunk's actual `layers.count` -- indexing
    /// `layers[layerIndex]` unguarded would otherwise trap on an
    /// out-of-range access instead of failing predictably.
    @Test(
        "a StoredChunk whose layers.count doesn't match layerCount yields an empty stack rather than crashing"
    )
    func mismatchedLayerCountYieldsEmptyStack() {
        let chunkSize = 4
        let cache = makeCache(tokenCount: chunkSize)
        let tokens = Array(0 ..< chunkSize)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        #expect(chunks[0].layers.count == 1)

        let assembled = PromptCache.assemble(chunks: chunks, layerCount: 2)

        #expect(assembled.isEmpty)
    }

    // MARK: - Offset

    @Test("assembled cache's offset equals the matched token count")
    func assembledOffsetEqualsMatchedTokenCount() {
        let chunkSize = 4
        let chunkCount = 3
        let tokenCount = chunkSize * chunkCount
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let matched = Array(chunks.prefix(2))
        let assembled = PromptCache.assemble(chunks: matched, layerCount: 1)

        #expect(assembled[0].offset == 2 * chunkSize)
    }

    // MARK: - Post-assembly update()

    @Test(
        "feeding one new token through update() after assembly advances offset and preserves the assembled prefix"
    )
    func updateAfterAssemblyAdvancesOffsetAndPreservesPrefix() {
        let chunkSize = 4
        let headDim = 4
        let chunkCount = 3
        let tokenCount = chunkSize * chunkCount
        let cache = makeCache(tokenCount: tokenCount, headDim: headDim)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        let matched = Array(chunks.prefix(2))
        let matchedTokenCount = 2 * chunkSize

        let assembled = PromptCache.assemble(chunks: matched, layerCount: 1)
        let assembledCache = assembled[0] as! KVCacheSimple
        let prefixKeysBefore = assembledCache.state[0]

        let newKeys = MLXArray(Int32(90_000) ..< Int32(90_000 + headDim), [1, 1, 1, headDim])
        let newValues = MLXArray(Int32(91_000) ..< Int32(91_000 + headDim), [1, 1, 1, headDim])
        let (returnedKeys, _) = assembledCache.update(keys: newKeys, values: newValues)

        #expect(assembledCache.offset == matchedTokenCount + 1)
        #expect(returnedKeys.dim(2) == matchedTokenCount + 1)
        #expect(
            (returnedKeys[.ellipsis, ..<matchedTokenCount, 0...] .== prefixKeysBefore).all().item())
        #expect(
            (returnedKeys[.ellipsis, matchedTokenCount ..< (matchedTokenCount + 1), 0...]
                .== newKeys
            ).all().item())
    }

    // MARK: - Distinct instances

    @Test(
        "two assemble() calls from the same stored chunks return distinct cache instances with equal content"
    )
    func assembleTwiceReturnsDistinctInstancesWithEqualContent() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let first = PromptCache.assemble(chunks: chunks, layerCount: 1)
        let second = PromptCache.assemble(chunks: chunks, layerCount: 1)

        let firstSimple = first[0] as! KVCacheSimple
        let secondSimple = second[0] as! KVCacheSimple

        #expect(ObjectIdentifier(firstSimple) != ObjectIdentifier(secondSimple))
        #expect((firstSimple.state[0] .== secondSimple.state[0]).all().item())
        #expect((firstSimple.state[1] .== secondSimple.state[1]).all().item())
    }

    /// The acceptance criterion literally says "two CONCURRENT resolves" --
    /// this exercises genuine `Task`-group concurrency (mirroring
    /// `PromptCacheConcurrencyTests`'s pattern of boxing non-`Sendable`
    /// results via `SendableBox`), rather than merely calling `assemble`
    /// twice sequentially on the same thread (see the test above). Each
    /// `Task` gets its own `SendableBox` wrapping a separate COPY of the
    /// `chunks` array value -- since `StoredChunk` is a struct, copying the
    /// array does not copy its `MLXArray` tensor fields, so both tasks are
    /// still genuinely backed by the same underlying stored tensors.
    @Test(
        "two concurrently running assemble() calls for the same stored chunks return distinct cache instances with equal content"
    )
    func concurrentAssembleCallsReturnDistinctInstancesWithEqualContent() async {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let firstBox = SendableBox(chunks)
        let secondBox = SendableBox(chunks)

        let resultBoxes = await withTaskGroup(of: SendableBox<[KVCache]>.self) { group in
            group.addTask {
                SendableBox(PromptCache.assemble(chunks: firstBox.consume(), layerCount: 1))
            }
            group.addTask {
                SendableBox(PromptCache.assemble(chunks: secondBox.consume(), layerCount: 1))
            }
            var results: [SendableBox<[KVCache]>] = []
            for await box in group {
                results.append(box)
            }
            return results
        }

        #expect(resultBoxes.count == 2)
        let first = resultBoxes[0].consume()
        let second = resultBoxes[1].consume()
        let firstSimple = first[0] as! KVCacheSimple
        let secondSimple = second[0] as! KVCacheSimple

        #expect(ObjectIdentifier(firstSimple) != ObjectIdentifier(secondSimple))
        #expect((firstSimple.state[0] .== secondSimple.state[0]).all().item())
        #expect((firstSimple.state[1] .== secondSimple.state[1]).all().item())
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
    /// prior version of this test compared `ObjectIdentifier`s and passed
    /// even with `ownedCopy(of:)` temporarily removed from `assemble`'s
    /// single-chunk path). Comparing raw buffer addresses closes that gap:
    /// two arrays with the SAME address are backed by the identical C++
    /// buffer no matter how many distinct Swift wrapper objects point at it.
    private func rawBufferAddress(of array: MLXArray) -> UInt? {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress) }
    }

    /// Writes directly into `array`'s underlying buffer, in place, at its
    /// first raw byte -- bypasses MLX's functional update path entirely.
    /// Unlike `liveKeys[...] = zeros` (the prior version of this test's
    /// approach), which builds a NEW array via a functional/scatter-style
    /// update and never touches the original buffer in place, this reaches
    /// through ``MLXArray/asData(access:)``'s ``MLXArray/AccessMethod/noCopy``
    /// mode -- which wraps the actual backing pointer with no copy -- and
    /// writes through it directly, so any OTHER array that genuinely shares
    /// this same underlying buffer would observe the change.
    private func mutateFirstElementInPlace(of array: MLXArray) {
        array.asData(access: .noCopy).data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            UnsafeMutableRawPointer(mutating: raw.baseAddress!)
                .storeBytes(of: Int32(-1), as: Int32.self)
        }
    }

    /// Regression test for a real MLX behavior discovered during
    /// implementation: `mlx::core::concatenate` special-cases a SINGLE input
    /// array by returning that exact array unchanged (`if (arrays.size() == 1)
    /// { return arrays[0]; }` in `mlx/ops.cpp`) -- it never allocates a fresh
    /// buffer or invokes the `Concatenate` primitive at all in that case.
    /// A naive `assemble()` that concatenated a single matched chunk's
    /// tensors without an explicit copy step would hand back the chunk
    /// store's own owned tensor's buffer BY REFERENCE, defeating the "every
    /// resolve gets its own assembled copy" guarantee this task's
    /// description requires -- even though its Swift wrapper object would
    /// still be a distinct instance (see ``rawBufferAddress(of:)``'s doc
    /// comment for why `ObjectIdentifier` alone cannot catch this).
    @Test(
        "assembling a single matched chunk allocates a genuinely independent underlying buffer, not the stored chunk's own buffer under a fresh Swift wrapper"
    )
    func singleChunkAssemblyAllocatesIndependentBuffer() {
        let chunkSize = 4
        let cache = makeCache(tokenCount: chunkSize)
        let tokens = Array(0 ..< chunkSize)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        #expect(chunks.count == 1)

        let assembled = PromptCache.assemble(chunks: chunks, layerCount: 1)
        let assembledCache = assembled[0] as! KVCacheSimple

        #expect(
            rawBufferAddress(of: assembledCache.state[0]) != rawBufferAddress(of: chunks[0].layers[0].keys),
            "assemble() must allocate a fresh buffer, not merely a new Swift wrapper around the stored chunk's own C++ buffer"
        )
        #expect(
            rawBufferAddress(of: assembledCache.state[1])
                != rawBufferAddress(of: chunks[0].layers[0].values),
            "assemble() must allocate a fresh buffer, not merely a new Swift wrapper around the stored chunk's own C++ buffer"
        )
        #expect((assembledCache.state[0] .== chunks[0].layers[0].keys).all().item())
        #expect((assembledCache.state[1] .== chunks[0].layers[0].values).all().item())
    }

    /// End-to-end version of the same hazard, but with a mutation that
    /// genuinely reaches the underlying C++ buffer in place (see
    /// ``mutateFirstElementInPlace(of:)``'s doc comment for why a plain
    /// `liveKeys[...] = zeros` assignment -- the prior version of this
    /// test's approach -- does NOT: it builds a new array via a functional
    /// update and never touches the original buffer, so it passed even
    /// against a naive `assemble()` with `ownedCopy(of:)` removed from the
    /// single-chunk path). Mirrors
    /// `PromptCacheChunkTests.chunkTensorsAreOwnedCopies`'s ownership proof,
    /// one level up the pipeline.
    @Test(
        "in-place mutation of a stored chunk's underlying buffer after assembly leaves the already-assembled cache unchanged"
    )
    func mutatingSourceChunkBufferInPlaceAfterAssemblyLeavesAssembledCacheUnchanged() {
        let chunkSize = 4
        let cache = makeCache(tokenCount: chunkSize)
        let tokens = Array(0 ..< chunkSize)
        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let assembled = PromptCache.assemble(chunks: chunks, layerCount: 1)
        let assembledKeysBefore = assembled[0].state[0].asArray(Int32.self)
        let assembledValuesBefore = assembled[0].state[1].asArray(Int32.self)

        mutateFirstElementInPlace(of: chunks[0].layers[0].keys)
        mutateFirstElementInPlace(of: chunks[0].layers[0].values)

        let assembledKeysAfter = assembled[0].state[0].asArray(Int32.self)
        let assembledValuesAfter = assembled[0].state[1].asArray(Int32.self)

        #expect(assembledKeysAfter == assembledKeysBefore)
        #expect(assembledValuesAfter == assembledValuesBefore)
    }
}

#endif
