// Copyright © 2026 Apple Inc.
//
// Tests for `PromptCache.sliceChunks` (kanban wx4w28j): cutting a verified
// `KVCacheSimple` stack into full token-range chunks with owned, evaluated
// tensor copies and a hash-chained `chunkKey`.
//
// These tests populate real `MLXArray` content (not just `offset`
// bookkeeping, unlike `PromptCacheTestSupport`'s fixtures) so slice
// correctness and ownership can be verified by tensor *content*, which
// forces a real GPU-device eval under plain `swift test` -- see
// `TestBootstrap.swift`'s `MetalLibraryTestBootstrap` (kanban 23ff1zx,
// memory note `swiftpm-test-gpu-metallib-limit`) for why the `init()`
// bootstrap call below is required.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

// .serialized: `chunkCopiesReleaseSourceBufferMemory` measures MLX's PROCESS-GLOBAL
// `Memory.activeMemory` around a large (200,000-token) allocation -- Swift Testing
// parallelizes tests within a suite by default, so a concurrently-running sibling
// test IN THIS SUITE could otherwise land inside that measurement window and mask
// a real regression or flake the result.
//
// NOTE this only orders this suite's OWN tests against each other; per
// `ModelCacheEvictionTests.swift`'s own doc comment, `.serialized` does NOT order
// two top-level suites against each other, so it does not by itself rule out
// interference from a *different* suite's tests running concurrently in the same
// process. Unlike `ModelCacheEvictionTests`'s two suites (which share one exact
// mutable structure and so are nested under one serialized parent), every other
// suite in this test target only builds trivial `MLXArray`s (single-element or
// small token-count arrays -- confirmed by inspection), many orders of magnitude
// below this test's ~6MB safety margin, so the residual cross-suite race is
// negligible in practice even though it isn't structurally eliminated.
@Suite("PromptCache chunk slicing", .serialized)
struct PromptCacheChunkTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixtures

    /// A fully-packed `KVCacheSimple` layer covering exactly `tokenCount` sequence
    /// positions, whose key/value tensors hold distinct, position-derived values
    /// (rather than zeros) so slices can be checked for correctness by content.
    ///
    /// Setting `state` directly (rather than feeding through `update(keys:values:)`)
    /// leaves `offset == keys.dim(2) == tokenCount` exactly -- the "verified" shape
    /// `sliceChunks` requires (see its doc comment).
    private func makeCache(tokenCount: Int, headDim: Int = 4, valueOffset: Int = 0)
        -> KVCacheSimple
    {
        let cache = KVCacheSimple()
        let count = tokenCount * headDim
        let keys = MLXArray(Int32(valueOffset) ..< Int32(valueOffset + count), [1, 1, tokenCount, headDim])
        let values = MLXArray(
            Int32(valueOffset + 10_000) ..< Int32(valueOffset + 10_000 + count),
            [1, 1, tokenCount, headDim])
        cache.state = [keys, values]
        return cache
    }

    // MARK: - Boundary math

    @Test("slices a cache of 5×chunkSize+7 tokens into exactly 5 full chunks, tail dropped")
    func boundaryMath() {
        let chunkSize = 8
        let tokenCount = 5 * chunkSize + 7
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)

        #expect(chunks?.count == 5)
        #expect(chunks?.allSatisfy { $0.tokens.count == chunkSize } == true)
        #expect(chunks?.last?.tokens == Array(tokens[(4 * chunkSize) ..< (5 * chunkSize)]))
    }

    @Test("a cache with fewer tokens than one chunk yields no chunks")
    func shorterThanOneChunkYieldsEmpty() {
        let chunkSize = 8
        let tokenCount = 5
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)

        #expect(chunks?.isEmpty == true)
    }

    // MARK: - Hash-chain keying

    @Test("chunk keys form an identical chain for identical prefix tokens across separate slicings")
    func keyChainIdenticalForSamePrefix() {
        let chunkSize = 4
        let tokenCount = chunkSize * 3
        let tokens = Array(0 ..< tokenCount)

        let cacheA = makeCache(tokenCount: tokenCount)
        let cacheB = makeCache(tokenCount: tokenCount)

        let chunksA = PromptCache.sliceChunks(tokens: tokens, cache: [cacheA], chunkSize: chunkSize)
        let chunksB = PromptCache.sliceChunks(tokens: tokens, cache: [cacheB], chunkSize: chunkSize)

        #expect(chunksA?.map(\.chunkKey) == chunksB?.map(\.chunkKey))
        #expect(chunksA?.map(\.parentKey) == chunksB?.map(\.parentKey))
    }

    @Test("chunk keys diverge from the point of token divergence onward")
    func keyChainDivergesAtDivergencePoint() {
        let chunkSize = 4
        let tokenCount = chunkSize * 3
        let tokensA = Array(0 ..< tokenCount)
        var tokensB = tokensA
        tokensB[chunkSize + 1] = 9_999  // diverge inside chunk index 1

        let cacheA = makeCache(tokenCount: tokenCount)
        let cacheB = makeCache(tokenCount: tokenCount)

        let chunksA = PromptCache.sliceChunks(tokens: tokensA, cache: [cacheA], chunkSize: chunkSize)!
        let chunksB = PromptCache.sliceChunks(tokens: tokensB, cache: [cacheB], chunkSize: chunkSize)!

        #expect(chunksA[0].chunkKey == chunksB[0].chunkKey)
        #expect(chunksA[1].chunkKey != chunksB[1].chunkKey)
        #expect(chunksA[2].chunkKey != chunksB[2].chunkKey)
    }

    @Test("the first chunk's parent key is the fixed root seed")
    func firstChunkParentsFromRoot() {
        let chunkSize = 4
        let cache = makeCache(tokenCount: chunkSize)
        let tokens = Array(0 ..< chunkSize)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        #expect(chunks[0].parentKey == PromptCache.rootChunkKey)
    }

    // MARK: - Non-chunkable degradation

    @Test("a stack containing a RotatingKVCache layer is not chunkable")
    func rotatingLayerReturnsNil() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let simple = makeCache(tokenCount: tokenCount)
        let rotating = RotatingKVCache(maxSize: tokenCount * 2)
        rotating.offset = tokenCount
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [simple, rotating], chunkSize: chunkSize)

        #expect(chunks == nil)
    }

    @Test("a KVCacheSimple layer whose offset doesn't match the token count is not chunkable")
    func offsetMismatchReturnsNil() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< (tokenCount - 1))  // mismatched on purpose

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)

        #expect(chunks == nil)
    }

    /// `ChunkedKVCache` is a SUBCLASS of `KVCacheSimple` (see `Libraries/MLXLMCommon/KVCache.swift`)
    /// that overrides `update`/`trim`/`copy`/`metaState` but NOT `state`/`isTrimmable` -- once its
    /// `maybeTrimFront()` has physically trimmed `keys`/`values` down to its own `chunkSize` while
    /// `offset` keeps tracking the full logical token position, the inherited `state` getter's
    /// `offset == keys.dim(2)` branch no longer holds, and `sliceChunks` would slice against the
    /// wrong physical extent. A plain `layer as? KVCacheSimple` check does NOT exclude this
    /// subclass, so this constructs the "safe-looking" case (offset == keys.dim(2), matching
    /// `tokens.count`) to prove the type check itself is the bug -- not the corrupted-shape case,
    /// which risks tripping an out-of-bounds MLX slice/crash rather than failing a test cleanly.
    @Test("a ChunkedKVCache layer (a KVCacheSimple subclass) is not chunkable, even with a safe-looking offset/state shape")
    func chunkedKVCacheLayerReturnsNil() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = ChunkedKVCache(chunkSize: chunkSize)
        let keys = MLXArray(0 ..< Int32(tokenCount * 4), [1, 1, tokenCount, 4])
        let values = MLXArray(0 ..< Int32(tokenCount * 4), [1, 1, tokenCount, 4])
        cache.state = [keys, values]
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)

        #expect(chunks == nil)
    }

    @Test("an empty cache stack returns nil rather than silently producing zero-layer chunks")
    func emptyCacheArrayReturnsNil() {
        let chunkSize = 4
        let tokens = Array(0 ..< (chunkSize * 2))

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [], chunkSize: chunkSize)

        #expect(chunks == nil)
    }

    // MARK: - Content correctness

    @Test("chunk tensors are element-equal to the corresponding slices of the source cache")
    func chunkTensorsMatchSourceSlices() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let sourceKeys = cache.state[0]
        let sourceValues = cache.state[1]
        for (index, chunk) in chunks.enumerated() {
            let a = index * chunkSize
            let b = a + chunkSize
            let expectedKeys = sourceKeys[.ellipsis, a ..< b, 0...]
            let expectedValues = sourceValues[.ellipsis, a ..< b, 0...]
            #expect((chunk.layers[0].keys .== expectedKeys).all().item())
            #expect((chunk.layers[0].values .== expectedValues).all().item())
        }
    }

    @Test("multi-layer stacks slice each layer independently, preserving per-layer content")
    func multiLayerSlicingPreservesPerLayerContent() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache0 = makeCache(tokenCount: tokenCount, valueOffset: 0)
        let cache1 = makeCache(tokenCount: tokenCount, valueOffset: 500)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(
            tokens: tokens, cache: [cache0, cache1], chunkSize: chunkSize)!

        #expect(chunks.count == 2)
        for (index, chunk) in chunks.enumerated() {
            #expect(chunk.layers.count == 2)
            let a = index * chunkSize
            let b = a + chunkSize
            #expect((chunk.layers[0].keys .== cache0.state[0][.ellipsis, a ..< b, 0...]).all().item())
            #expect((chunk.layers[1].keys .== cache1.state[0][.ellipsis, a ..< b, 0...]).all().item())
        }
    }

    // MARK: - Ownership

    @Test("chunk tensors are owned copies: mutating the source cache afterward leaves them unchanged")
    func chunkTensorsAreOwnedCopies() {
        let chunkSize = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        let firstChunkKeysBefore = chunks[0].layers[0].keys.asArray(Int32.self)
        let firstChunkValuesBefore = chunks[0].layers[0].values.asArray(Int32.self)

        // Mutate the SAME backing array objects `sliceChunks` sliced from, IN
        // PLACE via index assignment -- the same in-place update pattern
        // `KVCacheSimple.update(keys:values:)` itself uses in production. This
        // is the real hazard the ownership requirement guards against: merely
        // reassigning `cache.state` to fresh objects would leave the original
        // buffers completely untouched (and would pass even a naive
        // slice-without-copy implementation, proving nothing).
        let liveKeys = cache.state[0]
        let liveValues = cache.state[1]
        liveKeys[.ellipsis, 0 ..< tokenCount, 0...] = MLXArray.zeros(
            [1, 1, tokenCount, 4], dtype: .int32)
        liveValues[.ellipsis, 0 ..< tokenCount, 0...] = MLXArray.zeros(
            [1, 1, tokenCount, 4], dtype: .int32)

        let firstChunkKeysAfter = chunks[0].layers[0].keys.asArray(Int32.self)
        let firstChunkValuesAfter = chunks[0].layers[0].values.asArray(Int32.self)

        #expect(firstChunkKeysAfter == firstChunkKeysBefore)
        #expect(firstChunkValuesAfter == firstChunkValuesBefore)
    }

    /// The real ownership hazard: MLX's `Slice` primitive (what
    /// `array[.ellipsis, a..<b, 0...]` lazily builds) always evaluates via
    /// `copy_shared_buffer` (see `mlx/backend/common/slicing.cpp`) -- a
    /// zero-copy view that shares the SOURCE array's underlying allocator
    /// buffer, unconditionally, regardless of contiguity. Evaluating that
    /// slice (even calling `.eval()` on it) does NOT copy the data; it only
    /// detaches the *compute graph* (see `array::detach()` in `array.cpp`),
    /// which is orthogonal to the *buffer* the resulting array's data still
    /// points into. So a naive "slice + eval, no explicit copy" chunk keeps
    /// the ENTIRE source cache's buffer allocated for as long as the chunk
    /// lives, even after the source `KVCacheSimple` itself is unreachable --
    /// invisible to both content-equality checks (the slice's logical content
    /// is correct) and to `nbytes` (which reports the slice's own *logical*
    /// size, not the shared buffer's real size). The only way to observe this
    /// is real memory accounting: `MLX.Memory.activeMemory`.
    @Test("owned chunk copies release the source cache's buffers once the source is dropped")
    func chunkCopiesReleaseSourceBufferMemory() {
        let chunkSize = 8
        let headDim = 8
        // Deliberately large relative to one chunk's footprint, so a shared
        // (un-copied) buffer is unmistakable against the noise floor of
        // whatever else is resident on the device.
        let bigTokenCount = 200_000
        let bigSourceFootprint = 2 * bigTokenCount * headDim * MemoryLayout<Int32>.size

        func sliceOneChunkFromLargeCache() -> PromptCache.StoredChunk {
            let cache = makeCache(tokenCount: bigTokenCount, headDim: headDim)
            let tokens = Array(0 ..< bigTokenCount)
            let chunks = PromptCache.sliceChunks(
                tokens: tokens, cache: [cache], chunkSize: chunkSize)!
            return chunks[0]
            // `cache`, and the large keys/values arrays it alone retained,
            // have no other strong references once this function returns.
        }

        Memory.clearCache()
        let before = Memory.snapshot().activeMemory

        let chunk = sliceOneChunkFromLargeCache()

        let after = Memory.snapshot().activeMemory
        let grew = after - before

        // An owned copy grows active memory by roughly one small chunk's
        // footprint. An aliased slice would grow it by roughly the ENTIRE
        // large source's footprint (many times a chunk's own byteSize) --
        // this threshold sits far below that, with a wide margin either way.
        #expect(grew < bigSourceFootprint / 2)
        #expect(chunk.byteSize < bigSourceFootprint / 100)
    }

    @Test("byteSize reflects the owned copies' real footprint")
    func byteSizeMatchesOwnedFootprint() {
        let chunkSize = 4
        let headDim = 4
        let tokenCount = chunkSize * 2
        let cache = makeCache(tokenCount: tokenCount, headDim: headDim)
        let tokens = Array(0 ..< tokenCount)

        let chunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!

        let expectedPerTensor: Int = chunkSize * headDim * MemoryLayout<Int32>.size
        let expectedTotal: Int = expectedPerTensor * 2
        for chunk in chunks {
            var actualTotal = 0
            for layer in chunk.layers {
                actualTotal += layer.keys.nbytes
                actualTotal += layer.values.nbytes
            }
            #expect(chunk.byteSize == actualTotal)
            #expect(chunk.byteSize == expectedTotal)
        }
    }

    // MARK: - Configurable chunk size (kanban bbda7xg)

    /// Store/resolve round-trips at a configured chunk size using MORE than
    /// `2 * chunkSize` tokens, then asserts ACTUAL reuse on a second resolve
    /// of an extension of that prompt: `tokensToFeed` must be strictly
    /// smaller than the full extended prompt. A > 2×chunkSize token count
    /// guarantees at least two full chunks are sliced and stored (never a
    /// vacuous zero-chunks-stored pass), and the extension shares every
    /// stored chunk's exact token content, so a correct implementation must
    /// reuse them.
    @Test(
        "store/resolve round-trips with actual reuse at a configured chunk size",
        arguments: [1, 64, 256]
    )
    func configurableChunkSizeRoundTripsWithReuse(chunkSize: Int) async {
        let cache = PromptCache()
        await cache.setChunkSize(chunkSize)
        let modelID = "chunk-size-roundtrip-\(chunkSize)-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let tokenCount = 2 * chunkSize + 7
        let tokens = Array(0 ..< tokenCount)

        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeCache(tokenCount: tokenCount)]))

        let storedChunkCount = await cache.chunkCount(modelID: modelID)
        #expect(
            storedChunkCount > 0,
            "chunkSize \(chunkSize): expected at least one chunk actually stored, ruling out a vacuous pass"
        )

        let extensionTokens = tokens + Array(90_000 ..< 90_005)
        let resolved = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: extensionTokens, model: model)

        #expect(
            resolved.tokensToFeed.count < extensionTokens.count,
            "chunkSize \(chunkSize): expected actual reuse -- tokensToFeed must be strictly smaller than the full prompt"
        )
    }

    @Test("setChunkSize to a new value evicts all stored chunks")
    func setChunkSizeToNewValueEvictsStore() async {
        let cache = PromptCache()
        let modelID = "chunk-size-change-evicts-\(UUID().uuidString)"
        let chunkSize = 8
        let tokens = Array(0 ..< (chunkSize * 3))

        await cache.setChunkSize(chunkSize)
        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeCache(tokenCount: tokens.count)]))
        #expect(await cache.chunkCount(modelID: modelID) > 0)

        await cache.setChunkSize(chunkSize * 2)

        #expect(
            await cache.chunkCount(modelID: modelID) == 0,
            "changing chunkSize to a genuinely different value must evict all stored chunks")
    }

    @Test("setChunkSize to the SAME (already-clamped) value does not evict stored chunks")
    func setChunkSizeToSameValueDoesNotEvict() async {
        let cache = PromptCache()
        let modelID = "chunk-size-same-no-evict-\(UUID().uuidString)"
        let chunkSize = 8
        let tokens = Array(0 ..< (chunkSize * 3))

        await cache.setChunkSize(chunkSize)
        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeCache(tokenCount: tokens.count)]))
        let countBefore = await cache.chunkCount(modelID: modelID)
        #expect(countBefore > 0)

        await cache.setChunkSize(chunkSize)

        #expect(
            await cache.chunkCount(modelID: modelID) == countBefore,
            "setting the same chunkSize again must not evict stored chunks")
    }

    @Test("setChunkSize clamps non-positive requests up to 1", arguments: [0, -3])
    func setChunkSizeClampsNonPositiveToOne(requested: Int) async {
        let cache = PromptCache()
        let modelID = "chunk-size-clamp-\(requested)-\(UUID().uuidString)"

        await cache.setChunkSize(requested)

        // A clamped chunkSize of 1 chunks every single token -- store 3
        // tokens and expect exactly 3 stored chunks, proving the effective
        // chunkSize is 1, not `requested` (0 or negative, which would slice
        // zero or crash on division by zero).
        let tokens = Array(0 ..< 3)
        await cache.store(
            modelID: modelID, tokens: tokens,
            cache: SendableBox([makeCache(tokenCount: tokens.count)]))

        #expect(
            await cache.chunkCount(modelID: modelID) == 3,
            "requested chunkSize \(requested) must clamp to 1, chunking every token individually")
    }

    // MARK: - sliceTailChunk (kanban 5ra1wzm): the trailing partial span

    @Test(
        "sliceTailChunk captures the trailing partial span sliceChunks leaves uncovered"
    )
    func sliceTailChunkCapturesTrailingPartialSpan() {
        let chunkSize = 8
        let tokenCount = 2 * chunkSize + 5
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let fullChunks = PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)!
        #expect(fullChunks.count == 2, "sliceChunks itself must still only slice full chunks")

        let parentKey = fullChunks.last!.chunkKey
        let tail = PromptCache.sliceTailChunk(
            tokens: tokens, cache: [cache], chunkSize: chunkSize, parentKey: parentKey)!

        #expect(tail.tokens == Array(tokens[(2 * chunkSize)...]))
        #expect(tail.tokens.count == 5)
        #expect(tail.parentKey == parentKey)
        #expect(tail.chunkKey == PromptCache.chunkKey(parentKey: parentKey, tokens: tail.tokens))
        #expect(tail.byteSize > 0)

        let sourceKeys = cache.state[0]
        let sourceValues = cache.state[1]
        let expectedKeys = sourceKeys[.ellipsis, (2 * chunkSize)..., 0...]
        let expectedValues = sourceValues[.ellipsis, (2 * chunkSize)..., 0...]
        #expect((tail.layers[0].keys .== expectedKeys).all().item())
        #expect((tail.layers[0].values .== expectedValues).all().item())
    }

    @Test("sliceTailChunk returns nil when tokens.count is an exact multiple of chunkSize")
    func sliceTailChunkReturnsNilForExactMultiple() {
        let chunkSize = 8
        let tokenCount = chunkSize * 3
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        let tail = PromptCache.sliceTailChunk(
            tokens: tokens, cache: [cache], chunkSize: chunkSize,
            parentKey: PromptCache.rootChunkKey)

        #expect(tail == nil, "an exact multiple of chunkSize has no partial span to store")
    }

    @Test("sliceTailChunk chains from the caller-supplied parentKey even when there are no full chunks")
    func sliceTailChunkChainsFromSuppliedParentKeyWithNoFullChunks() {
        let chunkSize = 64
        let tokenCount = 40
        let cache = makeCache(tokenCount: tokenCount)
        let tokens = Array(0 ..< tokenCount)

        #expect(
            PromptCache.sliceChunks(tokens: tokens, cache: [cache], chunkSize: chunkSize)?.isEmpty
                == true, "fewer tokens than one chunk yields zero full chunks")

        let tail = PromptCache.sliceTailChunk(
            tokens: tokens, cache: [cache], chunkSize: chunkSize,
            parentKey: PromptCache.rootChunkKey)!

        #expect(tail.tokens == tokens)
        #expect(tail.parentKey == PromptCache.rootChunkKey)
    }

    @Test("sliceTailChunk mirrors sliceChunks's non-chunkable degradation")
    func sliceTailChunkReturnsNilForUnchunkableCache() {
        let chunkSize = 8
        let tokenCount = chunkSize + 3
        let rotating = RotatingKVCache(maxSize: tokenCount * 2)
        rotating.offset = tokenCount
        let tokens = Array(0 ..< tokenCount)

        let tail = PromptCache.sliceTailChunk(
            tokens: tokens, cache: [rotating], chunkSize: chunkSize,
            parentKey: PromptCache.rootChunkKey)

        #expect(tail == nil)
    }
}

#endif
