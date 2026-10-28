import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

// MARK: - ArraysCache.advance moves offset (fork)

/// In this fork, `ArraysCache.advance(_:)` also moves `offset`. Upstream moves only the batch
/// bookkeeping. The prompt cache of this fork compares the offset of each cache with the length
/// of its token ledger, thus a recurrent cache must count its tokens as an attention cache does.
@Test func arraysCacheAdvanceMovesOffsetAndSequenceMetadata() throws {
    let slotCount = 2
    let startOffset = 7
    let leftPadding = [3, 5]
    let lengths = [4, 6]
    let fedTokenCount = 2
    let cache = ArraysCache(size: slotCount, leftPadding: leftPadding)
    cache.offset = startOffset
    cache.prepare(lengths: lengths)

    cache.advance(fedTokenCount)

    #expect(cache.offset == startOffset + fedTokenCount)
    #expect(cache.leftPaddingValues == leftPadding.map { $0 - fedTokenCount })
    #expect(cache.lengthsValues == lengths.map { $0 - fedTokenCount })
}

/// A hybrid layer of a prefill and a few decode steps, without a model.
private enum RecurrentAdvanceFixture {
    /// The number of tokens that the prefill feeds.
    static let prefillTokenCount = 5

    /// The number of single-token decode steps that follow the prefill.
    static let decodeStepCount = 3

    /// Feeds `tokenCount` tokens through a hybrid layer the way a model layer does: the
    /// recurrent cache takes new conv and recurrent states and then one `advance`, and the
    /// attention cache takes one `update(keys:values:)`.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens that the layer feeds.
    ///   - recurrent: The recurrent cache of the layer.
    ///   - attention: The attention cache of the layer.
    static func feed(tokenCount: Int, recurrent: MambaCache, attention: KVCacheSimple) {
        recurrent[0] = MLXArray.zeros([1, 1, 1])
        recurrent[1] = MLXArray.zeros([1, 1, 1])
        recurrent.advance(tokenCount)
        let block = MLXArray.zeros([1, 1, tokenCount, 1])
        _ = attention.update(keys: block, values: block)
    }
}

@Test func mambaCacheAdvanceMovesOffsetByTheTokenCount() throws {
    let cache = MambaCache()

    cache.advance(RecurrentAdvanceFixture.prefillTokenCount)

    #expect(cache.offset == RecurrentAdvanceFixture.prefillTokenCount)
}

/// A layer that calls `advance` and then also moves the offset by hand counts each token two
/// times. This test fails for such a layer.
@Test func layerStyleAdvanceCountsEachTokenOnceOverAPrefillAndDecodeSteps() throws {
    let recurrent = MambaCache()
    let attention = KVCacheSimple()
    let layer = CacheList(recurrent, attention)

    RecurrentAdvanceFixture.feed(
        tokenCount: RecurrentAdvanceFixture.prefillTokenCount, recurrent: recurrent,
        attention: attention)
    for _ in 0 ..< RecurrentAdvanceFixture.decodeStepCount {
        RecurrentAdvanceFixture.feed(tokenCount: 1, recurrent: recurrent, attention: attention)
    }

    let total = RecurrentAdvanceFixture.prefillTokenCount + RecurrentAdvanceFixture.decodeStepCount
    #expect(recurrent.offset == total)
    #expect(attention.offset == total)
    #expect(layer.offset == total)
}

// MARK: - trim() + continuation semantics
//
// `PromptCache` (MLXFoundationModels) assumes that trimming a KVCacheSimple
// back to an earlier offset and then continuing generation from there
// produces IDENTICAL cache state to having only ever prefilled the
// truncated sequence followed by the same continuation. That assumption is
// what makes `.trimTo` a safe substitute for a full rebuild on a diverged
// prompt. These tests exercise `trim()` itself (not `PromptCache`'s pure
// decision logic, which `PromptCacheTests` already covers) against a real
// `KVCacheSimple`, proving the assumption holds at the tensor level.

/// One single-position (seqLen == 1) key/value pair filled with a distinct
/// scalar so each fed "token" is individually identifiable in the resulting
/// state tensor.
/// The offset that sets the values of a fed token apart from its keys.
private let valueFillOffset: Float = 1000

private func singleTokenKV(fill: Float, kvHeads: Int = 2, headDim: Int = 4) -> (
    keys: MLXArray, values: MLXArray
) {
    (
        MLXArray.ones([1, kvHeads, 1, headDim], dtype: .float32) * fill,
        MLXArray.ones([1, kvHeads, 1, headDim], dtype: .float32) * (fill + valueFillOffset)
    )
}

@Test func testTrimThenContinueMatchesFreshPrefillOfTruncatedSequence() throws {
    // Prefill 7 tokens (fills 0...6) into cacheA.
    let cacheA = KVCacheSimple()
    for i in 0 ..< 7 {
        let (k, v) = singleTokenKV(fill: Float(i))
        _ = cacheA.update(keys: k, values: v)
    }
    #expect(cacheA.offset == 7)

    // Trim back by 3 (offset 7 -> 4) -- KVCacheSimple.trim is unbounded, so
    // the full request is honored.
    let trimmed = cacheA.trim(3)
    #expect(trimmed == 3)
    #expect(cacheA.offset == 4)

    // Continue with 2 NEW tokens (fills 100, 101) from the trimmed offset.
    for fill: Float in [100, 101] {
        let (k, v) = singleTokenKV(fill: fill)
        _ = cacheA.update(keys: k, values: v)
    }
    #expect(cacheA.offset == 6)

    // Independent cache, fed ONLY the truncated prefix (fills 0...3) plus
    // the identical continuation (fills 100, 101) -- what `PromptCache`
    // assumes trim+continue on cacheA is equivalent to.
    let cacheB = KVCacheSimple()
    for i in 0 ..< 4 {
        let (k, v) = singleTokenKV(fill: Float(i))
        _ = cacheB.update(keys: k, values: v)
    }
    for fill: Float in [100, 101] {
        let (k, v) = singleTokenKV(fill: fill)
        _ = cacheB.update(keys: k, values: v)
    }
    #expect(cacheB.offset == 6)

    #expect(cacheA.offset == cacheB.offset)
    eval(cacheA.state)
    eval(cacheB.state)
    assertArraysClose(cacheA.state, cacheB.state)
}

// MARK: - MambaCache speculative snapshot slots
//
// A verify pass of the gated delta net kernel keeps the recurrent state after
// each of its tokens, and the conv input window, as transient speculative data
// of the cache. A rollback selects one slot; nothing else reads the slots.

/// The slots of one verify, without a model: each slot and each window row
/// holds its own numbers, so a test sees which one the cache selected.
private enum SnapshotSlotFixture {
    /// The tokens of the verify.
    static let tokenCount = 4

    /// The fewest tokens of a verify.
    static let fewestTokens = 2

    /// The rows of the conv state: the conv taps less one.
    static let stateRows = 3

    /// The sequences of the batch.
    static let batch = 2

    /// The heads, and the rows and width of each head's state.
    static let heads = 2
    static let headRows = 2
    static let headWidth = 2

    /// The shape of one recurrent state: `(batch, heads, rows, width)`.
    static let stateShape = [batch, heads, headRows, headWidth]

    /// The conv channels.
    static let channels = 2

    /// The position of the cache before the verify.
    static let startOffset = 7

    /// The left padding of each sequence before the verify: different for
    /// each sequence, so a test sees that the cache keeps them apart.
    static let firstLeftPadding = 9
    static let secondLeftPadding = 11
    static let leftPadding = [firstLeftPadding, secondLeftPadding]

    /// The salts that give the recurrent slots and the conv window
    /// different numbers.
    static let recurrentSalt = 1
    static let windowSalt = 2

    /// The snapshot slots of a verify of `tokens` tokens.
    ///
    /// - Parameter tokens: The tokens of the verify.
    /// - Returns: The slots.
    static func snapshots(tokens: Int = tokenCount) -> GatedDeltaDecodeKernel.Snapshots {
        let recurrent = deterministicTensor(shape: [tokens] + stateShape, salt: recurrentSalt)
            .asType(.float32)
        let window = deterministicTensor(
            shape: [batch, stateRows + tokens, channels], salt: windowSalt)
        eval(recurrent, window)
        return GatedDeltaDecodeKernel.Snapshots(recurrent: recurrent, convWindow: window)
    }

    /// A cache after a verify, the way a layer leaves it: the states after the
    /// last token, the slots saved, then one `advance`.
    ///
    /// - Parameter snapshots: The slots of the verify.
    /// - Returns: The cache.
    static func cache(after snapshots: GatedDeltaDecodeKernel.Snapshots) -> MambaCache {
        let cache = MambaCache(leftPadding: leftPadding)
        cache.offset = startOffset
        cache[0] = snapshots.convState(afterToken: snapshots.tokenCount)
        cache[1] = snapshots.recurrentState(afterToken: snapshots.tokenCount)
        cache.saveSpeculativeSnapshots(snapshots)
        cache.advance(snapshots.tokenCount)
        return cache
    }

    /// A cache that holds the same states and position, with no slots.
    ///
    /// - Parameter cache: The cache to match.
    /// - Returns: The cache without slots.
    static func plainCache(like cache: MambaCache) -> MambaCache {
        let plain = MambaCache()
        plain.state = cache.state
        plain.offset = cache.offset
        return plain
    }

    /// Tells if a cache holds the states after `accepted` tokens of the
    /// verify, at the position of those tokens.
    static func holdsTheSlot(
        _ cache: MambaCache, of snapshots: GatedDeltaDecodeKernel.Snapshots, accepted: Int
    ) -> Bool {
        guard let conv = cache[0], let recurrent = cache[1] else { return false }
        return cache.offset == startOffset + accepted
            && cache.leftPaddingValues == leftPadding.map { $0 - accepted }
            && arrayEqual(conv, snapshots.convWindow[0..., accepted ..< accepted + stateRows])
                .item(Bool.self)
            && arrayEqual(recurrent, snapshots.recurrent[accepted - 1]).item(Bool.self)
    }
}

@Test func mambaCacheSelectsTheSnapshotSlotOfEachRollback() throws {
    for rollback in 1 ..< SnapshotSlotFixture.tokenCount {
        let snapshots = SnapshotSlotFixture.snapshots()
        let cache = SnapshotSlotFixture.cache(after: snapshots)
        #expect(cache.speculativeSlotCount == SnapshotSlotFixture.tokenCount)

        #expect(cache.selectSpeculativeSlot(rollingBack: rollback))

        let accepted = SnapshotSlotFixture.tokenCount - rollback
        #expect(SnapshotSlotFixture.holdsTheSlot(cache, of: snapshots, accepted: accepted))
        #expect(cache.speculativeSlotCount == 0)
    }
}

@Test func mambaCacheRefusesARollbackItHoldsNoSlotFor() throws {
    let snapshots = SnapshotSlotFixture.snapshots()
    let cache = SnapshotSlotFixture.cache(after: snapshots)
    let offset = cache.offset

    #expect(!cache.selectSpeculativeSlot(rollingBack: 0))
    #expect(!cache.selectSpeculativeSlot(rollingBack: SnapshotSlotFixture.tokenCount))
    #expect(cache.offset == offset)
    #expect(cache.speculativeSlotCount == SnapshotSlotFixture.tokenCount)
    #expect(!MambaCache().selectSpeculativeSlot(rollingBack: 1))
}

/// An attention cache that holds the tokens of a verify, and a recurrent cache that holds its
/// snapshot slots, the way a verify leaves the two kinds of a hybrid cache.
private func hybridCacheAfterAVerify(of snapshots: GatedDeltaDecodeKernel.Snapshots) -> (
    attention: KVCacheSimple, recurrent: MambaCache
) {
    let attention = KVCacheSimple()
    let keys = MLXArray.zeros([1, 1, snapshots.tokenCount, SnapshotSlotFixture.channels])
    _ = attention.update(keys: keys, values: keys)
    return (attention, SnapshotSlotFixture.cache(after: snapshots))
}

@Test func rewindingAHybridCacheByEachDraftCountSelectsItsSlotAndTrimsTheAttention() throws {
    for rollback in 1 ..< SnapshotSlotFixture.tokenCount {
        let snapshots = SnapshotSlotFixture.snapshots()
        let (attention, recurrent) = hybridCacheAfterAVerify(of: snapshots)

        let rewound = rewindSpeculativePromptCache([attention, recurrent], numTokens: rollback)

        #expect(rewound == rollback)

        let accepted = SnapshotSlotFixture.tokenCount - rollback
        #expect(attention.offset == accepted)
        #expect(SnapshotSlotFixture.holdsTheSlot(recurrent, of: snapshots, accepted: accepted))
        #expect(recurrent.speculativeSlotCount == 0)
    }
}

@Test func aHybridRewindItHoldsNoSlotForLeavesTheCacheAsItWas() throws {
    for rollback in [0, SnapshotSlotFixture.tokenCount] {
        let snapshots = SnapshotSlotFixture.snapshots()
        let (attention, recurrent) = hybridCacheAfterAVerify(of: snapshots)

        #expect(rewindSpeculativePromptCache([attention, recurrent], numTokens: rollback) == 0)

        #expect(attention.offset == SnapshotSlotFixture.tokenCount)
        #expect(recurrent.speculativeSlotCount == SnapshotSlotFixture.tokenCount)
        #expect(
            SnapshotSlotFixture.holdsTheSlot(
                recurrent, of: snapshots, accepted: SnapshotSlotFixture.tokenCount))
    }
}

@Test func discardingTheSpeculativeCheckpointsReleasesTheSlots() throws {
    let snapshots = SnapshotSlotFixture.snapshots()
    let cache = SnapshotSlotFixture.cache(after: snapshots)
    let offset = cache.offset

    discardSpeculativePromptCacheCheckpoints([cache])

    #expect(cache.speculativeSlotCount == 0)
    #expect(!cache.selectSpeculativeSlot(rollingBack: 1))
    #expect(cache.offset == offset)
    #expect(
        SnapshotSlotFixture.holdsTheSlot(
            cache, of: snapshots, accepted: SnapshotSlotFixture.tokenCount))
}

@Test func theSlotsOfANewerVerifyReplaceTheSlotsOfAnEarlierVerify() throws {
    let cache = SnapshotSlotFixture.cache(after: SnapshotSlotFixture.snapshots())
    let newer = SnapshotSlotFixture.snapshots(tokens: SnapshotSlotFixture.fewestTokens)

    cache.saveSpeculativeSnapshots(newer)

    #expect(cache.speculativeSlotCount == SnapshotSlotFixture.fewestTokens)
    #expect(cache.selectSpeculativeSlot(rollingBack: 1))
    let accepted = SnapshotSlotFixture.fewestTokens - 1
    #expect(
        arrayEqual(try #require(cache[1]), newer.recurrent[accepted - 1]).item(Bool.self))
}

@Test func theSnapshotSlotsAreLeftOutOfACopy() throws {
    let cache = SnapshotSlotFixture.cache(after: SnapshotSlotFixture.snapshots())

    let copied = try #require(cache.copy() as? MambaCache)

    #expect(copied.speculativeSlotCount == 0)
    #expect(copied.offset == cache.offset)
    assertArraysClose(copied.state, cache.state)
}

@Test func theSnapshotSlotsAreLeftOutOfTheByteCount() throws {
    let few = SnapshotSlotFixture.cache(
        after: SnapshotSlotFixture.snapshots(tokens: SnapshotSlotFixture.fewestTokens))
    let many = SnapshotSlotFixture.cache(
        after: SnapshotSlotFixture.snapshots(tokens: GatedDeltaDecodeKernel.mostTokens))

    #expect(few.residentByteCount == SnapshotSlotFixture.plainCache(like: few).residentByteCount)
    #expect(few.residentByteCount == many.residentByteCount)
}

@Test func theSnapshotSlotsAreLeftOutOfAPromptCacheCheckpoint() throws {
    let cache = SnapshotSlotFixture.cache(after: SnapshotSlotFixture.snapshots())
    let tokens = Array(0 ..< cache.offset)

    let withSlots = try #require(PromptCacheCheckpoint(caches: [cache], tokens: tokens, state: nil))
    let withoutSlots = try #require(
        PromptCacheCheckpoint(
            caches: [SnapshotSlotFixture.plainCache(like: cache)], tokens: tokens, state: nil))

    #expect(withSlots.byteCount == withoutSlots.byteCount)
}

@Test func theSnapshotSlotsAreLeftOutOfASavedPromptCache() throws {
    let cache = SnapshotSlotFixture.cache(after: SnapshotSlotFixture.snapshots())
    let url = TemporaryPromptCacheFile.makeURL()

    try savePromptCache(url: url, cache: [cache], metadata: [:])
    let (loaded, _) = try loadPromptCache(url: url)

    let restored = try #require(loaded.first as? MambaCache)
    #expect(restored.speculativeSlotCount == 0)
    #expect(restored.state.count == cache.state.count)
    assertArraysClose(restored.state, cache.state)
}
