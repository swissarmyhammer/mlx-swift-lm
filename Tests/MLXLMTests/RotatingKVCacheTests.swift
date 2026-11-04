import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

extension KVCacheTests {
    // MARK: - RotatingKVCache.logicalView
    //
    // `logicalView(tail:)` is the read-only counterpart to `update(keys:values:)`: it exposes the
    // ring's contents in chronological order without writing. Speculative decoding needs it to
    // present committed history alongside K/V it has not committed yet.
    //
    // Nothing in this repo drove a `RotatingKVCache` ring past its wrap before these tests, so the
    // rotation in `updateInPlace`, the linearization in `temporalOrder`, and the windowed mask that
    // `makeMask` builds over the multi-token presentation were all uncovered. They are the
    // foundation the view rests on, so they are pinned here too.

    /// K/V whose every element encodes its own sequence position: keys hold `+p`, values `-p`.
    /// Any reordering, duplication, or K/V mix-up changes the numbers.
    private func positionedKV(
        _ positions: Range<Int>, headDim: Int = 2
    ) -> (MLXArray, MLXArray) {
        let count = positions.count
        let keys = MLXArray(
            positions.flatMap { Array(repeating: Float($0), count: headDim) },
            [1, 1, count, headDim])
        let values = MLXArray(
            positions.flatMap { Array(repeating: Float(-$0), count: headDim) },
            [1, 1, count, headDim])
        return (keys, values)
    }

    /// Recover the sequence positions encoded by `positionedKV` from a `[B, H, S, D]` key array.
    private func encodedPositions(_ keys: MLXArray) -> [Int] {
        keys[0, 0, 0..., 0].asArray(Float.self).map { Int($0) }
    }

    /// Drive `cache` through `count` single-token writes, so the ring actually rotates.
    private func fillOneAtATime(_ cache: RotatingKVCache, positions: Range<Int>) {
        for p in positions {
            let (k, v) = positionedKV(p ..< (p + 1))
            _ = cache.update(keys: k, values: v)
        }
    }

    @Test func testRotatingLogicalViewIsNilBeforeFirstWrite() {
        #expect(RotatingKVCache(maxSize: 8, keep: 0).logicalView(tail: 8) == nil)
    }

    @Test func testRotatingLogicalViewReturnsChronologicalTailPastWrap() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        // The ring has wrapped twice; physical order is rotated, chronological order is not.
        #expect(cache.offset == 20)

        let full = try #require(cache.logicalView(tail: 8))
        #expect(
            encodedPositions(full.0) == Array(12 ..< 20),
            "logicalView did not linearize the rotated ring")
        #expect(
            encodedPositions(full.1).map { -$0 } == Array(12 ..< 20), "values diverged from keys")

        let short = try #require(cache.logicalView(tail: 3))
        #expect(encodedPositions(short.0) == [17, 18, 19], "a short tail took the wrong end")
    }

    @Test func testRotatingKVCacheReportsEvictedTokensThroughKVCache() {
        let rotating = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(rotating, positions: 0 ..< 20)

        let cache: any KVCache = rotating
        #expect(cache.evictedTokenCount == 12)
    }

    @Test func testRotatingLogicalViewClampsTailToWhatTheCacheHolds() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 5)

        #expect(encodedPositions(try #require(cache.logicalView(tail: 99)).0) == Array(0 ..< 5))
        #expect(encodedPositions(try #require(cache.logicalView(tail: 5)).0) == Array(0 ..< 5))
        #expect(try #require(cache.logicalView(tail: 0)).0.dim(2) == 0)
        #expect(try #require(cache.logicalView(tail: -1)).0.dim(2) == 0, "negative tail must clamp")
    }

    @Test func testRotatingLogicalViewLeavesTheRingUntouched() {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        let beforeState = cache.innerState().map { MLXArray($0.asArray(Float.self), $0.shape) }
        let beforeOffset = cache.offset
        let beforeMeta = cache.metaState

        _ = cache.logicalView(tail: 7)
        _ = cache.logicalView(tail: 3)

        let afterState = cache.innerState()
        #expect(afterState.count == beforeState.count)
        for (index, (before, after)) in zip(beforeState, afterState).enumerated() {
            #expect(before.shape == after.shape, "innerState[\(index)] was reshaped by a read")
            #expect(
                allClose(before, after, rtol: 0, atol: 0).item(Bool.self),
                "innerState[\(index)] was mutated by a read")
        }
        #expect(cache.offset == beforeOffset, "logicalView moved the offset")
        #expect(cache.metaState == beforeMeta, "logicalView disturbed idx/offset bookkeeping")
    }

    @Test func testRotatingLogicalViewPreservesPinnedPrefix() throws {
        // `keep` pins the oldest entries; the write path front-trims *after* them, so a view must
        // splice around the prefix rather than take a flat tail.
        let cache = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(cache, positions: 0 ..< 20)

        let view = try #require(cache.logicalView(tail: 5))
        let positions = encodedPositions(view.0)
        #expect(positions.count == 5)
        #expect(Array(positions.prefix(2)) == [0, 1], "pinned prefix was evicted by the view")
        #expect(positions == [0, 1] + Array(positions.suffix(3)))
        #expect(positions.suffix(3).sorted() == Array(positions.suffix(3)), "tail out of order")
    }

    @Test func testRotatingLogicalViewFloorsAtThePinnedPrefix() throws {
        // A pinned entry is never evictable, so `keep` is a floor on the view and not just a
        // splice point: a `tail` below it still returns the prefix. The alternative -- honouring
        // `tail` exactly -- would hand back a context the ring itself can never present.
        let cache = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(cache, positions: 0 ..< 20)

        for tail in 0 ... 2 {
            let view = try #require(cache.logicalView(tail: tail))
            #expect(
                encodedPositions(view.0) == [0, 1],
                "tail \(tail) below `keep` dropped the pinned prefix")
            #expect(encodedPositions(view.1).map { -$0 } == [0, 1], "values diverged from keys")
        }

        // One past the floor is the first tail that actually selects anything.
        let positions = encodedPositions(try #require(cache.logicalView(tail: 3)).0)
        #expect(positions.count == 3)
        #expect(Array(positions.prefix(2)) == [0, 1], "the prefix moved once the tail cleared it")
    }

    @Test func testRotatingLogicalViewFloorIsBoundedByWhatTheCacheHolds() throws {
        // Fewer entries written than `keep` pins: the floor is what exists, not what is reserved.
        let cache = RotatingKVCache(maxSize: 8, keep: 4)
        fillOneAtATime(cache, positions: 0 ..< 2)

        #expect(encodedPositions(try #require(cache.logicalView(tail: 0)).0) == [0, 1])
        #expect(encodedPositions(try #require(cache.logicalView(tail: 9)).0) == [0, 1])
    }

    /// The property the speculative overlay rests on: for a multi-token write, the presentation the
    /// cache *would* return equals `logicalView(tail: maxSize - 1)` followed by the new rows. An
    /// adapter can therefore stage beside the ring and hand the model an identical view.
    @Test func testRotatingLogicalViewPlusNewRowsEqualsTheWritePresentation() throws {
        let window = 8
        let staged = 4

        let cache = RotatingKVCache(maxSize: window, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        // Read the view and the mask *before* the write, exactly as an adapter would.
        let view = try #require(cache.logicalView(tail: window - 1))
        let mask = cache.makeMask(n: staged, windowSize: window, returnArray: false)

        let (newKeys, newValues) = positionedKV(20 ..< (20 + staged))
        let presented = cache.update(keys: newKeys, values: newValues)

        let composedKeys = concatenated([view.0, newKeys], axis: 2)
        let composedValues = concatenated([view.1, newValues], axis: 2)

        #expect(
            encodedPositions(presented.0) == encodedPositions(composedKeys),
            "staged presentation diverged from the live write path")
        #expect(
            allClose(presented.0, composedKeys, rtol: 0, atol: 0).item(Bool.self),
            "keys diverged from the live write path")
        #expect(
            allClose(presented.1, composedValues, rtol: 0, atol: 0).item(Bool.self),
            "values diverged from the live write path")

        // The mask the cache built before the write has to span exactly that presentation, or an
        // adapter could not delegate mask construction to the live cache.
        guard case .array(let maskArray) = mask else {
            Issue.record("expected an array mask once offset + n exceeds the window, got \(mask)")
            return
        }
        #expect(
            maskArray.dim(-1) == composedKeys.dim(2),
            "mask key axis \(maskArray.dim(-1)) != presented length \(composedKeys.dim(2))")
        #expect(maskArray.dim(-2) == staged)
    }

    /// Attention through the rotating cache's own presentation and mask, past a wrap, must equal a
    /// reference that never rotated: every position kept in a plain array under an explicit
    /// causal-intersect-window mask over absolute positions.
    @Test func testRotatingCacheAttentionPastWrapMatchesLogicalOrderReference() {
        withRandomState(MLXRandom.RandomState(seed: 0)) {
            let window = 8
            let history = 20
            let queries = 4
            let heads = 2
            let headDim = 4
            let scale = 1.0 / Float(headDim).squareRoot()

            let allKeys = MLXRandom.normal([1, heads, history + queries, headDim])
            let allValues = MLXRandom.normal([1, heads, history + queries, headDim])
            let q = MLXRandom.normal([1, heads, queries, headDim])

            let cache = RotatingKVCache(maxSize: window, keep: 0)
            for p in 0 ..< history {
                _ = cache.update(
                    keys: allKeys[0..., 0..., p ..< (p + 1), 0...],
                    values: allValues[0..., 0..., p ..< (p + 1), 0...])
            }

            let mask = cache.makeMask(n: queries, windowSize: window, returnArray: false)
            let (cachedKeys, cachedValues) = cache.update(
                keys: allKeys[0..., 0..., history ..< (history + queries), 0...],
                values: allValues[0..., 0..., history ..< (history + queries), 0...])
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: cachedKeys, values: cachedValues, scale: scale, mask: mask)

            // Reference: all history retained, masked by absolute position.
            let queryPositions = MLXArray(Int32(history) ..< Int32(history + queries))[
                0..., .newAxis]
            let keyPositions = MLXArray(Int32(0) ..< Int32(history + queries))[.newAxis]
            let referenceMask =
                (queryPositions .>= keyPositions) & (queryPositions .< keyPositions + Int32(window))
            let reference = MLXFast.scaledDotProductAttention(
                queries: q, keys: allKeys, values: allValues, scale: scale,
                mask: .array(referenceMask))

            #expect(out.shape == reference.shape)
            #expect(
                allClose(out, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self),
                "post-wrap attention diverged from the logical-order reference")
        }
    }

    // MARK: - RotatingKVCache wrap-aware trim
    //
    // Trimming a wrapped ring used to silently corrupt it (the ring invariants cannot
    // express a hole in the timeline). A wrapped trim now linearizes the ring to temporal
    // order and cuts the newest rows: consistent and clamped, though not exact -- rows the
    // rewound writes overwrote at the old edge of the window cannot come back, which is why
    // `isTrimmable` stays false past the window for the exact-rewind machinery. Every
    // consumer of the buffer (writes, masks, state) now bounds live rows by `idx` instead
    // of assuming the logical `offset` still equals the fill.

    @Test func testRotatingTrimAfterWrapCutsNewestAndKeepsChronology() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)  // wrapped ring holding 12 ..< 20

        let trimmed = cache.trim(3)  // rewind 17, 18, 19

        #expect(trimmed == 3)
        #expect(cache.offset == 17)
        let view = try #require(cache.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(12 ..< 17), "trim did not cut the newest rows")
        #expect(
            encodedPositions(view.1).map { -$0 } == Array(12 ..< 17), "values diverged from keys")
    }

    @Test func testRotatingTrimAfterWrapClampsToNonPinnedSpan() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(cache, positions: 0 ..< 13)  // pinned [0, 1], ring holds 7 ..< 13

        let trimmed = cache.trim(100)

        #expect(trimmed == 6, "the pinned prefix must bound the cut")
        #expect(cache.offset == 7)
        let view = try #require(cache.logicalView(tail: 8))
        #expect(
            encodedPositions(view.0) == [0, 1], "the pinned prefix must survive a maximal trim")
    }

    @Test func testRotatingSingleTokenRegrowthAfterWrapTrimPresentsOnlyLiveRows() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)
        cache.trim(3)  // holds 12 ..< 17

        // The buffer regrows from 5 rows; the presentation must never include the
        // freshly allocated (dead) rows.
        let (k, v) = positionedKV(17 ..< 18)
        let presented = cache.update(keys: k, values: v)
        #expect(encodedPositions(presented.0) == Array(12 ..< 18))
        #expect(cache.offset == 18)

        // Refill through the rotation boundary and verify chronology survives re-wrapping.
        fillOneAtATime(cache, positions: 18 ..< 25)
        #expect(cache.offset == 25)
        let view = try #require(cache.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(17 ..< 25))
    }

    @Test func testRotatingMultiTokenWriteAfterWrapTrimStaysChronological() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)
        cache.trim(3)  // holds 12 ..< 17

        // The speculative verify shape: a multi-token write straight after a rewind.
        let (k, v) = positionedKV(17 ..< 21)
        let presented = cache.update(keys: k, values: v)
        #expect(encodedPositions(presented.0) == Array(12 ..< 21))
        #expect(cache.offset == 21)

        fillOneAtATime(cache, positions: 21 ..< 24)
        let view = try #require(cache.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(16 ..< 24))
    }

    @Test func testRotatingRepeatedTrimAfterWrapTakesTemporalPath() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        #expect(cache.trim(2) == 2)  // ring path: holds 12 ..< 18
        #expect(cache.trim(2) == 2)  // temporal path: holds 12 ..< 16

        #expect(cache.offset == 16)
        let view = try #require(cache.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(12 ..< 16))
    }

    @Test func testRotatingWrapTrimSurvivesCopyAndMetaStateRoundTrip() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        let copied = try #require(cache.copy() as? RotatingKVCache)
        #expect(copied.trim(3) == 3)
        let view = try #require(copied.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(12 ..< 17))
        #expect(cache.offset == 20, "trimming the copy must not touch the original")
    }

    @Test func testRotatingLegacyMetaStateDerivesRingLayout() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)

        // A cache saved before the wrapped flag existed has 6 metaState values; the
        // ring layout must be derived, not assumed away.
        let legacyMetaState = Array(cache.metaState.dropLast())
        #expect(legacyMetaState.count == 6)

        let restored = RotatingKVCache(maxSize: 8, keep: 0)
        restored.state = cache.state.map { $0[.ellipsis] }
        restored.metaState = legacyMetaState

        let view = try #require(restored.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(12 ..< 20), "legacy restore lost the ring layout")
        #expect(restored.trim(3) == 3)
        let trimmedView = try #require(restored.logicalView(tail: 8))
        #expect(encodedPositions(trimmedView.0) == Array(12 ..< 17))
    }

    @Test(arguments: [5, 6, 7])
    func testRotatingRestoredTrimAtRingBoundary(metadataCount: Int) throws {
        let source = RotatingKVCache(maxSize: 8)
        fillOneAtATime(source, positions: 0 ..< 16)

        let restored = RotatingKVCache(maxSize: 8)
        restored.state = source.state.map { $0[.ellipsis] }
        restored.metaState = Array(source.metaState.prefix(metadataCount))

        #expect(restored.trim(3) == 3)
        #expect(restored.offset == 13)
        let view = try #require(restored.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(8 ..< 13))
        #expect(encodedPositions(view.1) == Array(8 ..< 13).map { -$0 })
        #expect(restored.state.allSatisfy { $0.dim(2) == 5 })
    }

    @Test(arguments: [5, 6, 7])
    func testRotatingRestoredTrimKeepsKeysAndValuesAligned(metadataCount: Int) throws {
        let source = RotatingKVCache(maxSize: 8)
        fillOneAtATime(source, positions: 0 ..< 20)

        let restored = RotatingKVCache(maxSize: 8)
        restored.state = source.state.map { $0[.ellipsis] }
        restored.metaState = Array(source.metaState.prefix(metadataCount))

        // The shortened key buffer ends before the old write index. Its new shape
        // must not change how the value buffer is ordered during the same trim.
        #expect(restored.trim(5) == 5)
        #expect(restored.offset == 15)
        let view = try #require(restored.logicalView(tail: 8))
        #expect(encodedPositions(view.0) == Array(12 ..< 15))
        #expect(encodedPositions(view.1) == Array(12 ..< 15).map { -$0 })
    }

    private func expectRotatingContents(
        _ cache: RotatingKVCache, _ positions: [Int],
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let view = try #require(cache.logicalView(tail: Int.max), sourceLocation: sourceLocation)
        #expect(view.0.dim(HeadRowsAxis.row) == positions.count, sourceLocation: sourceLocation)
        #expect(view.1.dim(HeadRowsAxis.row) == positions.count, sourceLocation: sourceLocation)
        #expect(
            cache.state.allSatisfy { $0.dim(HeadRowsAxis.row) == positions.count },
            sourceLocation: sourceLocation)
        // Empty MLX arrays need no host readback.
        if !positions.isEmpty {
            #expect(encodedPositions(view.0) == positions, sourceLocation: sourceLocation)
            #expect(
                encodedPositions(view.1) == positions.map { -$0 }, sourceLocation: sourceLocation)
        }
    }

    @Test(arguments: [5, 6, 7], [false, true])
    func testRotatingRestorationAcceptsEitherSetterOrder(
        metadataCount: Int, metadataFirst: Bool
    ) throws {
        for count in [3, 8, 9, 16, 20] {
            for prefill in [false, true] {
                let source = RotatingKVCache(maxSize: 8, step: 4)
                if prefill {
                    let (k, v) = positionedKV(0 ..< count)
                    _ = source.update(keys: k, values: v)
                } else {
                    fillOneAtATime(source, positions: 0 ..< count)
                }
                let metadata = Array(source.metaState.prefix(metadataCount))
                let arrays = source.state.map { $0[.ellipsis] }
                // The saved capacity must replace the constructor's capacity before inference.
                let restored = RotatingKVCache(maxSize: 1)
                if metadataFirst {
                    restored.metaState = metadata
                    restored.state = arrays
                } else {
                    restored.state = arrays
                    restored.metaState = metadata
                }

                let start = prefill ? 0 : max(0, count - 8)
                try expectRotatingContents(restored, Array(start ..< count))
                #expect(restored.trim(2) == 2)
                #expect(restored.offset == count - 2)
                try expectRotatingContents(restored, Array(start ..< (count - 2)))
            }
        }
    }

    @Test(arguments: [5, 6, 7], [false, true])
    func testRotatingRestoredShortChronologicalBufferCanGrow(
        metadataCount: Int, metadataFirst: Bool
    ) throws {
        let source = RotatingKVCache(maxSize: 8)
        fillOneAtATime(source, positions: 0 ..< 20)
        // Legacy updateConcat could leave maxSize - 1 chronological rows when given
        // an empty append. idx < maxSize && offset > idx cannot identify its layout.
        let empty = positionedKV(20 ..< 20)
        _ = source.update(keys: empty.0, values: empty.1)
        try expectRotatingContents(source, Array(13 ..< 20))

        let restored = RotatingKVCache(maxSize: 8)
        let metadata = Array(source.metaState.prefix(metadataCount))
        let arrays = source.state.map { $0[.ellipsis] }
        if metadataFirst {
            restored.metaState = metadata
            restored.state = arrays
        } else {
            restored.state = arrays
            restored.metaState = metadata
        }
        #expect(restored.metaState.last == "false")
        fillOneAtATime(restored, positions: 20 ..< 21)
        try expectRotatingContents(restored, Array(13 ..< 21))
        #expect(restored.trim(3) == 3)
        try expectRotatingContents(restored, Array(13 ..< 18))
    }

    @Test(arguments: [5, 6, 7], [13, 14, 20])
    func testRotatingRestoredTrimPreservesPinnedPrefix(metadataCount: Int, count: Int) throws {
        let source = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(source, positions: 0 ..< count)

        for firstTrim in [3, 100] {
            let restored = RotatingKVCache(maxSize: 8)
            restored.state = source.state.map { $0[.ellipsis] }
            restored.metaState = Array(source.metaState.prefix(metadataCount))
            let removed = min(firstTrim, 6)
            #expect(restored.trim(firstTrim) == removed)
            try expectRotatingContents(
                restored, [0, 1] + Array((count - 6) ..< (count - removed)))
            // The second trim operates on temporal storage, even if the first used a ring.
            #expect(restored.trim(100) == 6 - removed)
            #expect(restored.trim(1) == 0)
            #expect(restored.offset == count - 6)
            try expectRotatingContents(restored, [0, 1])

            let next = restored.offset
            fillOneAtATime(restored, positions: next ..< (next + 8))
            try expectRotatingContents(restored, [0, 1] + Array((next + 2) ..< (next + 8)))
        }
        try expectRotatingContents(source, [0, 1] + Array((count - 6) ..< count))
    }

    @Test(arguments: [1, 2, 5, 8])
    func testRotatingTrimBeforeEvictionCanRewindThroughPrefix(count: Int) throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 2)
        fillOneAtATime(cache, positions: 0 ..< count)
        let metadata = cache.metaState
        #expect(cache.trim(0) == 0)
        #expect(cache.trim(-1) == 0)
        #expect(cache.metaState == metadata)
        #expect(cache.trim(100) == count)
        #expect(cache.offset == 0)
        try expectRotatingContents(cache, [])
        fillOneAtATime(cache, positions: 0 ..< 3)
        try expectRotatingContents(cache, [0, 1, 2])
    }

    @Test(arguments: [0, 2], [1, 3])
    func testRotatingTrimOversizedPrefillThenResume(keep: Int, nextCount: Int) throws {
        let cache = RotatingKVCache(maxSize: 8, keep: keep, step: 4)
        fillOneAtATime(cache, positions: 0 ..< 20)
        let (k, v) = positionedKV(20 ..< 23)
        _ = cache.update(keys: k, values: v)  // ten chronological rows
        #expect(cache.trim(5) == 5)
        #expect(cache.offset == 18)
        let prefix = Array(0 ..< keep)
        try expectRotatingContents(cache, prefix + Array((13 + keep) ..< 18))

        let mask = cache.makeMask(n: nextCount, windowSize: 8, returnArray: true)
        let next = positionedKV(18 ..< (18 + nextCount))
        let presented = cache.update(keys: next.0, values: next.1)
        let expected = prefix + Array((13 + keep) ..< (18 + nextCount))
        #expect(encodedPositions(presented.0) == expected)
        #expect(encodedPositions(presented.1) == expected.map { -$0 })
        if case .array(let array) = mask {
            #expect(array.dim(-1) == presented.0.dim(2))
        }
        fillOneAtATime(cache, positions: (18 + nextCount) ..< 30)
        try expectRotatingContents(cache, prefix + Array((22 + keep) ..< 30))
    }

    @Test(arguments: [5, 6, 7])
    func testRotatingTrimSurvivesPromptCachePersistenceAndCopy(metadataCount: Int) throws {
        let source = RotatingKVCache(maxSize: 8)
        fillOneAtATime(source, positions: 0 ..< 16)
        let url = TemporaryPromptCacheFile.makeURL()
        let legacyURL = TemporaryPromptCacheFile.makeURL()
        let roundTripURL = TemporaryPromptCacheFile.makeURL()
        defer {
            for file in [url, legacyURL, roundTripURL] {
                try? FileManager.default.removeItem(at: file)
            }
        }
        try savePromptCache(url: url, cache: [source])
        // Exercise the real file loader with each historical metadata format.
        let (arrays, savedMetadata) = try loadArraysAndMetadata(url: url)
        var metadata = savedMetadata
        for index in metadataCount ..< 7 {
            metadata.removeValue(forKey: "0.0.\(index)")
        }
        // MLX loads lazily, so keep the source files intact while their arrays are live.
        try save(arrays: arrays, metadata: metadata, url: legacyURL)
        let (loaded, _) = try loadPromptCache(url: legacyURL)
        let restored = try #require(loaded.first as? RotatingKVCache)
        #expect(restored.trim(3) == 3)
        try expectRotatingContents(restored, Array(8 ..< 13))

        // A post-trim cache has more history than live rows; its explicit false flag
        // must survive serialization instead of being inferred from those counters.
        try savePromptCache(url: roundTripURL, cache: [restored])
        let (reloaded, _) = try loadPromptCache(url: roundTripURL)
        let roundTrip = try #require(reloaded.first as? RotatingKVCache)
        let copied = try #require(roundTrip.copy() as? RotatingKVCache)
        #expect(copied.trim(2) == 2)
        fillOneAtATime(copied, positions: 11 ..< 20)
        try expectRotatingContents(copied, Array(12 ..< 20))
        try expectRotatingContents(roundTrip, Array(8 ..< 13))
        try expectRotatingContents(restored, Array(8 ..< 13))
        try expectRotatingContents(source, Array(8 ..< 16))
        #expect(roundTrip.offset == 13)
        #expect(source.offset == 16)
    }

    @Test(arguments: [5, 6, 7], [1, 3])
    func testRotatingRestoredTrimAttentionMatchesSurvivingContext(
        metadataCount: Int, queryCount: Int
    ) throws {
        let keys = deterministicTensor(shape: [1, 2, 20, 4], salt: 1).asType(.float32)
        let values = deterministicTensor(shape: [1, 2, 20, 4], salt: 2).asType(.float32)
        let queries = deterministicTensor(shape: [1, 2, queryCount, 4], salt: 3).asType(.float32)
        let source = RotatingKVCache(maxSize: 8)
        for p in 0 ..< 20 {
            _ = source.update(
                keys: keys[.ellipsis, p ..< (p + 1), 0...],
                values: values[.ellipsis, p ..< (p + 1), 0...])
        }
        let restored = RotatingKVCache(maxSize: 8)
        restored.state = source.state.map { $0[.ellipsis] }
        restored.metaState = Array(source.metaState.prefix(metadataCount))
        #expect(restored.trim(5) == 5)  // only positions 12, 13, 14 survive
        let mask = restored.makeMask(n: queryCount, windowSize: 4, returnArray: false)
        let (cachedKeys, cachedValues) = restored.update(
            keys: keys[.ellipsis, 15 ..< (15 + queryCount), 0...],
            values: values[.ellipsis, 15 ..< (15 + queryCount), 0...])
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues, scale: 0.5, mask: mask)
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys[.ellipsis, 12 ..< (15 + queryCount), 0...],
            values: values[.ellipsis, 12 ..< (15 + queryCount), 0...],
            scale: 0.5,
            mask: .array(createCausalMask(n: queryCount, offset: 3, windowSize: 4)))
        #expect(allClose(output, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }

    @Test func testRotatingSingleTokenMaskAfterWrapTrimMatchesEquivalentFreshCache() throws {
        // After a wrapped trim the cache holds 5 live rows; the sliding-window mask for
        // the next single-token step must match a fresh cache holding the same rows.
        let windowSize = 4

        let trimmedCache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(trimmedCache, positions: 0 ..< 20)
        trimmedCache.trim(3)  // 5 live rows

        let freshCache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(freshCache, positions: 0 ..< 5)  // 5 live rows

        let trimmedMask = trimmedCache.makeMask(n: 1, windowSize: windowSize, returnArray: false)
        let freshMask = freshCache.makeMask(n: 1, windowSize: windowSize, returnArray: false)

        guard case .array(let trimmedArray) = trimmedMask, case .array(let freshArray) = freshMask
        else {
            Issue.record("expected array masks, got \(trimmedMask) and \(freshMask)")
            return
        }
        #expect(trimmedArray.shape == freshArray.shape)
        #expect((trimmedArray .== freshArray).all().item(Bool.self))
    }

    @Test func testRotatingMultiTokenMaskWidthMatchesPresentationAfterWrapTrim() throws {
        let cache = RotatingKVCache(maxSize: 8, keep: 0)
        fillOneAtATime(cache, positions: 0 ..< 20)
        cache.trim(3)  // 5 live rows

        // Models build the mask before the write; its key width must equal the rows the
        // write presents, or attention shapes diverge.
        let mask = cache.makeMask(n: 3, windowSize: 4, returnArray: true)
        let (k, v) = positionedKV(17 ..< 20)
        let presented = cache.update(keys: k, values: v)

        guard case .array(let maskArray) = mask else {
            Issue.record("expected an array mask, got \(mask)")
            return
        }
        #expect(maskArray.dim(-1) == presented.0.dim(2))
    }
}
