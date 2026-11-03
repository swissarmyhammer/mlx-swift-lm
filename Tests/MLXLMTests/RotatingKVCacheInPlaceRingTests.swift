// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// A ring with a rewind reserve writes short runs of rows in place and rewinds by moving its
/// write position back. These tests drive it next to the linearizing path
/// (``RotatingKVCache/writesRingInPlace`` `false`) and check that both present the same keys,
/// values and masks at the same offsets.
final class RotatingKVCacheInPlaceRingTests: XCTestCase {
    private static let window = 8
    private static let valueShift = 1000

    /// One operation of a schedule.
    private enum Operation {
        case write(Int)
        case rewind(Int)
        case restore
    }

    /// The cache under test and its linearizing reference, with the next logical position.
    private struct Pair {
        var inPlace: RotatingKVCache
        let reference: RotatingKVCache
        var position: Int

        init(keep: Int, step: Int, reserve: Int) {
            inPlace = RotatingKVCache(
                maxSize: RotatingKVCacheInPlaceRingTests.window, keep: keep, step: step,
                rewindCapacity: reserve)
            reference = RotatingKVCache(
                maxSize: RotatingKVCacheInPlaceRingTests.window, keep: keep, step: step,
                rewindCapacity: reserve)
            reference.writesRingInPlace = false
            position = 0
        }
    }

    private func keys(_ positions: Range<Int>) -> MLXArray {
        MLXArray(positions.map(Float.init)).reshaped(1, 1, positions.count, 1)
    }

    private func values(_ positions: Range<Int>) -> MLXArray {
        MLXArray(positions.map { Float($0 + Self.valueShift) }).reshaped(1, 1, positions.count, 1)
    }

    private func floats(_ array: MLXArray) -> [Float] {
        array.dim(2) == 0 ? [] : array.asArray(Float.self)
    }

    /// The mask as comparable text: its case, and the values of an array mask.
    private func describe(_ mode: MLXFast.ScaledDotProductAttentionMaskMode) -> String {
        switch mode {
        case .none: return "none"
        case .causal: return "causal"
        case .array(let mask): return "array \(mask.shape) \(mask.asArray(Bool.self))"
        case .arrays(let masks): return "arrays \(masks.map { $0.asArray(Bool.self) })"
        @unknown default: return "unknown"
        }
    }

    /// The saved rows of `cache` in temporal order, read from its state and metadata alone.
    private func savedTemporalRows(_ cache: RotatingKVCache) -> [Float] {
        let state = cache.state
        guard let saved = state.first else { return [] }
        let meta = cache.metaState
        let keep = Int(meta[0])!
        let idx = Int(meta[4])!
        let wrapped = Bool(meta[6])!
        let rows = floats(saved)
        guard wrapped else { return Array(rows.prefix(idx)) }
        return Array(rows[0 ..< keep]) + Array(rows[idx...]) + Array(rows[keep ..< idx])
    }

    private func assertSameObservables(
        _ pair: Pair, _ context: @autoclosure () -> String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let message = context()
        XCTAssertEqual(pair.inPlace.offset, pair.reference.offset, message, file: file, line: line)
        // The linearizing path keeps up to `count - 1` more rows right after a write of
        // `count` rows, so it reports fewer evicted rows until its next single-row write.
        XCTAssertGreaterThanOrEqual(
            pair.inPlace.evictedTokenCount, pair.reference.evictedTokenCount, message,
            file: file, line: line)
        XCTAssertLessThanOrEqual(
            pair.inPlace.evictedTokenCount,
            pair.reference.evictedTokenCount + pair.inPlace.rewindCapacity, message, file: file,
            line: line)
        XCTAssertEqual(
            pair.inPlace.isTrimmable, pair.reference.isTrimmable, message, file: file, line: line)
        // The rows the reference holds past the in-place ring are its oldest ones, so the
        // in-place ring rewinds exactly that many rows less.
        let extraRows = pair.inPlace.evictedTokenCount - pair.reference.evictedTokenCount
        XCTAssertEqual(
            pair.inPlace.maxTrimCount, max(0, pair.reference.maxTrimCount - extraRows), message,
            file: file, line: line)
        for tail in [0, 3, Self.window, 3 * Self.window] {
            let actual = pair.inPlace.logicalView(tail: tail)
            let expected = pair.reference.logicalView(tail: tail)
            XCTAssertEqual(
                actual.map { floats($0.0) }, expected.map { floats($0.0) }, message, file: file,
                line: line)
            XCTAssertEqual(
                actual.map { floats($0.1) }, expected.map { floats($0.1) }, message, file: file,
                line: line)
        }
        let inPlaceRows = savedTemporalRows(pair.inPlace)
        let referenceRows = savedTemporalRows(pair.reference)
        // The reference may hold older rows past the reserve right after a long write; the
        // newest rows that the in-place ring saves are the same rows.
        XCTAssertEqual(
            inPlaceRows,
            referenceRows.count > inPlaceRows.count
                ? Array(referenceRows.prefix(pair.inPlace.keepCount))
                    + Array(
                        referenceRows.suffix(inPlaceRows.count - pair.inPlace.keepCount))
                : referenceRows,
            message, file: file, line: line)
    }

    private func write(
        _ count: Int, _ pair: inout Pair, _ context: String, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let positions = pair.position ..< (pair.position + count)
        for window: Int? in [nil, Self.window - 3] {
            for returnArray in [false, true] {
                XCTAssertEqual(
                    describe(
                        pair.inPlace.makeMask(
                            n: count, windowSize: window, returnArray: returnArray)),
                    describe(
                        pair.reference.makeMask(
                            n: count, windowSize: window, returnArray: returnArray)),
                    "\(context) mask window=\(String(describing: window))", file: file, line: line)
            }
        }
        let actual = pair.inPlace.update(keys: keys(positions), values: values(positions))
        let expected = pair.reference.update(keys: keys(positions), values: values(positions))
        XCTAssertEqual(
            floats(actual.0), floats(expected.0), "\(context) keys", file: file, line: line)
        XCTAssertEqual(
            floats(actual.1), floats(expected.1), "\(context) values", file: file, line: line)
        pair.position += count
    }

    private func run(
        _ operations: [Operation], _ pair: inout Pair, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        for (index, operation) in operations.enumerated() {
            let context = "\(label) op \(index) \(operation)"
            switch operation {
            case .write(let count):
                write(count, &pair, context, file: file, line: line)
            case .rewind(let requested):
                // Right after a long write the linearizing path holds more rows past the
                // reserve; a rewind there stays inside what the in-place ring holds exactly.
                let count =
                    pair.inPlace.evictedTokenCount == pair.reference.evictedTokenCount
                    ? requested : min(requested, pair.inPlace.maxTrimCount)
                let trimmed = pair.inPlace.trim(count)
                XCTAssertEqual(
                    trimmed, pair.reference.trim(count), context, file: file, line: line)
                pair.position -= trimmed
            case .restore:
                let restored = try RotatingKVCache.restore(
                    state: pair.inPlace.state, metaState: pair.inPlace.metaState)
                pair.inPlace = try XCTUnwrap(restored as? RotatingKVCache, context)
            }
            assertSameObservables(pair, context, file: file, line: line)
        }
    }

    /// A deterministic schedule of short writes (up to two past the reserve), long writes,
    /// rewinds inside and past the reserve, and snapshot round trips.
    private func schedule(seed: UInt64, reserve: Int, length: Int) -> [Operation] {
        var state = seed
        func next(_ bound: Int) -> Int {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Int((state >> 33) % UInt64(bound))
        }
        let writeCounts = [1, 1, 1, 2, 3, reserve, reserve + 1, reserve + 2, 16]
        return [.write(Self.window + reserve + 3)]
            + (0 ..< length).map { _ in
                switch next(10) {
                case 0 ..< 5: return .write(max(1, writeCounts[next(writeCounts.count)]))
                case 5 ..< 9: return .rewind(1 + next(reserve + 3))
                default: return .restore
                }
            }
    }

    func testInPlaceRingMatchesTheLinearizingPathOverMixedSchedules() throws {
        for keep in [0, 2, 3] {
            for reserve in [1, 4, 10] {
                for step in [1, 4, 256] {
                    for seed: UInt64 in [1, 7] {
                        var pair = Pair(keep: keep, step: step, reserve: reserve)
                        try run(
                            schedule(seed: seed, reserve: reserve, length: 160), &pair,
                            "keep=\(keep) reserve=\(reserve) step=\(step) seed=\(seed)")
                    }
                }
            }
        }
    }

    func testShortWriteSplitsAtTheRingEndInPlace() throws {
        for keep in [0, 2] {
            var pair = Pair(keep: keep, step: 4, reserve: 6)
            let capacity = Self.window + 6
            // Wrap, fill to two rows before the end of the ring, then write five rows across it.
            try run(
                Array(repeating: .write(1), count: 2 * capacity - keep - 2) + [.write(5)],
                &pair, "keep=\(keep)")
            XCTAssertEqual(pair.inPlace.innerState()[0].dim(2), capacity)
            // A full ring and more single and short writes keep the buffer at the capacity.
            try run(
                [.write(1), .write(3), .write(6), .write(7), .write(1), .write(2)], &pair,
                "keep=\(keep) wrapped")
            XCTAssertEqual(pair.inPlace.innerState()[0].dim(2), capacity)
        }
    }

    func testFirstShortWriteAfterAnEvictingPrefillCompactsOnce() throws {
        for keep in [0, 3] {
            var pair = Pair(keep: keep, step: 256, reserve: 10)
            // The second prefill chunk evicts rows and leaves an oversized buffer.
            try run([.write(30), .write(20), .write(4)], &pair, "keep=\(keep)")
            XCTAssertEqual(pair.inPlace.innerState()[0].dim(2), Self.window + 10)
            try run([.write(1), .write(4), .rewind(3), .write(2)], &pair, "keep=\(keep) after")
        }
    }

    func testRewindsAcrossTheWrapStayInPlace() throws {
        for keep in [0, 2] {
            for reserve in [4, 10] {
                var pair = Pair(keep: keep, step: 4, reserve: reserve)
                let capacity = Self.window + reserve
                // Wrap, then put the write position two rows past the kept prefix.
                try run(
                    Array(repeating: .write(1), count: 2 * capacity - keep + 2), &pair,
                    "keep=\(keep) reserve=\(reserve) fill")
                // Rewind across the wrap: the ring keeps its buffer.
                try run(
                    [.rewind(min(reserve, 4))], &pair, "keep=\(keep) reserve=\(reserve) rewind")
                XCTAssertEqual(pair.inPlace.innerState()[0].dim(2), capacity)
                // Write across it again, and rewind past it twice.
                try run(
                    [
                        .write(1), .write(2), .rewind(3), .rewind(1), .write(3), .write(1),
                        .rewind(min(reserve, 5)),
                    ], &pair, "keep=\(keep) reserve=\(reserve) rewinds")
                XCTAssertEqual(pair.inPlace.innerState()[0].dim(2), capacity)
                try run([.write(1)], &pair, "keep=\(keep) reserve=\(reserve) refill")
            }
        }
    }

    func testFirstWrapOfTheWholeHistoryKeepsEveryRow() throws {
        for keep in [0, 2] {
            var pair = Pair(keep: keep, step: 4, reserve: 10)
            // Nothing is evicted before the write that wraps, so it concatenates, and the whole
            // history stays rewindable until the next single-row write.
            try run([.write(15), .write(10)], &pair, "keep=\(keep)")
            XCTAssertEqual(pair.inPlace.maxTrimCount, 25)
            try run([.rewind(13), .write(1), .write(3)], &pair, "keep=\(keep) after")
        }
    }

    func testRewindsPastTheReserveMatchTheLinearizingPath() throws {
        for keep in [0, 2] {
            var pair = Pair(keep: keep, step: 4, reserve: 4)
            try run(
                Array(repeating: .write(1), count: 30)
                    + [
                        .rewind(7), .write(1), .rewind(9), .write(3), .write(1), .rewind(20),
                        .write(5), .write(5), .write(1), .rewind(2), .write(1),
                    ], &pair, "keep=\(keep)")
        }
    }

    func testRewoundRingSavesTheSnapshotOfTheLinearizingPath() throws {
        for keep in [0, 2] {
            var pair = Pair(keep: keep, step: 4, reserve: 10)
            try run(
                Array(repeating: .write(1), count: 41) + [.rewind(6)], &pair, "keep=\(keep)")
            XCTAssertEqual(pair.inPlace.metaState, pair.reference.metaState)
            XCTAssertEqual(
                pair.inPlace.state.map(floats), pair.reference.state.map(floats))
            // The snapshot restores and continues as the reference does.
            try run([.restore, .write(1), .write(4), .rewind(2), .write(3)], &pair, "restored")
        }
    }

    func testRingWithoutAReserveKeepsTheLinearizingPath() throws {
        let cache = RotatingKVCache(maxSize: Self.window, keep: 2, step: 4)
        let reference = RotatingKVCache(maxSize: Self.window, keep: 2, step: 4)
        reference.writesRingInPlace = false
        var position = 0
        for count in [1, 1, 9, 1, 3, 1, 1] {
            let range = position ..< (position + count)
            let actual = cache.update(keys: keys(range), values: values(range))
            let expected = reference.update(keys: keys(range), values: values(range))
            XCTAssertEqual(floats(actual.0), floats(expected.0))
            XCTAssertEqual(cache.metaState, reference.metaState)
            position += count
        }
        XCTAssertEqual(cache.trim(2), reference.trim(2))
        XCTAssertEqual(cache.metaState, reference.metaState)
        XCTAssertEqual(cache.state.map(floats), reference.state.map(floats))
    }
}
