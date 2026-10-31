// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import Testing

/// The values of ``createCausalMask(n:offset:windowSize:lengths:)`` after cached positions: each
/// new position attends to the key positions at or before it, inside the window, and before the
/// length of its sequence.
struct CausalMaskTests {

    /// The new positions of each mask.
    private static let queryCount = 3

    /// The cached positions before them.
    private static let cachedCount = 5

    /// The window of a sliding mask.
    private static let windowSize = 4

    /// The lengths of a batch of two sequences: the first ends before the last new position, the
    /// second spans the whole mask.
    private static let lengths = [6, cachedCount + queryCount]

    /// The key positions of each mask: the cached and the new ones.
    private static var keyCount: Int { cachedCount + queryCount }

    /// The mask the rule gives, built position by position on the host and flattened by row.
    ///
    /// - Parameters:
    ///   - windowSize: The window, or `nil` for full attention.
    ///   - length: The length of the sequence, or `nil` for one that spans the mask.
    /// - Returns: One value for each new position and key position.
    private static func reference(windowSize: Int?, length: Int?) -> [Bool] {
        (0 ..< queryCount).flatMap { row in
            let query = cachedCount + row
            return (0 ..< keyCount).map { key in
                key <= query && query < key + (windowSize ?? keyCount) && key < (length ?? keyCount)
            }
        }
    }

    @Test func maskHasARowForEachNewPositionAndAColumnForEachKey() {
        let mask = createCausalMask(n: Self.queryCount, offset: Self.cachedCount)
        #expect(mask.shape == [Self.queryCount, Self.keyCount])
    }

    @Test func newPositionsAttendToEveryKeyUpToThemselves() {
        let mask = createCausalMask(n: Self.queryCount, offset: Self.cachedCount)
        #expect(mask.asArray(Bool.self) == Self.reference(windowSize: nil, length: nil))
    }

    @Test func slidingMaskKeepsTheWindowBeforeEachNewPosition() {
        let mask = createCausalMask(
            n: Self.queryCount, offset: Self.cachedCount, windowSize: Self.windowSize)
        #expect(mask.asArray(Bool.self) == Self.reference(windowSize: Self.windowSize, length: nil))
    }

    @Test func lengthsMaskTheKeysPastEachSequence() {
        let mask = createCausalMask(
            n: Self.queryCount, offset: Self.cachedCount,
            lengths: MLXArray(Self.lengths.map(Int32.init)))
        #expect(
            mask.asArray(Bool.self)
                == Self.lengths.flatMap { Self.reference(windowSize: nil, length: $0) })
    }
}
