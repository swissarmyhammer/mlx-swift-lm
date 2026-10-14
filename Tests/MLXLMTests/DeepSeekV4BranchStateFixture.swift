// Copyright © 2026 Apple Inc.

import MLX

/// The serialized state of one DeepSeek-V4 pooled-cache branch, for the test
/// suites that write a branch state into a cache.
///
/// More than one suite writes this state. Each suite gives its own array
/// maker, thus a suite that counts bytes can use lazy arrays and a suite that
/// compares values can use seeded arrays.
enum DeepSeekV4BranchStateFixture {

    /// The number of row slots of one branch: the projected values, the
    /// projected gate logits, and the unprojected rows of the run in progress.
    ///
    /// The fixture gives every row slot the same row count and width. The
    /// cache checks only the rank of the unprojected rows and that they are
    /// fewer than one chunk, thus a row count under the compress ratio of the
    /// layer is a valid state.
    static let rowSlotCount = 3

    /// The place of the pooled chunks inside the state of one branch.
    static let chunksSlot = 0

    /// The place of the first projected row slot inside the state of one
    /// branch.
    static let firstRowSlot = chunksSlot + 1

    /// Makes the five state slots of one branch.
    ///
    /// - Parameters:
    ///   - batchSize: The batch size of each tensor.
    ///   - chunkCount: The number of pooled chunks.
    ///   - carryRowCount: The number of rows of each row slot.
    ///   - rowWidth: The width of a chunk and of a row.
    ///   - makeArray: Makes the tensor of one slot from its shape and the
    ///     place of the slot inside the state.
    /// - Returns: The pooled chunks, the projected values, the projected gate
    ///   logits, the unprojected rows and the carry start, which is 0.
    static func slots(
        batchSize: Int, chunkCount: Int, carryRowCount: Int, rowWidth: Int,
        makeArray: (_ shape: [Int], _ slot: Int) -> MLXArray
    ) -> [MLXArray] {
        [makeArray([batchSize, chunkCount, rowWidth], chunksSlot)]
            + (firstRowSlot ..< firstRowSlot + rowSlotCount).map {
                makeArray([batchSize, carryRowCount, rowWidth], $0)
            }
            + [MLXArray([Int32(0)])]
    }
}
