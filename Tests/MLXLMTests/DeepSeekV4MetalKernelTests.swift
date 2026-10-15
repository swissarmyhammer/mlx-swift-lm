// Copyright © 2026 Apple Inc.
//
// The contract of the shared helpers of ``DeepSeekV4MetalKernel``, which
// every DeepSeek-V4 Metal kernel reads for its dispatch shape.

import Foundation
import MLX
import Testing

@testable import MLXLLM

struct DeepSeekV4MetalKernelTests {

    /// The number of items in one group of the cover test.
    private static let groupSize = 4

    /// The cases of the cover test: the number of items and the number of
    /// groups of ``groupSize`` that cover them.
    private static let coverCases: [(count: Int, groups: Int)] = [
        (0, 0), (1, 1), (groupSize - 1, 1), (groupSize, 1), (groupSize + 1, 2),
        (3 * groupSize, 3),
    ]

    @Test(arguments: coverCases)
    func theGroupCountCoversEachItemWithTheFewestGroups(count: Int, groups: Int) {
        #expect(
            DeepSeekV4MetalKernel.groupCount(covering: count, groupSize: Self.groupSize) == groups)
    }

    @Test
    func theTokenShapeOfAResidualStreamDropsTheCopyAndTheWidthAxes() {
        // A stream of 2 sequences of 3 tokens, each token 4 copies of width 8.
        let stream = MLXArray.zeros([2, 3, 4, 8])

        #expect(DeepSeekV4MetalKernel.tokenShape(of: stream) == [2, 3])
    }
}
