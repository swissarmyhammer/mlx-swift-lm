// Copyright © 2026 Apple Inc.
//
// The contract of ``DecodeAttentionKernel/BlockPass``.
//
// The decode, the verify and the quantized attention kernels run a block
// pass with the same inputs and outputs: the queries, the cache arrays, the
// score factor and the keys of one block in, the weighted values, the
// largest scores and the weight sums of each block of each query row out.
// One value builds them for each kernel.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

struct AttentionBlockPassTests {

    /// The sequences of the tests.
    static let batch = 2

    /// The query heads of the tests.
    static let queryHeads = 4

    /// The KV heads of the tests.
    static let kvHeads = 2

    /// The query tokens of the tests: a verify of three tokens.
    static let tokens = 3

    /// The width of one head.
    static let headDim = 128

    /// The block count of the tests.
    static let blocks = 8

    /// The number of cached keys of the tests: the last block is shorter
    /// than the others.
    static let keyCount = 3001

    /// The score factor of the tests.
    static let scale: Float = 0.5

    /// The block pass of the tests, over one cache array.
    let pass = DecodeAttentionKernel.BlockPass(
        queries: MLXArray.zeros([batch, queryHeads, tokens, headDim], type: Float.self),
        cache: [MLXArray.zeros([batch, kvHeads, keyCount, headDim], type: Float.self)],
        scale: scale, keyCount: keyCount, blocks: blocks)

    /// Each query row of each block has a weighted value of one head width, a
    /// largest score and a weight sum. The rows are the sequences, the query
    /// heads and the tokens of the queries.
    @Test func theOutputsHoldOneResultForEachRowOfEachBlock() {
        let rows = Self.batch * Self.queryHeads * Self.tokens

        #expect(
            pass.outputShapes == [
                [rows, Self.blocks, Self.headDim], [rows, Self.blocks], [rows, Self.blocks],
            ])
    }

    /// The inputs are the queries, the cache arrays, the score factor and the
    /// keys of one block: here one cache array.
    @Test func theInputsAreTheQueriesTheCacheTheScaleAndTheBlockKeys() {
        #expect(pass.inputs.count == 4)
    }

    /// The score factor stands after the cache arrays, as one float32.
    @Test func theScoreFactorFollowsTheCache() {
        #expect(pass.inputs[2].asArray(Float.self) == [Self.scale])
    }

    /// The last input holds the keys of each block but the last, as one
    /// int32.
    @Test func theLastInputHoldsTheKeysOfOneBlock() {
        let keysPerBlock = DecodeAttentionKernel.keysPerBlock(
            keys: Self.keyCount, blocks: Self.blocks)

        #expect(pass.inputs[3].asArray(Int32.self) == [Int32(keysPerBlock)])
    }
}
