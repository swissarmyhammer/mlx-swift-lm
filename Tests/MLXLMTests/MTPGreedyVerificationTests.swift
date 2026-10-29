// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct MTPGreedyVerificationTests {
    @Test(arguments: [2, 4, 8], [DType.float32, .float16, .bfloat16])
    func matchesSequentialVerification(blockSize: Int, dtype: DType) {
        for accepted in 0 ..< blockSize {
            var data = [Float](repeating: -10, count: blockSize * 32)
            for row in 0 ..< blockSize {
                let token = row < accepted ? 7 : 9
                data[row * 32 + token] = 10
                // Equal maxima must keep the sequential argmax's tie breaking.
                data[row * 32 + token + 1] = 10
            }
            let logits = MLXArray(data, [blockSize, 32]).asType(dtype)
            let drafts = MLXArray(Array(repeating: Int32(7), count: blockSize - 1))
            let reference = SequentialGreedyMTPVerification.verify(
                logits: logits, draftTokens: drafts)
            let batched = batchedGreedyMTPVerification(logits: logits, draftTokens: drafts)

            #expect(batched.accepted == accepted)
            #expect(batched.accepted == reference.accepted)
            #expect(batched.tokens == reference.tokens)
            #expect(batched.finalToken.shape == reference.finalToken.shape)
            #expect(batched.finalToken.dtype == reference.finalToken.dtype)
            #expect(batched.finalToken.item(Int.self) == reference.finalToken.item(Int.self))
        }
    }

    @Test
    func suppressionMatchesSequentialVerification() throws {
        let suppressor = try #require(SuppressTokensProcessor(tokenIds: [7, 1000]))
        let logits = MLXArray(
            [
                Float(0), 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
                0, 1, 2, 3, 4, 5, 6, 100,
            ], [4, 8])
        let drafts = MLXArray([Int32(6), 6, 7])
        let batched = batchedGreedyMTPVerification(
            logits: suppressor.process(logits: logits), draftTokens: drafts)
        let reference = SequentialGreedyMTPVerification.verify(
            logits: concatenated(
                (0 ..< 4).map {
                    suppressor.process(logits: logits[$0 ..< ($0 + 1)])
                }), draftTokens: drafts)
        #expect(batched.accepted == 2)
        #expect(batched.tokens == [6, 6, 6])
        #expect(batched.tokens == reference.tokens)
    }
}
