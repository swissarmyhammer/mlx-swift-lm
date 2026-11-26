// Copyright © 2026 Apple Inc.

import MLX
import Testing

@testable import MLXGuidedGeneration

/// Greedy guided decoding at an exact logit tie: after the grammar mask and the bias, the
/// lowest tied token id wins, whatever the element type of the model's logits (^yc6jghy).
@Suite(.serialized)
struct GuidedGreedyTieTests {

    /// The logit dimension of the Qwen 3.5 family.
    static let logitDimension = 248_320

    /// The tokens the grammar allows: the digits 1 to 4 of the Qwen tokenizer, as the
    /// `nights` field of the parity suite's trip plan allows them.
    static let allowedTokens = 16 ... 19

    /// The tied tokens: the digits 2 and 3, the tie of the parity suite at its step 44.
    static let tiedTokens = [18, 17]

    /// The tied logit; every other allowed logit is lower. Exact in each element type.
    static let tiedLogit: Float = 9.5

    /// The logits of one position, `[1, 1, logitDimension]`: the tie, a higher logit on a
    /// token the grammar excludes, and lower logits elsewhere.
    static func logits(dtype: DType) -> MLXArray {
        var values = [Float](repeating: -4, count: logitDimension)
        for token in tiedTokens { values[token] = tiedLogit }
        values[allowedTokens.upperBound + 1] = tiedLogit + 1
        return MLXArray(values).asType(dtype)[.newAxis, .newAxis]
    }

    /// The additive grammar mask that allows ``allowedTokens`` alone.
    static func mask() -> MLXArray {
        var words = [UInt32](repeating: 0, count: logitDimension / UInt32.bitWidth + 1)
        for token in allowedTokens {
            words[token / UInt32.bitWidth] |= 1 << UInt32(token % UInt32.bitWidth)
        }
        return words.withUnsafeBufferPointer { buffer in
            GuidedGenerationLoop.bitmaskToMLXArray(
                buffer.baseAddress!, maskBitCount: logitDimension, totalCount: logitDimension)
        }
    }

    @Test(arguments: [DType.bfloat16, .float16, .float32])
    func greedyPicksTheLowestTiedAllowedToken(dtype: DType) {
        let token = GuidedGenerationLoop.applyMaskAndSample(
            logits: Self.logits(dtype: dtype), maskArray: Self.mask(),
            closingBias: MLXArray.zeros([Self.logitDimension]))
        #expect(Int(token) == Self.tiedTokens.min())
    }
}
