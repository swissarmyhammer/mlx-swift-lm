// Copyright © 2026 Apple Inc.
//
// Parity tests for the one-call rotary position of DeepSeek-V4 (task
// `^phf8pwt`).
//
// `DeepSeekV4RoPE` turns a whole head with one `MLXFast.RoPE` call. The call
// reads a table of periods: a pair that takes no position has an infinite
// period, thus it turns by zero. Before that task the port built a cosine
// table and a sine table, cut the head in two, turned the trailing half with
// separate MLX operations, and joined the two halves again. ``OpPathRoPE``
// below is that op path, copied without change, and each test compares the
// one-call path against it.
//
// The two paths cannot give the same bits. The `MLXFast.RoPE` kernel computes
// the reciprocal of each period, reads `metal::fast::cos` and
// `metal::fast::sin`, turns the pair in float32, and rounds to the dtype of the
// input one time. The op path reads the precise `cos` and `sin` of MLX, rounds
// both to the dtype of the input, and rounds again after each product. The
// tolerances below hold the measured gap with a margin; the comment on each
// one gives the measurement.
//
// The dimensions that take no position must come back with the SAME BITS: an
// infinite period gives an angle of zero, thus a cosine of 1 and a sine of 0.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4RoPEParityTests {

    // MARK: - The checkpoint shape

    /// The width of one attention head of the DeepSeek-V4-Flash checkpoint.
    private static let headDim = 512

    /// The width of one indexer head of the DeepSeek-V4-Flash checkpoint.
    private static let indexHeadDim = 128

    /// The number of trailing head dimensions that take a position.
    private static let ropeDim = DeepSeekV4FlashYarn.ropeDim

    /// The `rope_theta` of the DeepSeek-V4-Flash checkpoint.
    private static let ropeTheta: Float = 10_000

    /// A YaRN factor that leaves every frequency alone.
    private static let noScaling: Float = 1

    /// A layer whose compress ratio is 0, thus a layer that reads
    /// `rope_theta` and no YaRN scaling.
    private static let plainLayer = 0

    /// A layer whose compress ratio is 4, thus a layer that holds an indexer
    /// and reads `compress_rope_theta` with the YaRN scaling.
    private static let overlapLayer = 2

    /// A layer whose compress ratio is 128, thus a layer that reads
    /// `compress_rope_theta` with the YaRN scaling and holds no indexer.
    private static let wideLayer = 3

    /// A checkpoint of four layers with the real head widths, the real rope
    /// thetas and the real YaRN block.
    private static func configuration() throws -> DeepSeekV4Configuration {
        let yarn = DeepSeekV4FlashYarn.self
        let json = """
            {
              "num_hidden_layers": 4,
              "head_dim": \(headDim),
              "qk_rope_head_dim": \(ropeDim),
              "index_head_dim": \(indexHeadDim),
              "rope_theta": \(ropeTheta),
              "compress_rope_theta": \(yarn.compressRopeTheta),
              "rope_scaling": {
                "type": "yarn",
                "factor": \(yarn.factor),
                "original_max_position_embeddings": \(yarn.originalMaxPositionEmbeddings),
                "beta_fast": \(yarn.betaFast),
                "beta_slow": \(yarn.betaSlow)
              },
              "compress_ratios": [0, 0, 4, 128]
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    /// The inverse-frequency table of a layer that reads `rope_theta`.
    private static func plainInverseFrequency() -> MLXArray {
        DeepSeekV4Math.yarnInvFreq(
            dim: ropeDim, base: ropeTheta,
            originalMaxPositionEmbeddings: DeepSeekV4FlashYarn.originalMaxPositionEmbeddings,
            factor: noScaling, betaFast: DeepSeekV4FlashYarn.betaFast,
            betaSlow: DeepSeekV4FlashYarn.betaSlow)
    }

    // MARK: - The inputs

    /// The number of sequences in each block.
    private static let batchSize = 1

    /// The number of heads each block holds.
    private static let headCount = 2

    /// The number of tokens in each block.
    private static let tokenCount = 3

    /// The number of pooled chunks the pooled-row test turns.
    private static let chunkCount = 5

    /// The position of the first token of each block. The last two stand at
    /// the long contexts the decode benchmark reads, where the angle of the
    /// fastest pair passes 4000 and 32000 radians.
    private static let offsets = [0, 9, 4_093, 32_765]

    /// The position of the first pooled chunk of each pooled block. Each one
    /// is a multiple of every compress ratio, as a chunk start must be.
    private static let firstChunkPositions = [0, 4_096, 32_768]

    /// The dtypes each test reads: float32, and the bfloat16 the published
    /// checkpoint computes in.
    private static let dtypes: [DType] = [.float32, .bfloat16]

    /// The seed of the random head values.
    private static let inputSeed: UInt64 = 20_261_013

    /// A repeatable random array of standard-normal values.
    private static func randomArray(_ shape: [Int], dtype: DType) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(inputSeed)).asType(dtype)
    }

    /// A run of positions.
    private static func positions(from first: Int, count: Int, stride: Int = 1) -> MLXArray {
        MLXArray(Int32(0) ..< Int32(count)) * Int32(stride) + Int32(first)
    }

    // MARK: - Tolerances

    /// The largest gap allowed on a float32 rotary dimension: twice the
    /// largest gap measured.
    ///
    /// Measured on the inputs below: 2.4e-7 at the offsets 0 and 9 on every
    /// layer, 2.0e-5 at offset 4093 and 1.6e-4 at offset 32765 on the
    /// `rope_theta` layer, and 4.8e-6 or less at every offset on the
    /// `compress_rope_theta` layers and the pooled rows. The gap grows with
    /// the angle: the fastest pair turns 32765 radians at offset 32765, and
    /// `metal::fast::cos` and `metal::fast::sin` lose accuracy on a large
    /// angle.
    private static let float32Tolerance: Float = 3.2e-4

    /// The largest gap allowed on a bfloat16 rotary dimension: two bfloat16
    /// steps at the magnitudes 2 through 4, where the largest
    /// standard-normal inputs stand.
    ///
    /// Measured on the inputs below: 0.015625, which is one such step, at
    /// every offset, on every layer and on the pooled rows. The op path
    /// rounds the cosine, the sine and each product to bfloat16; the kernel
    /// rounds one time.
    private static let bfloat16Tolerance: Float = 0.03125

    /// The tolerance of one dtype.
    private static func tolerance(_ dtype: DType) -> Float {
        dtype == .bfloat16 ? bfloat16Tolerance : float32Tolerance
    }

    /// Checks one turned block against the op path.
    ///
    /// The leading `width - ropeDim` dimensions must hold the bits of the
    /// input. The trailing `ropeDim` dimensions must stay within the
    /// tolerance of the dtype.
    private func expectParity(
        input: MLXArray, turned: MLXArray, reference: MLXArray, dtype: DType, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        let noPositionDim = input.dim(-1) - Self.ropeDim
        #expect(turned.shape == input.shape, "\(what): shape", sourceLocation: sourceLocation)
        #expect(turned.dtype == input.dtype, "\(what): dtype", sourceLocation: sourceLocation)
        #expect(
            DeepSeekV4TestArrays.floats(turned[.ellipsis, 0 ..< noPositionDim])
                == DeepSeekV4TestArrays.floats(input[.ellipsis, 0 ..< noPositionDim]),
            "\(what): the dimensions with no position must not move",
            sourceLocation: sourceLocation)
        let gaps = zip(
            DeepSeekV4TestArrays.floats(turned[.ellipsis, noPositionDim...]),
            DeepSeekV4TestArrays.floats(reference[.ellipsis, noPositionDim...])
        ).map { abs($0.0 - $0.1) }
        let largestGap = gaps.max() ?? 0
        #expect(
            largestGap <= Self.tolerance(dtype),
            "\(what): largest gap \(largestGap)", sourceLocation: sourceLocation)
    }

    // MARK: - Heads

    /// Turns a block of heads through the layer rope and through the op path,
    /// and checks the two against each other.
    private func checkHeads(
        layer: Int, width: Int, inverse: Bool, inverseFrequency: MLXArray, dtype: DType
    ) throws {
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let input = Self.randomArray(
            [Self.batchSize, Self.headCount, Self.tokenCount, width], dtype: dtype)
        for offset in Self.offsets {
            let turned = rope(input, offset: DeepSeekV4RoPE.position(offset), inverse: inverse)
            let reference = OpPathRoPE.turned(
                input, inverseFrequency: inverseFrequency,
                positions: Self.positions(from: offset, count: Self.tokenCount),
                ropeDim: Self.ropeDim, inverse: inverse)
            expectParity(
                input: input, turned: turned, reference: reference, dtype: dtype,
                "layer \(layer), width \(width), inverse \(inverse), offset \(offset), \(dtype)")
        }
    }

    @Test(arguments: dtypes)
    func theRopeThetaLayerTurnsTheHeadAsTheOpPath(dtype: DType) throws {
        try checkHeads(
            layer: Self.plainLayer, width: Self.headDim, inverse: false,
            inverseFrequency: Self.plainInverseFrequency(), dtype: dtype)
    }

    @Test(arguments: dtypes)
    func theCompressRopeThetaLayerTurnsTheHeadWithYaRNAsTheOpPath(dtype: DType) throws {
        try checkHeads(
            layer: Self.overlapLayer, width: Self.headDim, inverse: false,
            inverseFrequency: DeepSeekV4FlashYarn.inverseFrequency(), dtype: dtype)
    }

    @Test(arguments: dtypes)
    func theIndexerHeadTurnsAsTheOpPath(dtype: DType) throws {
        try checkHeads(
            layer: Self.overlapLayer, width: Self.indexHeadDim, inverse: false,
            inverseFrequency: DeepSeekV4FlashYarn.inverseFrequency(), dtype: dtype)
    }

    @Test(arguments: dtypes)
    func theInverseRotationOfTheRopeThetaLayerTurnsAsTheOpPath(dtype: DType) throws {
        try checkHeads(
            layer: Self.plainLayer, width: Self.headDim, inverse: true,
            inverseFrequency: Self.plainInverseFrequency(), dtype: dtype)
    }

    @Test(arguments: dtypes)
    func theInverseRotationOfTheCompressRopeThetaLayerTurnsAsTheOpPath(dtype: DType) throws {
        try checkHeads(
            layer: Self.overlapLayer, width: Self.headDim, inverse: true,
            inverseFrequency: DeepSeekV4FlashYarn.inverseFrequency(), dtype: dtype)
    }

    // MARK: - Pooled rows

    /// Turns a block of pooled chunks through the layer rope and through the
    /// op path, and checks the two against each other.
    private func checkPooledRows(layer: Int, width: Int, dtype: DType) throws {
        let configuration = try Self.configuration()
        let rope = DeepSeekV4RoPE(configuration: configuration, layer: layer)
        let ratio = configuration.compressRatio(ofLayer: layer)
        let input = Self.randomArray([Self.batchSize, Self.chunkCount, width], dtype: dtype)
        for first in Self.firstChunkPositions {
            let turned = rope.rotatedChunks(input, firstPosition: first)
            let reference = OpPathRoPE.turned(
                input, inverseFrequency: DeepSeekV4FlashYarn.inverseFrequency(),
                positions: Self.positions(from: first, count: Self.chunkCount, stride: ratio),
                ropeDim: Self.ropeDim, inverse: false)
            expectParity(
                input: input, turned: turned, reference: reference, dtype: dtype,
                "pooled rows of layer \(layer), width \(width), first \(first), \(dtype)")
        }
    }

    @Test(arguments: dtypes)
    func thePooledRowsOfTheFineRatioTurnAsTheOpPath(dtype: DType) throws {
        try checkPooledRows(layer: Self.overlapLayer, width: Self.headDim, dtype: dtype)
        try checkPooledRows(layer: Self.overlapLayer, width: Self.indexHeadDim, dtype: dtype)
    }

    @Test(arguments: dtypes)
    func thePooledRowsOfTheCoarseRatioTurnAsTheOpPath(dtype: DType) throws {
        try checkPooledRows(layer: Self.wideLayer, width: Self.headDim, dtype: dtype)
    }
}

/// The rotary position of DeepSeek-V4 as the op path computed it before task
/// `^phf8pwt`: `DeepSeekV4RoPE.cosSin(positions:)` and
/// `DeepSeekV4Math.applyPartialRoPE` / `applyInversePartialRoPE`, copied
/// without change.
private enum OpPathRoPE {

    /// Two dimensions make one rotary pair.
    private static let rotaryPairWidth = 2

    /// Turns the trailing `ropeDim` dimensions of `x` by the angle of each
    /// position, forward or backward.
    ///
    /// - Parameters:
    ///   - x: The head values, shape `(..., positions, width)`.
    ///   - inverseFrequency: The inverse frequency of each pair, shape
    ///     `(ropeDim / 2)`.
    ///   - positions: The position of each row, shape `(positions)`.
    ///   - ropeDim: The number of trailing dimensions that take a position.
    ///   - inverse: True for the backward rotation.
    /// - Returns: An array of the shape of `x`.
    static func turned(
        _ x: MLXArray, inverseFrequency: MLXArray, positions: MLXArray, ropeDim: Int,
        inverse: Bool
    ) -> MLXArray {
        let angles =
            positions.asType(.float32).expandedDimensions(axis: -1)
            * inverseFrequency.expandedDimensions(axis: 0)
        let noPositionDim = x.dim(-1) - ropeDim
        let noPosition = x[.ellipsis, 0 ..< noPositionDim]
        let rotated = rotatePairs(
            x[.ellipsis, noPositionDim...], cos: MLX.cos(angles), sin: MLX.sin(angles),
            inverse: inverse)
        return concatenated([noPosition, rotated], axis: -1)
    }

    /// Turns each adjacent pair of the last axis of `x`.
    private static func rotatePairs(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, inverse: Bool
    ) -> MLXArray {
        let cosine = cos.asType(x.dtype)
        let sine = (inverse ? -sin : sin).asType(x.dtype)
        let pairCount = (x.shape.last ?? 0) / rotaryPairWidth
        let paired = x.reshaped(Array(x.shape.dropLast()) + [pairCount, rotaryPairWidth])
        let real = paired[.ellipsis, 0]
        let imaginary = paired[.ellipsis, 1]
        let turnedReal = real * cosine - imaginary * sine
        let turnedImaginary = real * sine + imaginary * cosine
        return stacked([turnedReal, turnedImaginary], axis: -1).reshaped(x.shape)
    }
}
