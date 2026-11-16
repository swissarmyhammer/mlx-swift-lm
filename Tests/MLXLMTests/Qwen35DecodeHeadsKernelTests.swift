// Copyright © 2026 Apple Inc.
//
// Parity tests for ``Qwen35DecodeHeadsKernel``.
//
// The kernel replaces the query and key RMS norms, the M-RoPE and the head
// layout of one Qwen 3.5 decode token. It must give the SAME BITS as the MLX
// operations it replaces: one unit in the last place of one key reaches every
// later token through the cache, thus a tolerance would hide a real
// difference. Every parity test below compares the bits.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite(.serialized)
struct Qwen35DecodeHeadsKernelTests {

    // MARK: - The shapes

    /// The query heads of the test layout: two for each KV head.
    private static let queryHeads = 4

    /// The KV heads of the test layout.
    private static let kvHeads = 2

    /// The head width of the Qwen3.8 checkpoints.
    private static let headDim = 256

    /// The rotated numbers of a head: the `partial_rotary_factor` of 0.25 of
    /// the Qwen3.8 checkpoints.
    private static let rotaryDim = 64

    /// The query and the output gate of each head of `q_proj`.
    private static let gatedQueryParts = 2

    /// The M-RoPE sections of the Qwen3.8 checkpoints.
    private static let mropeSection = [11, 11, 10]

    /// The `rope_theta` of the Qwen3.8 checkpoints.
    private static let ropeBase: Float = 10_000_000

    /// The M-RoPE axes of a position: time, height and width.
    private static let positionAxes = 3

    /// A head width the kernel does not read: not a whole number of SIMD
    /// groups of norm reads.
    private static let otherHeadDim = 192

    /// A rotary width the kernel does not read: wider than one SIMD group of
    /// norm reads.
    private static let wideRotaryDim = 256

    /// The epsilon of the query and key norms, the `rms_norm_eps` of Qwen3.8.
    private static let normEps: Float = 1e-6

    /// The dtypes of a decode step: the float32 of the tests, the bfloat16 of
    /// the checkpoints, and float16.
    private static let dtypes: [DType] = [.float32, .bfloat16, .float16]

    /// The M-RoPE positions of the decode token, as time, height and width:
    /// the first one, a small one, a long context, and a token after an
    /// image, whose three axes differ.
    private static let positions: [[Int32]] = [
        [0, 0, 0], [1, 1, 1], [20_000, 20_000, 20_000], [7, 30, 41],
    ]

    /// One sequence, as a decode step has, and two.
    private static let batchSizes = [1, 2]

    /// The tokens of a decode step.
    private static let decodeTokens = 1

    /// The tokens of a block the kernel does not read.
    private static let verifyTokens = 2

    /// The spread of the random norm weights around one.
    private static let normWeightSpread: Float = 0.25

    /// The spread of the random projection values. A wide spread gives the
    /// norms and the rotation large values to round.
    private static let projectionSpread: Float = 4

    /// The axis order that swaps the token axis and the head axis.
    private static let headMajor = [0, 2, 1, 3]

    /// The seeds of the random arrays, one for each role.
    private enum Seed {
        static let queries: UInt64 = 41
        static let keys: UInt64 = 42
        static let queryNorm: UInt64 = 43
        static let keyNorm: UInt64 = 44
        static let layerWeights: UInt64 = 45
        static let hidden: UInt64 = 46
    }

    /// One parity case of the kernel.
    struct HeadsCase: CustomTestStringConvertible, Sendable {
        /// The dtype of the projections, of the norm weights and of the
        /// rotation.
        let dtype: DType
        /// The M-RoPE position of the decode token.
        let position: [Int32]
        /// The number of sequences.
        let batchSize: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(dtype), position \(position), batch \(batchSize)"
        }
    }

    /// Each dtype with each position and batch size.
    static let headsCases = dtypes.flatMap { dtype in
        positions.flatMap { position in
            batchSizes.map { HeadsCase(dtype: dtype, position: position, batchSize: $0) }
        }
    }

    // MARK: - Helpers

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(
        _ shape: [Int], seed: UInt64, spread: Float = 1, dtype: DType = .float32
    ) -> MLXArray {
        (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * spread).asType(dtype)
    }

    /// The `q_proj` output of `tokens` tokens of `batchSize` sequences: the
    /// query and the output gate of each head, one after the other.
    private static func queryProjection(
        batchSize: Int, tokens: Int = decodeTokens, dtype: DType
    ) -> MLXArray {
        randomArray(
            [batchSize, tokens, queryHeads * gatedQueryParts * headDim], seed: Seed.queries,
            spread: projectionSpread, dtype: dtype)
    }

    /// The `k_proj` output of `tokens` tokens of `batchSize` sequences.
    private static func keyProjection(
        batchSize: Int, tokens: Int = decodeTokens, dtype: DType
    ) -> MLXArray {
        randomArray(
            [batchSize, tokens, kvHeads * headDim], seed: Seed.keys, spread: projectionSpread,
            dtype: dtype)
    }

    /// An RMS norm over `headDim` numbers whose weights are near one.
    private static func norm(seed: UInt64, dtype: DType) throws -> RMSNorm {
        let norm = RMSNorm(dimensions: headDim, eps: normEps)
        let weight = randomArray([headDim], seed: seed, spread: normWeightSpread) + 1
        try norm.update(
            parameters: ModuleParameters.unflattened(["weight": weight.asType(dtype)]),
            verify: [.all])
        return norm
    }

    /// The M-RoPE rotation of one token at `position` in each of `batchSize`
    /// sequences, in `dtype`.
    private static func rotation(
        position: [Int32], batchSize: Int, tokens: Int = decodeTokens, dtype: DType
    ) -> Qwen35Language.RotaryValues {
        let embedding = Qwen35Language.RotaryEmbedding(
            dim: rotaryDim, base: ropeBase, mropeSection: mropeSection)
        let positionIds = broadcast(
            MLXArray(position).reshaped(positionAxes, 1, 1),
            to: [positionAxes, batchSize, tokens])
        let (cosines, sines) = embedding(
            x: MLXArray.zeros([1], dtype: dtype), positionIds: positionIds)
        return (cosines, sines)
    }

    /// The kernel of the test layout.
    private static func kernel() throws -> Qwen35DecodeHeadsKernel {
        try #require(
            Qwen35DecodeHeadsKernel(
                queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim, rotaryDim: rotaryDim,
                queryEps: normEps, keyEps: normEps))
    }

    /// The heads of the projections through the MLX operations the kernel
    /// replaces: the split of the queries from their gates, the RMS norms,
    /// the swap of the token axis and the head axis, and the M-RoPE.
    private static func opPathHeads(
        queryProjection: MLXArray, keyProjection: MLXArray, queryNorm: RMSNorm,
        keyNorm: RMSNorm, rotation: Qwen35Language.RotaryValues
    ) -> (queries: MLXArray, keys: MLXArray) {
        let (batchSize, tokens) = (queryProjection.dim(0), queryProjection.dim(1))
        let queryHalves = queryProjection.reshaped(batchSize, tokens, queryHeads, -1)
            .split(parts: gatedQueryParts, axis: -1)
        let queries = queryNorm(queryHalves[0]).transposed(axes: headMajor)
        let keys = keyNorm(keyProjection.reshaped(batchSize, tokens, kvHeads, -1))
            .transposed(axes: headMajor)
        return Qwen35Language.applyMultimodalRotaryPosEmb(
            q: queries, k: keys, cos: rotation.cos, sin: rotation.sin)
    }

    /// Checks that two arrays hold the same dtype, the same shape and the
    /// same bits.
    private func expectSameBits(
        _ kernel: MLXArray, _ reference: MLXArray, _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(kernel.dtype == reference.dtype, "\(what): dtype", sourceLocation: sourceLocation)
        #expect(kernel.shape == reference.shape, "\(what): shape", sourceLocation: sourceLocation)
        let kernelValues = DeepSeekV4TestArrays.floats(kernel)
        let referenceValues = DeepSeekV4TestArrays.floats(reference)
        let firstGap = zip(kernelValues, referenceValues).enumerated().first {
            $0.element.0.bitPattern != $0.element.1.bitPattern
        }
        #expect(
            firstGap == nil,
            """
            \(what)[\(firstGap?.offset ?? 0)]: kernel \(firstGap?.element.0 ?? 0), \
            op path \(firstGap?.element.1 ?? 0)
            """,
            sourceLocation: sourceLocation)
    }

    // MARK: - The kernel

    @Test(arguments: headsCases)
    func theKernelGivesTheBitsOfTheOpPath(headsCase: HeadsCase) throws {
        let (batchSize, dtype) = (headsCase.batchSize, headsCase.dtype)
        let queryProjection = Self.queryProjection(batchSize: batchSize, dtype: dtype)
        let keyProjection = Self.keyProjection(batchSize: batchSize, dtype: dtype)
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: dtype)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: dtype)
        let rotation = Self.rotation(
            position: headsCase.position, batchSize: batchSize, dtype: dtype)

        let reference = Self.opPathHeads(
            queryProjection: queryProjection, keyProjection: keyProjection,
            queryNorm: queryNorm, keyNorm: keyNorm, rotation: rotation)
        let heads = try #require(
            Self.kernel().heads(
                queryProjection: queryProjection, keyProjection: keyProjection,
                queryWeight: queryNorm.weight, keyWeight: keyNorm.weight, rotation: rotation))

        expectSameBits(heads.queries, reference.queries, "queries")
        expectSameBits(heads.keys, reference.keys, "keys")
    }

    @Test func aBlockOfSeveralTokensTakesNoKernel() throws {
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: .float32)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: .float32)

        let heads = try Self.kernel().heads(
            queryProjection: Self.queryProjection(
                batchSize: 1, tokens: Self.verifyTokens, dtype: .float32),
            keyProjection: Self.keyProjection(
                batchSize: 1, tokens: Self.verifyTokens, dtype: .float32),
            queryWeight: queryNorm.weight, keyWeight: keyNorm.weight,
            rotation: Self.rotation(
                position: Self.positions[0], batchSize: 1, tokens: Self.verifyTokens,
                dtype: .float32))

        #expect(heads == nil)
    }

    @Test func aHeadOfAnotherWidthHasNoKernel() {
        let kernel = Qwen35DecodeHeadsKernel(
            queryHeads: Self.queryHeads, kvHeads: Self.kvHeads, headDim: Self.otherHeadDim,
            rotaryDim: Self.rotaryDim, queryEps: Self.normEps, keyEps: Self.normEps)

        #expect(kernel == nil)
    }

    @Test func aRotationWiderThanOneSimdGroupHasNoKernel() {
        let kernel = Qwen35DecodeHeadsKernel(
            queryHeads: Self.queryHeads, kvHeads: Self.kvHeads, headDim: Self.headDim,
            rotaryDim: Self.wideRotaryDim, queryEps: Self.normEps, keyEps: Self.normEps)

        #expect(kernel == nil)
    }

    @Test func aNormWeightOfAnotherDtypeTakesNoKernel() throws {
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: .float32)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: .bfloat16)

        let heads = try Self.kernel().heads(
            queryProjection: Self.queryProjection(batchSize: 1, dtype: .bfloat16),
            keyProjection: Self.keyProjection(batchSize: 1, dtype: .bfloat16),
            queryWeight: queryNorm.weight, keyWeight: keyNorm.weight,
            rotation: Self.rotation(position: Self.positions[0], batchSize: 1, dtype: .bfloat16))

        #expect(heads == nil)
    }

    // MARK: - The layer

    /// The width of the residual stream of the test layer.
    private static let hiddenSize = 64

    /// A Qwen 3.5 text configuration with the head shapes of Qwen3.8 and
    /// one full-attention layer.
    private static let layerJSON = """
        {
            "model_type": "qwen3_5_text", "hidden_size": \(hiddenSize),
            "num_hidden_layers": 2, "intermediate_size": 64,
            "num_attention_heads": \(queryHeads), "num_key_value_heads": \(kvHeads),
            "head_dim": \(headDim), "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "rms_norm_eps": \(normEps), "vocab_size": 32,
            "full_attention_interval": 2,
            "rope_parameters": {
                "type": "default", "mrope_section": \(mropeSection),
                "rope_theta": \(ropeBase), "partial_rotary_factor": 0.25
            }
        }
        """

    /// The attention layer of ``layerJSON`` with random weights in `dtype`.
    ///
    /// The weights draw from a random state of their own, so that building
    /// the layer leaves the global random state of the bundle as it was for
    /// every other suite.
    private static func attention(dtype: DType) throws -> Qwen35Language.Attention {
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(layerJSON.utf8))
        let attention = withRandomState(MLXRandom.RandomState(seed: Seed.layerWeights)) {
            Qwen35Language.Attention(configuration)
        }
        let parameters = attention.parameters().flattened().map { key, value in
            let weight = key.hasSuffix("norm.weight") ? value + normWeightSpread : value
            return (key, weight.asType(dtype))
        }
        try attention.update(
            parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(attention)
        return attention
    }

    /// The pre-norm projections of one random decode token of `attention`,
    /// with the rotation of the token at the long-context position.
    private static func decodeStep(
        _ attention: Qwen35Language.Attention, dtype: DType
    ) -> (
        projection: (queries: MLXArray, gate: MLXArray, keys: MLXArray, values: MLXArray),
        rotation: Qwen35Language.RotaryValues
    ) {
        let x = randomArray([1, decodeTokens, hiddenSize], seed: Seed.hidden).asType(dtype)
        let projection = attention.projectPreNorm(x)
        let positionIds = broadcast(
            MLXArray(positions[2]).reshaped(positionAxes, 1, 1), to: [positionAxes, 1, 1])
        let rotation = attention.rotaryEmbedding(x: projection.values, positionIds: positionIds)
        return (projection, rotation)
    }

    @Test(arguments: dtypes)
    func aDecodeStepOfTheLayerGivesTheBitsOfTheOpPath(dtype: DType) throws {
        let attention = try Self.attention(dtype: dtype)
        let step = Self.decodeStep(attention, dtype: dtype)

        let normed = attention.normedHeads(
            queries: step.projection.queries, keys: step.projection.keys)
        let reference = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: normed.queries, k: normed.keys, cos: step.rotation.cos, sin: step.rotation.sin)
        let heads = attention.rotatedHeads(
            queries: step.projection.queries, keys: step.projection.keys,
            rotation: step.rotation)

        expectSameBits(heads.queries, reference.0, "queries")
        expectSameBits(heads.keys, reference.1, "keys")
    }

    @Test func aDecodeStepOfTheLayerReadsTheKernel() throws {
        let attention = try Self.attention(dtype: .bfloat16)
        let step = Self.decodeStep(attention, dtype: .bfloat16)

        let heads = attention.rotatedHeads(
            queries: step.projection.queries, keys: step.projection.keys,
            rotation: step.rotation)

        #expect(
            DeepSeekV4TestArrays.nodeCount([heads.queries, heads.keys], primitive: "RMSNorm")
                == 0)
    }

    // MARK: - The model

    /// The decoder layers of ``modelJSON``: a gated delta net layer, then a
    /// full-attention layer, twice.
    private static let modelLayers = 4

    /// A Qwen 3.5 text configuration with two full-attention layers.
    private static let modelJSON = """
        {
            "model_type": "qwen3_5_text", "hidden_size": 32,
            "num_hidden_layers": \(modelLayers), "intermediate_size": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "rms_norm_eps": \(normEps), "vocab_size": 32,
            "full_attention_interval": 2,
            "rope_parameters": {
                "type": "default", "mrope_section": [8, 4, 4],
                "rope_theta": 100000.0, "partial_rotary_factor": 1.0
            }
        }
        """

    /// The dtype of the model test: a dtype the compiled decode holds
    /// exactly.
    private static let modelDType = DType.float16

    /// The token ids of the model test: a first token, which takes the
    /// general path and leaves the states of the gated delta net layers,
    /// then a token the compiled decode runs.
    private static let modelTokens: [Int32] = [3, 9]

    @Test func aCompiledDecodeStepComputesTheRotationOnce() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self, from: Data(Self.modelJSON.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: Seed.layerWeights)) {
            Qwen35Language.Model(configuration)
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(Self.modelDType) })
        model.train(false)
        let cache: [KVCache?] = (0 ..< Self.modelLayers).map {
            $0.isMultiple(of: 2) ? MambaCache() : KVCacheSimple()
        }

        /// The hidden states of token `position` of ``modelTokens``.
        func step(_ position: Int) -> MLXArray {
            model(
                MLXArray([Self.modelTokens[position]]).reshaped(1, 1), cache: cache,
                positionIds: broadcast(
                    MLXArray(Int32(position)).reshaped(1, 1, 1), to: [Self.positionAxes, 1, 1]))
        }
        eval([step(0)] + cache.compactMap { $0 }.flatMap { $0.innerState() })

        let decoded = step(1)

        #expect(DeepSeekV4TestArrays.nodeCount([decoded], primitive: "Cos") == 1)
    }
}
