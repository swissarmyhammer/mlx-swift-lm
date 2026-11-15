// Copyright © 2026 Apple Inc.
//
// Parity tests for ``FlintDecodeHeadsKernel``.
//
// The kernel replaces the query and key RMS norms, the RoPE and the head
// layout of one Flint decode token. It must give the SAME BITS as the MLX
// operations it replaces: one unit in the last place of one key reaches every
// later token through the cache, thus a tolerance would hide a real
// difference. Every parity test below compares the bits.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct FlintDecodeHeadsKernelTests {

    // MARK: - The shapes

    /// The query heads of the test layout: two for each KV head.
    private static let queryHeads = 4

    /// The KV heads of the test layout.
    private static let kvHeads = 2

    /// The head width the kernel reads: the width of the Flint checkpoints.
    private static let headDim = 128

    /// The KV projections of a fused `qkv_proj`: the keys and the values.
    private static let kvProjections = 2

    /// The axis of a `[B, L, heads, headDim]` projection that holds the heads.
    private static let headAxis = 2

    /// The index of the value heads among the split parts of `qkv_proj`.
    private static let valuesPart = 2

    /// The axis order that swaps the token axis and the head axis.
    private static let headMajor = [0, headAxis, 1, 3]

    /// A head width the kernel does not read.
    private static let otherHeadDim = 64

    /// The epsilon of the query and key norms, the `rms_norm_eps` of Flint.
    private static let normEps: Float = 1e-6

    /// The `rope_theta` of Flint.
    private static let ropeBase: Float = 10_000

    /// The position scale of a RoPE without `rope_scaling`.
    private static let unscaledPositions: Float = 1

    /// The dtypes of a decode step: the float32 of the tests, the bfloat16 of
    /// the checkpoints, and float16.
    private static let dtypes: [DType] = [.float32, .bfloat16, .float16]

    /// The rotations of a Flint layer: none, for a layer of
    /// `no_rope_layer_indices`, and the plain RoPE of every other layer.
    private static let rotations: [FlintHeadRotation] = [
        .unrotated, .rotary(base: ropeBase, scale: unscaledPositions),
    ]

    /// The positions of the decode token: the first one, a small one, one
    /// past a sliding window of 513 keys, and a long context.
    private static let offsets = [0, 1, 514, 20_000]

    /// One sequence, as a decode step has, and two.
    private static let batchSizes = [1, 2]

    /// The tokens of a decode step.
    private static let decodeTokens = 1

    /// The tokens of a block the kernel does not read.
    private static let prefillTokens = 2

    /// The spread of the random norm weights around one.
    private static let normWeightSpread: Float = 0.25

    /// The spread of the random projection values. A wide spread gives the
    /// norms and the rotation large values to round.
    private static let projectionSpread: Float = 4

    /// The seeds of the random arrays, one for each role.
    private enum Seed {
        static let projection: UInt64 = 31
        static let queryNorm: UInt64 = 32
        static let keyNorm: UInt64 = 33
        static let layerWeights: UInt64 = 34
        static let hidden: UInt64 = 35
    }

    /// One parity case of the kernel.
    struct HeadsCase: CustomTestStringConvertible, Sendable {
        /// The dtype of the projection and of the norm weights.
        let dtype: DType
        /// The rotation of the layer.
        let rotation: FlintHeadRotation
        /// The position of the decode token.
        let offset: Int
        /// The number of sequences.
        let batchSize: Int

        /// The text the test report shows for this case.
        var testDescription: String {
            "\(dtype), \(rotation), offset \(offset), batch \(batchSize)"
        }
    }

    /// Each dtype with each rotation, offset and batch size.
    static let headsCases = dtypes.flatMap { dtype in
        rotations.flatMap { rotation in
            offsets.flatMap { offset in
                batchSizes.map {
                    HeadsCase(dtype: dtype, rotation: rotation, offset: offset, batchSize: $0)
                }
            }
        }
    }

    // MARK: - Helpers

    /// Builds a repeatable random array of the given shape.
    private static func randomArray(
        _ shape: [Int], seed: UInt64, spread: Float = 1, dtype: DType = .float32
    ) -> MLXArray {
        (MLXRandom.normal(shape, key: MLXRandom.key(seed)) * spread).asType(dtype)
    }

    /// The fused query, key and value projection of `tokens` tokens of
    /// `batchSize` sequences, in the layout of `qkv_proj`.
    private static func projection(
        batchSize: Int, tokens: Int = decodeTokens, dtype: DType
    ) -> MLXArray {
        let width = (queryHeads + kvProjections * kvHeads) * headDim
        return randomArray(
            [batchSize, tokens, width], seed: Seed.projection, spread: projectionSpread,
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

    /// The kernel of the test layout with heads of ``headDim`` numbers.
    private static func kernel(rotation: FlintHeadRotation) throws -> FlintDecodeHeadsKernel {
        try #require(
            FlintDecodeHeadsKernel(
                queryHeads: queryHeads, kvHeads: kvHeads, headDim: headDim, queryEps: normEps,
                keyEps: normEps, rotation: rotation))
    }

    /// The heads of `projection` through the MLX operations the kernel
    /// replaces: the split into heads, the RMS norms, the swap of the token
    /// axis and the head axis, and the RoPE.
    private static func opPathHeads(
        _ projection: MLXArray, queryNorm: RMSNorm, keyNorm: RMSNorm,
        rotation: FlintHeadRotation, offset: Int
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let (batchSize, tokens) = (projection.dim(0), projection.dim(1))
        let heads = projection.reshaped(batchSize, tokens, -1, headDim)
        let parts = split(heads, indices: [queryHeads, queryHeads + kvHeads], axis: headAxis)
        let (queryPart, keyPart, valuePart) = (parts[0], parts[1], parts[valuesPart])
        var queries = queryNorm(queryPart).transposed(axes: headMajor)
        var keys = keyNorm(keyPart).transposed(axes: headMajor)
        if case .rotary(let base, let scale) = rotation {
            let rope = RoPE(dimensions: headDim, traditional: false, base: base, scale: scale)
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }
        return (queries, keys, valuePart.transposed(axes: headMajor))
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
        let projection = Self.projection(batchSize: headsCase.batchSize, dtype: headsCase.dtype)
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: headsCase.dtype)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: headsCase.dtype)

        let reference = Self.opPathHeads(
            projection, queryNorm: queryNorm, keyNorm: keyNorm, rotation: headsCase.rotation,
            offset: headsCase.offset)
        let heads = try #require(
            Self.kernel(rotation: headsCase.rotation).heads(
                of: projection, queryWeight: queryNorm.weight, keyWeight: keyNorm.weight,
                offset: headsCase.offset))

        expectSameBits(heads.queries, reference.queries, "queries")
        expectSameBits(heads.keys, reference.keys, "keys")
        expectSameBits(heads.values, reference.values, "values")
    }

    @Test func aBlockOfSeveralTokensTakesNoKernel() throws {
        let projection = Self.projection(
            batchSize: 1, tokens: Self.prefillTokens, dtype: .float32)
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: .float32)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: .float32)

        let heads = try Self.kernel(rotation: .unrotated).heads(
            of: projection, queryWeight: queryNorm.weight, keyWeight: keyNorm.weight, offset: 0)

        #expect(heads == nil)
    }

    @Test func aHeadOfAnotherWidthHasNoKernel() {
        let kernel = FlintDecodeHeadsKernel(
            queryHeads: Self.queryHeads, kvHeads: Self.kvHeads, headDim: Self.otherHeadDim,
            queryEps: Self.normEps, keyEps: Self.normEps, rotation: .unrotated)

        #expect(kernel == nil)
    }

    @Test func aNormWeightOfAnotherDtypeTakesNoKernel() throws {
        let projection = Self.projection(batchSize: 1, dtype: .bfloat16)
        let queryNorm = try Self.norm(seed: Seed.queryNorm, dtype: .float32)
        let keyNorm = try Self.norm(seed: Seed.keyNorm, dtype: .bfloat16)

        let heads = try Self.kernel(rotation: .unrotated).heads(
            of: projection, queryWeight: queryNorm.weight, keyWeight: keyNorm.weight, offset: 0)

        #expect(heads == nil)
    }

    // MARK: - The layer

    /// The width of the residual stream of the test layer.
    private static let hiddenSize = 64

    /// The index of the layer with RoPE in ``layerJSON(ropeScaling:)``.
    private static let rotaryLayer = 0

    /// The index of the layer without RoPE in ``layerJSON(ropeScaling:)``.
    private static let unrotatedLayer = 1

    /// The two layers of ``layerJSON(ropeScaling:)``.
    static let layerIndices = [rotaryLayer, unrotatedLayer]

    /// The position of the decode token of the layer tests.
    private static let layerOffset = 600

    /// A `rope_scaling` entry the kernel does not reproduce.
    private static let linearScaling = #", "rope_scaling": {"type": "linear", "factor": 2.0}"#

    /// A two-layer Flint configuration with heads of ``headDim`` numbers;
    /// layer 1 has no RoPE.
    private static func layerJSON(ropeScaling: String = "") -> String {
        """
        {
            "model_type": "flint", "hidden_size": \(hiddenSize), "num_hidden_layers": 2,
            "intermediate_size": 32, "num_attention_heads": \(queryHeads),
            "num_key_value_heads": \(kvHeads), "head_dim": \(headDim), "vocab_size": 64,
            "rms_norm_eps": \(normEps), "rope_theta": \(ropeBase),
            "layer_types": ["sliding_attention", "full_attention"],
            "no_rope_layer_indices": [\(unrotatedLayer)], "sliding_window": 4,
            "softmax_type": "learnable", "mlp_only_layers": [0, 1], "num_experts": 8,
            "num_experts_per_tok": 2, "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 16, "router_score_function": "sigmoid",
            "hidden_act": "silu"\(ropeScaling)
        }
        """
    }

    /// The attention layer `index` of ``layerJSON(ropeScaling:)``.
    private static func attention(index: Int, ropeScaling: String = "") throws -> FlintAttention {
        let configuration = try JSONDecoder().decode(
            FlintConfiguration.self, from: Data(layerJSON(ropeScaling: ropeScaling).utf8))
        return FlintAttention(configuration, layerIdx: index)
    }

    /// The fused attention of layer `index` with random weights in `dtype`.
    ///
    /// The weights draw from a random state of their own, so that building
    /// the layer leaves the global random state of the bundle as it was for
    /// every other suite.
    private static func fusedAttention(index: Int, dtype: DType) throws -> FlintFusedAttention {
        let fused = try withRandomState(MLXRandom.RandomState(seed: Seed.layerWeights)) {
            FlintFusedAttention(fusing: try attention(index: index))
        }
        let parameters = fused.parameters().flattened().map { key, value in
            let weight = key.hasSuffix("norm.weight") ? value + normWeightSpread : value
            return (key, weight.asType(dtype))
        }
        try fused.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(fused)
        return fused
    }

    @Test(arguments: dtypes, layerIndices)
    func aDecodeStepOfAFusedLayerGivesTheBitsOfTheOpPath(dtype: DType, index: Int) throws {
        let layer = try Self.fusedAttention(index: index, dtype: dtype)
        let x = Self.randomArray([1, Self.decodeTokens, Self.hiddenSize], seed: Seed.hidden)
            .asType(dtype)
        let offset = RoPEOffset.scalar(Self.layerOffset)

        let reference = layer.rotatedHeads(layer.projectHeads(x), offset: offset)
        let heads = layer.attentionInputs(x, offset: offset)

        expectSameBits(heads.queries, reference.queries, "queries")
        expectSameBits(heads.keys, reference.keys, "keys")
        expectSameBits(heads.values, reference.values, "values")
    }

    @Test(arguments: layerIndices)
    func aDecodeStepOfAFusedLayerReadsTheKernel(index: Int) throws {
        let layer = try Self.fusedAttention(index: index, dtype: .bfloat16)
        let x = Self.randomArray([1, Self.decodeTokens, Self.hiddenSize], seed: Seed.hidden)
            .asType(.bfloat16)

        let heads = layer.attentionInputs(x, offset: .scalar(Self.layerOffset))

        #expect(
            DeepSeekV4TestArrays.nodeCount(
                [heads.queries, heads.keys, heads.values], primitive: "RMSNorm") == 0)
    }

    @Test func aLayerWithPlainRoPERotatesItsHeadsWithTheBaseOfTheCheckpoint() throws {
        let layer = try Self.attention(index: Self.rotaryLayer)

        #expect(layer.headRotation == .rotary(base: Self.ropeBase, scale: Self.unscaledPositions))
    }

    @Test func aLayerWithoutRoPELeavesItsHeadsUnrotated() throws {
        let layer = try Self.attention(index: Self.unrotatedLayer)

        #expect(layer.headRotation == .unrotated)
    }

    @Test func aLayerWithScaledRoPEKeepsTheOpPath() throws {
        let layer = try Self.attention(index: Self.rotaryLayer, ropeScaling: Self.linearScaling)

        #expect(layer.headRotation == nil)
    }
}
