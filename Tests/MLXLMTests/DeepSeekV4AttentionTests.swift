// Copyright © 2026 Apple Inc.
//
// Numeric parity tests for `DeepSeekV4RoPE` and `DeepSeekV4Attention`.
//
// Every expected number below comes from the DeepSeek-V4 Python reference,
// `Thump604/mlx-lm` @ `deepseek-v4-support-fixes`,
// `mlx_lm/models/deepseek_v4.py` -- `DeepseekV4RoPE.__init__`,
// `DeepseekV4RoPE.__call__`, `V4Attention._grouped_output_projection` and
// `V4Attention.__call__` -- together with the attention-sink path of mlx
// itself, `mlx/fast.cpp`, which prepends the sink column after the mask and
// before the softmax and then drops column 0.
//
// Each of those routines was transcribed into NumPy one line at a time --
// `mx.*` became `np.*` and nothing else changed -- and run in float64. A
// fixture that a Swift function produced itself proves nothing, thus no
// number below was read out of this repository.
//
// The weights and the inputs are not random. Both the NumPy transcription
// and ``DeepSeekV4AttentionTests/fixtureValues(count:seed:)`` below fill an
// array with `(((i + seed) * 37) % 17 - 8) / 8`, which lands on multiples of
// an eighth and is thus exact in both float32 and float64.
//
// Tolerance. MLX evaluates in float32 and the fixtures are float64. The
// tests allow 1e-4, read as an absolute gap or a relative one, whichever is
// larger. The smallest separation any fixture pair carries is 0.0031 -- the
// gap between the plain-theta layer and the compressed-theta layer -- thus
// the limit stays 30 times below the effect each test measures.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4AttentionTests {

    // MARK: - The synthetic checkpoint

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 8

    /// The number of query heads of the synthetic checkpoint.
    private static let headCount = 2

    /// The width of one head of the synthetic checkpoint.
    private static let headDim = 16

    /// The number of trailing head dimensions that take a position.
    private static let ropeDim = 8

    /// The rank of the low-rank query projection.
    private static let queryLoraRank = 4

    /// The number of head groups the output projection reads.
    private static let outputGroups = 8

    /// The rank of the low-rank output projection of one group.
    private static let outputLoraRank = 3

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 4

    /// The index of a layer whose compress ratio is 0. Layers 0 and 1 hold
    /// no compressor, thus they read the plain `rope_theta`.
    private static let plainLayer = 0

    /// The index of a layer whose compress ratio is more than 0, thus a
    /// layer that reads `compress_rope_theta` with the YaRN scaling.
    private static let compressedLayer = 2

    /// The number of head-group features the output projection reads.
    private static let groupFeatures = (headCount * headDim) / outputGroups

    /// The number of tokens the prefill fixture holds.
    private static let prefillLength = 2

    /// The number of tokens one decode step holds.
    private static let decodeLength = 1

    /// The batch size every fixture uses.
    private static let batchSize = 1

    /// The largest gap allowed against a fixture, absolute or relative.
    private static let tolerance: Float = 1e-4

    /// The two positions of the fixture filler that pick a weight tensor.
    private enum Seed {
        static let queryA = 1
        static let queryNorm = 2
        static let queryB = 3
        static let keyValue = 4
        static let keyValueNorm = 5
        static let attentionSink = 6
        static let outputA = 7
        static let outputB = 8
        static let prefillInput = 11
        static let decodeInput = 23
        static let groupedInput = 31
    }

    /// The stride the fixture filler steps through its cycle with.
    private static let fillerStride = 37

    /// The length of the fixture filler's cycle. It shares no factor with the
    /// stride, thus the filler walks every value of the cycle.
    private static let fillerCycle = 17

    /// The value the filler takes away, which puts the cycle around zero.
    private static let fillerShift = 8

    /// The value the filler divides by. It is a power of two, thus every
    /// fixture value is exact in float32 and in float64 alike.
    private static let fillerDivisor: Float = 8

    /// The filler both this file and the NumPy transcription read.
    private static func fixtureValues(count: Int, seed: Int) -> [Float] {
        (0 ..< count).map { index in
            Float(((index + seed) * fillerStride) % fillerCycle - fillerShift) / fillerDivisor
        }
    }

    /// Builds a fixture tensor of the given shape.
    private static func fixtureArray(_ shape: [Int], seed: Int) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray(fixtureValues(count: count, seed: seed)).reshaped(shape)
    }

    /// The `config.json` of the synthetic checkpoint.
    private static func configuration(useAttnSink: Bool) throws -> DeepSeekV4Configuration {
        let json = """
            {
              "hidden_size": \(hiddenSize),
              "num_attention_heads": \(headCount),
              "num_key_value_heads": 1,
              "head_dim": \(headDim),
              "qk_rope_head_dim": \(ropeDim),
              "q_lora_rank": \(queryLoraRank),
              "o_groups": \(outputGroups),
              "o_lora_rank": \(outputLoraRank),
              "num_hidden_layers": \(layerCount),
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "compress_rope_theta": 160000.0,
              "rope_scaling": {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1
              },
              "compress_ratios": [0, 0, 4, 128],
              "use_attn_sink": \(useAttnSink)
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    /// Builds one attention layer of the synthetic checkpoint and loads the
    /// fixture weights into it.
    private static func attention(layer: Int, useAttnSink: Bool) throws -> DeepSeekV4Attention {
        let block = DeepSeekV4Attention(
            configuration: try configuration(useAttnSink: useAttnSink), layer: layer)
        try block.update(
            parameters: ModuleParameters.unflattened(fixtureWeights()), verify: [])
        return block
    }

    /// The fixture weights, keyed by the checkpoint path of each tensor.
    private static func fixtureWeights() -> [(String, MLXArray)] {
        [
            ("wq_a.weight", fixtureArray([queryLoraRank, hiddenSize], seed: Seed.queryA)),
            ("q_norm.weight", fixtureArray([queryLoraRank], seed: Seed.queryNorm)),
            (
                "wq_b.weight",
                fixtureArray([headCount * headDim, queryLoraRank], seed: Seed.queryB)
            ),
            ("wkv.weight", fixtureArray([headDim, hiddenSize], seed: Seed.keyValue)),
            ("kv_norm.weight", fixtureArray([headDim], seed: Seed.keyValueNorm)),
            ("attn_sink", fixtureArray([headCount], seed: Seed.attentionSink)),
            (
                "wo_a.weight",
                fixtureArray([outputGroups * outputLoraRank, groupFeatures], seed: Seed.outputA)
            ),
            (
                "wo_b.weight",
                fixtureArray([hiddenSize, outputGroups * outputLoraRank], seed: Seed.outputB)
            ),
        ]
    }

    /// The prefill input of every fixture, shape `(1, 2, 8)`.
    private static func prefillInput() -> MLXArray {
        fixtureArray([batchSize, prefillLength, hiddenSize], seed: Seed.prefillInput)
    }

    /// The decode input of every fixture, shape `(1, 1, 8)`.
    private static func decodeInput() -> MLXArray {
        fixtureArray([batchSize, decodeLength, hiddenSize], seed: Seed.decodeInput)
    }

    // MARK: - Comparison helpers

    private func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    private func expectClose(
        _ got: [Float], _ expected: [Float], _ what: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.count == expected.count, "\(what): length", sourceLocation: sourceLocation)
        for (index, pair) in zip(got, expected).enumerated() {
            let limit = max(Self.tolerance, Self.tolerance * abs(pair.1))
            let gap = abs(pair.0 - pair.1)
            #expect(
                gap <= limit,
                "\(what)[\(index)]: got \(pair.0), expected \(pair.1), gap \(gap)",
                sourceLocation: sourceLocation)
        }
    }

    /// The largest gap between two lists of the same length.
    private func largestGap(_ left: [Float], _ right: [Float]) -> Float {
        zip(left, right).map { abs($0 - $1) }.max() ?? 0
    }

    // MARK: - Fixtures from the Python reference

    /// `V4Attention.__call__` on the prefill input, layer 0, sink on.
    private static let prefillWithSink: [Float] = [
        0.3443022714, 0.1420486702, 1.2925117864, 0.2221380062,
        0.3374126713, -0.9054646407, 0.9944063563, -0.4027479155,
        0.5065818309, -0.8722846018, 0.1869354288, 0.5129791708,
        1.2495502693, -1.1775724924, -0.3387707136, 0.0243959002,
    ]

    /// The same call with the sink column left out.
    private static let prefillWithoutSink: [Float] = [
        0.7664525027, 0.1077271035, 1.8466651632, 0.1570390857,
        0.8687932510, -1.5003284883, 1.2242330650, -0.9080281776,
        0.6407177116, -1.0574579782, 0.2444062107, 0.5636049227,
        1.5091403420, -1.3945714669, -0.4299814283, 0.0033588046,
    ]

    /// The same call on a layer whose compress ratio is 4, thus a layer that
    /// reads `compress_rope_theta` with the YaRN scaling.
    private static let prefillCompressedLayer: [Float] = [
        0.3443022714, 0.1420486702, 1.2925117864, 0.2221380062,
        0.3374126713, -0.9054646407, 0.9944063563, -0.4027479155,
        0.5097274883, -0.8652749933, 0.1879118626, 0.5144927165,
        1.2544055709, -1.1943941283, -0.3373106927, 0.0381677144,
    ]

    /// One decode step at offset 2, on the cache the prefill left behind.
    private static let decodeWithSink: [Float] = [
        0.6759217087, -0.3472852914, 1.0644046232, 0.6855393316,
        0.6438894675, -1.4662212985, 0.1170964889, -0.1830741441,
    ]

    /// `V4Attention._grouped_output_projection` on the grouped-input fixture.
    private static let groupedProjection: [Float] = [
        -0.4843750000, -0.5625000000, 0.1562500000, 0.6250000000,
        0.7812500000, -0.9218750000, -0.5781250000, -0.4531250000,
        1.7968750000, 2.0156250000, -0.2812500000, -0.9843750000,
        -0.8906250000, 1.0312500000, 0.5625000000, 0.0000000000,
        -0.7656250000, 0.5937500000, 0.1718750000, 2.0312500000,
        -0.8906250000, 0.1562500000, -0.6718750000, -0.4375000000,
        0.6562500000, -0.4687500000, -0.7968750000, -0.8437500000,
        -0.1406250000, 0.0312500000, 1.7187500000, -0.5312500000,
        -0.6562500000, -0.4218750000, 0.7500000000, 0.5937500000,
        0.1718750000, -0.5468750000, -0.4687500000, 0.3125000000,
        0.0937500000, -0.9218750000, -0.5312500000, -1.0468750000,
        1.0937500000, 1.0937500000, -1.0468750000, -0.5312500000,
    ]

    /// `DeepseekV4RoPE.inv_freq` of a layer with no compressor, base 10000.
    private static let plainInverseFrequency: [Float] = [
        1.0, 0.1, 0.01, 0.001,
    ]

    /// `DeepseekV4RoPE.inv_freq` of a layer with a compressor: base 160000
    /// with the YaRN scaling of `factor` 16.
    private static let yarnInverseFrequency: [Float] = [
        1.0, 0.05, 0.0017187500, 0.0000468750,
    ]

    // MARK: - Forward pass

    @Test
    func prefillMatchesThePythonReference() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let output = block(Self.prefillInput(), mask: .causal, cache: nil)

        #expect(output.shape == [Self.batchSize, Self.prefillLength, Self.hiddenSize])
        expectClose(floats(output), Self.prefillWithSink, "prefill with sink")
    }

    @Test
    func attentionSinkChangesTheOutput() throws {
        let withSink = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let withoutSink = try Self.attention(layer: Self.plainLayer, useAttnSink: false)

        let sinkOutput = floats(withSink(Self.prefillInput(), mask: .causal, cache: nil))
        let plainOutput = floats(withoutSink(Self.prefillInput(), mask: .causal, cache: nil))

        expectClose(plainOutput, Self.prefillWithoutSink, "prefill without sink")
        let sinkEffect = largestGap(sinkOutput, plainOutput)
        #expect(
            sinkEffect > Self.tolerance,
            "the sink must change the output, and the largest gap was \(sinkEffect)")
    }

    @Test
    func compressedLayerReadsTheCompressRopeTheta() throws {
        let block = try Self.attention(layer: Self.compressedLayer, useAttnSink: true)
        let output = floats(block(Self.prefillInput(), mask: .causal, cache: nil))

        expectClose(output, Self.prefillCompressedLayer, "prefill, compressed layer")
        #expect(
            largestGap(output, Self.prefillWithSink) > Self.tolerance,
            "a compressed layer must not read the plain rope theta")
    }

    @Test
    func decodeStepMatchesThePythonReference() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let cache = KVCacheSimple()

        _ = block(Self.prefillInput(), mask: .causal, cache: cache)
        let output = block(Self.decodeInput(), mask: .causal, cache: cache)

        #expect(output.shape == [Self.batchSize, Self.decodeLength, Self.hiddenSize])
        expectClose(floats(output), Self.decodeWithSink, "decode with sink")
        #expect(cache.offset == Self.prefillLength + Self.decodeLength)
    }

    @Test
    func cacheOffsetCountsEveryDecodedToken() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let cache = KVCacheSimple()
        let steps = 5

        for _ in 0 ..< steps {
            let output = block(Self.decodeInput(), mask: .causal, cache: cache)
            #expect(output.shape == [Self.batchSize, Self.decodeLength, Self.hiddenSize])
        }

        #expect(cache.offset == steps)
    }

    @Test
    func attentionReadsTheArraysTheCacheReturned() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let gain: Float = 2
        let plainCache = KVCacheSimple()
        let gainCache = GainKVCache(valueGain: gain)

        let plainOutput = floats(block(Self.prefillInput(), mask: .causal, cache: plainCache))
        let gainOutput = floats(block(Self.prefillInput(), mask: .causal, cache: gainCache))

        // Nothing after the softmax carries a bias, thus the whole block is
        // linear in the values the cache hands back. A block that dropped the
        // return of `update` and fed its own tensor to SDPA would return the
        // plain output instead of the scaled one.
        expectClose(gainOutput, plainOutput.map { $0 * gain }, "values the cache returned")
        #expect(gainCache.offset == Self.prefillLength)
    }

    // MARK: - Grouped low-rank output projection

    @Test
    func groupedOutputProjectionMatchesThePythonReference() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)
        let input = Self.fixtureArray(
            [Self.batchSize, Self.prefillLength, Self.headCount * Self.headDim],
            seed: Seed.groupedInput)

        let projected = block.groupedOutputProjection(input)

        #expect(
            projected.shape == [
                Self.batchSize, Self.prefillLength, Self.outputGroups * Self.outputLoraRank,
            ])
        expectClose(floats(projected), Self.groupedProjection, "grouped output projection")
    }

    @Test
    func groupedOutputProjectionReachesTheFullHiddenSize() throws {
        let block = DeepSeekV4Attention(
            configuration: try Self.wideConfiguration(), layer: Self.plainLayer)

        let prefill = block(
            MLXRandom.normal([Self.batchSize, Self.widePrefillLength, Self.wideHiddenSize]),
            mask: .causal, cache: nil)
        let decode = block(
            MLXRandom.normal([Self.batchSize, Self.decodeLength, Self.wideHiddenSize]),
            mask: .causal, cache: nil)

        #expect(prefill.shape == [Self.batchSize, Self.widePrefillLength, Self.wideHiddenSize])
        #expect(decode.shape == [Self.batchSize, Self.decodeLength, Self.wideHiddenSize])
        #expect(floats(prefill).allSatisfy { $0.isFinite })
        #expect(floats(decode).allSatisfy { $0.isFinite })
    }

    /// The width of the residual stream of the real DeepSeek-V4-Flash
    /// checkpoint, which the grouped projection must reach from 8 groups.
    private static let wideHiddenSize = 4096

    /// The prompt length the wide shape test reads.
    private static let widePrefillLength = 8

    /// A checkpoint that keeps the real `hidden_size` and the real
    /// `o_groups`, and shrinks everything else so the test stays small.
    private static func wideConfiguration() throws -> DeepSeekV4Configuration {
        let json = """
            {
              "hidden_size": \(wideHiddenSize),
              "num_attention_heads": \(headCount),
              "num_key_value_heads": 1,
              "head_dim": \(headDim),
              "qk_rope_head_dim": \(ropeDim),
              "q_lora_rank": \(queryLoraRank),
              "o_groups": \(outputGroups),
              "o_lora_rank": \(outputLoraRank),
              "num_hidden_layers": \(layerCount),
              "compress_ratios": [0, 0, 4, 128]
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Rotary position

    @Test
    func perLayerThetaSplitsThePlainLayerFromTheCompressedLayer() throws {
        let config = try Self.configuration(useAttnSink: true)
        let plain = DeepSeekV4RoPE(configuration: config, layer: Self.plainLayer)
        let compressed = DeepSeekV4RoPE(configuration: config, layer: Self.compressedLayer)
        let position = 1
        let length = 1

        let plainAngles = plain.cosSin(offset: position, length: length)
        let compressedAngles = compressed.cosSin(offset: position, length: length)

        // At position 1 the angle of each pair is its own inverse frequency,
        // thus the cosine table is `cos(inv_freq)` term by term.
        expectClose(
            floats(plainAngles.cos), Self.plainInverseFrequency.map { cos($0) },
            "plain layer cosine")
        expectClose(
            floats(compressedAngles.cos), Self.yarnInverseFrequency.map { cos($0) },
            "compressed layer cosine")
        expectClose(
            floats(plainAngles.sin), Self.plainInverseFrequency.map { sin($0) },
            "plain layer sine")
        expectClose(
            floats(compressedAngles.sin), Self.yarnInverseFrequency.map { sin($0) },
            "compressed layer sine")
    }

    /// The width of one head of the real DeepSeek-V4-Flash checkpoint.
    private static let realHeadDim = 512

    /// The number of trailing head dimensions the real checkpoint rotates.
    private static let realRopeDim = 64

    /// The first head dimension the real checkpoint rotates.
    private static let realNoPositionDim = realHeadDim - realRopeDim

    /// The position the rotation test reads. A position of 0 leaves every
    /// dimension unchanged, thus the test needs a position of more than 0.
    private static let rotationPosition = 3

    /// The smallest change the test reads as a rotation. The slowest pair of
    /// the real table turns by 4e-4 radians at position 3, thus every
    /// rotated dimension moves well above this limit.
    private static let rotationFloor: Float = 1e-5

    /// The number of entries a rotary table holds for each rotary pair. Two
    /// head dimensions make one pair, thus the table is half as wide.
    private static let rotaryPairWidth = 2

    /// The one head the rotation test rotates.
    private static let singleHead = 1

    @Test
    func rotationReachesOnlyTheTrailingHeadDimensions() throws {
        let config = try JSONDecoder().decode(
            DeepSeekV4Configuration.self, from: Data("{}".utf8))
        let rope = DeepSeekV4RoPE(configuration: config, layer: Self.plainLayer)
        let head = MLXArray.ones([
            Self.batchSize, Self.singleHead, Self.decodeLength, Self.realHeadDim,
        ])

        let angles = rope.cosSin(offset: Self.rotationPosition, length: Self.decodeLength)
        let rotated = DeepSeekV4Math.applyPartialRoPE(
            head,
            cos: angles.cos.expandedDimensions(axes: [0, 1]),
            sin: angles.sin.expandedDimensions(axes: [0, 1]),
            ropeDim: Self.realRopeDim)

        #expect(angles.cos.shape.last == Self.realRopeDim / Self.rotaryPairWidth)
        let before = floats(head)
        let after = floats(rotated)
        for dimension in 0 ..< Self.realNoPositionDim {
            #expect(after[dimension] == before[dimension], "dimension \(dimension) must not turn")
        }
        for dimension in Self.realNoPositionDim ..< Self.realHeadDim {
            #expect(
                abs(after[dimension] - before[dimension]) > Self.rotationFloor,
                "dimension \(dimension) must turn")
        }
    }

    // MARK: - Checkpoint key paths

    @Test
    func parameterKeysMatchTheCheckpoint() throws {
        let block = try Self.attention(layer: Self.plainLayer, useAttnSink: true)

        let keys = Set(block.parameters().flattened().map { $0.0 })

        #expect(
            keys == [
                "wq_a.weight", "wq_b.weight", "wkv.weight", "wo_a.weight", "wo_b.weight",
                "q_norm.weight", "kv_norm.weight", "attn_sink",
            ])
    }
}

/// A key/value cache that scales the values it hands back.
///
/// The scale makes the cache's own return observable from outside: a model
/// that fed its own tensor to attention rather than the tensor this cache
/// returned would leave the scale out of its answer. That is the shape of the
/// buffer leak `ml-explore/mlx-lm` issue 1662 reports.
private final class GainKVCache: BaseKVCache {

    private let valueGain: Float
    private var storedKeys: MLXArray?
    private var storedValues: MLXArray?

    /// The axis of a `(batch, heads, tokens, width)` tensor that holds the
    /// tokens.
    private static let tokenAxis = 2

    init(valueGain: Float) {
        self.valueGain = valueGain
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let allKeys = storedKeys.map { concatenated([$0, keys], axis: Self.tokenAxis) } ?? keys
        let allValues =
            storedValues.map { concatenated([$0, values], axis: Self.tokenAxis) } ?? values
        storedKeys = allKeys
        storedValues = allValues
        offset = allKeys.dim(Self.tokenAxis)
        return (allKeys, allValues * valueGain)
    }
}
