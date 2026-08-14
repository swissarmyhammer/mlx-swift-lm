// Copyright © 2026 Apple Inc.
//
// The pooled-key half of DeepSeek-V4 compressed sparse attention.
//
// ``DeepSeekV4Compressor`` pools each run of `compress_ratio` tokens into one
// chunk. The chunks are the keys the global context of a long prompt is read
// through, thus a wrong chunk width, a wrong pooled width or a leak from a
// token the chunk has not reached yet is a silent quality loss rather than a
// crash.
//
// Two rules decide the whole file, and the published
// `mlx-community/DeepSeek-V4-Flash-4bit` checkpoint states both:
//
//  1. A layer whose compress ratio is 0 holds no compressor. Layers 0 and 1
//     are such layers, and their attention must stay exactly what it was
//     before the compressor landed.
//  2. A layer whose compress ratio is 4 pools with overlap, thus its
//     projections answer twice the pooled width and each pooled chunk also
//     reads the chunk before it. A layer whose ratio is 128 does not.
//
// The causal test below states the pooling rule again in plain Swift rather
// than calling the module, thus a wrong rule in the module cannot make the
// test agree with it.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4CompressorTests {

    // MARK: - The synthetic checkpoint

    /// The number of tokens in the vocabulary of the synthetic checkpoint.
    private static let vocabSize = 12

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 8

    /// The width of one attention head, which is also the width of one pooled
    /// chunk of the compressor of an attention layer.
    private static let headDim = 8

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 4

    /// The index of a layer whose compress ratio is 0, thus a layer that holds
    /// no compressor at all.
    private static let plainLayer = 0

    /// The index of a layer whose compress ratio is ``overlapRatio``, thus a
    /// layer that pools with overlap and holds an indexer.
    private static let overlapLayer = 2

    /// The index of a layer whose compress ratio is ``wideRatio``, thus a
    /// layer that pools without overlap and holds no indexer.
    private static let wideLayer = 3

    /// The compress ratio of ``overlapLayer``.
    private static let overlapRatio = 4

    /// The compress ratio of ``wideLayer``. It is also the sliding window of
    /// the published checkpoint.
    private static let wideRatio = 128

    /// The compress ratios the synthetic checkpoint gives its four layers.
    private static let compressRatios = "[0, 0, \(overlapRatio), \(wideRatio)]"

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// - Parameter compressRatios: The `compress_ratios` list, as JSON. An
    ///   empty list gives a checkpoint that holds no compressor on any layer,
    ///   which is the shape of the model before this feature landed.
    /// - Returns: The decoded configuration.
    private static func configuration(
        compressRatios: String = compressRatios
    ) throws -> DeepSeekV4Configuration {
        let json = """
            {
              "vocab_size": \(vocabSize),
              "hidden_size": \(hiddenSize),
              "num_hidden_layers": \(layerCount),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": \(headDim),
              "qk_rope_head_dim": 4,
              "q_lora_rank": 6,
              "o_groups": 2,
              "o_lora_rank": 4,
              "n_routed_experts": 8,
              "n_shared_experts": 1,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 8,
              "num_hash_layers": 0,
              "hc_mult": 2,
              "hc_sinkhorn_iters": 4,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "compress_rope_theta": 160000.0,
              "compress_ratios": \(compressRatios),
              "index_n_heads": 2,
              "index_head_dim": 4,
              "index_topk": 3,
              "use_attn_sink": true
            }
            """
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    // MARK: - Fixture builders

    /// The low end of every random fixture value.
    private static let fixtureLow: Float = -0.5

    /// The high end of every random fixture value.
    private static let fixtureHigh: Float = 0.5

    /// The seed the weight filler starts at.
    private static let weightSeed: UInt64 = 20_260_814

    /// Builds a repeatable random tensor.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - seed: The seed of this tensor.
    /// - Returns: The tensor, in float32.
    private static func fixture(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.uniform(low: fixtureLow, high: fixtureHigh, shape, key: MLXRandom.key(seed))
    }

    /// Builds one repeatable random weight for each parameter of a module.
    ///
    /// - Parameters:
    ///   - module: The module whose parameters the checkpoint must fill.
    ///   - seed: The seed the filler starts at.
    /// - Returns: The weights, by parameter path.
    private static func checkpoint(for module: Module, seed: UInt64) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        var next = seed
        for (key, value) in module.parameters().flattened().sorted(by: { $0.0 < $1.0 }) {
            next += 1
            weights[key] = fixture(value.shape, seed: next)
        }
        return weights
    }

    /// The values of an array, in row-major order.
    ///
    /// - Parameter array: The array to read.
    /// - Returns: Every value of the array, in float32.
    private static func floats(_ array: MLXArray) -> [Float] {
        eval(array)
        return array.asType(.float32).reshaped(-1).asArray(Float.self)
    }

    // MARK: - The plain-attention path

    /// The number of sequences every forward pass below carries.
    private static let batchSize = 1

    /// The number of tokens the attention fixture carries.
    private static let attentionTokenCount = 5

    /// The output of one attention layer, given the weights a checkpoint with
    /// no compressor at all would fill.
    ///
    /// Both configurations declare the same parameters at a layer whose
    /// compress ratio is 0, thus the same checkpoint loads into both and the
    /// two answers are comparable value for value.
    ///
    /// - Parameters:
    ///   - layer: The index of the decoder layer.
    ///   - compressRatios: The `compress_ratios` list, as JSON.
    ///   - weights: The checkpoint to load, by parameter path.
    /// - Returns: The block output, shape `(batch, tokens, hidden)`.
    private static func attentionOutput(
        layer: Int, compressRatios: String, weights: [String: MLXArray]
    ) throws -> MLXArray {
        let block = DeepSeekV4Attention(
            configuration: try configuration(compressRatios: compressRatios), layer: layer)
        try block.update(parameters: ModuleParameters.unflattened(weights), verify: [])
        let input = fixture([batchSize, attentionTokenCount, hiddenSize], seed: weightSeed)
        return block(input, mask: .causal, cache: nil)
    }

    /// The `compress_ratios` list of a checkpoint that holds no compressor on
    /// any layer, which is the shape of the model before this feature landed.
    private static let noCompressRatios = "[]"

    @Test func aCompressRatioZeroLayerIsBitIdenticalToPlainAttention() throws {
        // The weights come from a layer of the plain checkpoint, thus they name
        // exactly the tensors a layer with no compressor carries.
        let plain = DeepSeekV4Attention(
            configuration: try Self.configuration(compressRatios: Self.noCompressRatios),
            layer: Self.plainLayer)
        let weights = Self.checkpoint(for: plain, seed: Self.weightSeed)

        let zeroRatio = try Self.attentionOutput(
            layer: Self.plainLayer, compressRatios: Self.compressRatios, weights: weights)
        let noCompressor = try Self.attentionOutput(
            layer: Self.plainLayer, compressRatios: Self.noCompressRatios, weights: weights)

        #expect(
            Self.floats(zeroRatio) == Self.floats(noCompressor),
            "a layer whose compress ratio is 0 must read the plain-attention path")

        // The same comparison on a layer whose ratio is more than 0 fails,
        // because that layer turns its positions with `compress_rope_theta`.
        // Without this the equality above would hold for a comparison that can
        // never fail.
        let compressed = try Self.attentionOutput(
            layer: Self.overlapLayer, compressRatios: Self.compressRatios, weights: weights)
        let compressedAsPlain = try Self.attentionOutput(
            layer: Self.overlapLayer, compressRatios: Self.noCompressRatios, weights: weights)
        #expect(
            Self.floats(compressed) != Self.floats(compressedAsPlain),
            "a compressed layer must not answer what the plain-attention path answers")
    }

    // MARK: - The pooled chunks

    /// The number of trailing head dimensions that take a position.
    private static let ropeDim = 4

    /// The width of one pooled chunk of the compressor inside an indexer,
    /// which is `index_head_dim` rather than `head_dim`.
    private static let indexHeadDim = 4

    /// The factor an overlapping compressor widens its projections by, because
    /// each pooled chunk also reads the chunk before it.
    private static let overlapWidthFactor = 2

    /// Builds one compressor of the synthetic checkpoint and loads a
    /// repeatable random weight into every parameter it declares.
    ///
    /// The load runs through `update(parameters:verify:[.all])`, which is the
    /// verification `MLXLMCommon.loadWeights` applies, thus this builder also
    /// states that the compressor declares exactly the parameters the
    /// published checkpoint fills.
    ///
    /// - Parameters:
    ///   - layer: The index of the decoder layer.
    ///   - seed: The seed the weight filler starts at.
    /// - Returns: The loaded compressor.
    private static func loadedCompressor(layer: Int, seed: UInt64) throws -> DeepSeekV4Compressor {
        let compressor = DeepSeekV4Compressor(
            configuration: try configuration(), layer: layer, headDim: headDim)
        try compressor.update(
            parameters: ModuleParameters.unflattened(checkpoint(for: compressor, seed: seed)),
            verify: [.all])
        eval(compressor)
        return compressor
    }

    /// The chunks one compressor pools out of a block of tokens.
    ///
    /// - Parameters:
    ///   - compressor: The compressor under test.
    ///   - layer: The index of the decoder layer that compressor belongs to.
    ///   - input: The block input, shape `(batch, tokens, hidden)`.
    ///   - offset: The absolute position of the first token of the block.
    /// - Returns: The pooled chunks, shape `(batch, chunks, headDim)`.
    private static func pooled(
        from compressor: DeepSeekV4Compressor, layer: Int, input: MLXArray, offset: Int
    ) throws -> MLXArray {
        let rope = DeepSeekV4RoPE(configuration: try configuration(), layer: layer)
        let chunks = compressor(input, rope: rope, offset: offset)
        eval(chunks)
        return chunks
    }

    /// A block of tokens for the compressor to pool.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens in the block.
    ///   - seed: The seed of this block.
    /// - Returns: The block, shape `(batch, tokenCount, hidden)`.
    private static func block(tokenCount: Int, seed: UInt64) -> MLXArray {
        fixture([batchSize, tokenCount, hiddenSize], seed: seed)
    }

    /// The number of chunks the pooling tests read.
    private static let pooledChunkCount = 3

    @Test func theProjectionsTakeTheirWidthFromTheCompressRatio() throws {
        let overlapping = DeepSeekV4Compressor(
            configuration: try Self.configuration(), layer: Self.overlapLayer,
            headDim: Self.headDim)
        let wide = DeepSeekV4Compressor(
            configuration: try Self.configuration(), layer: Self.wideLayer, headDim: Self.headDim)

        // A ratio-4 layer pools with overlap, thus both projections answer
        // twice the pooled width, and `ape` gives one bias for each position
        // of a window.
        let overlapWidth = Self.headDim * Self.overlapWidthFactor
        #expect(overlapping.wkv.weight.shape == [overlapWidth, Self.hiddenSize])
        #expect(overlapping.wgate.weight.shape == [overlapWidth, Self.hiddenSize])
        #expect(overlapping.ape.shape == [Self.overlapRatio, overlapWidth])
        #expect(overlapping.norm.weight.shape == [Self.headDim])

        // A ratio-128 layer does not overlap, thus its projections answer the
        // pooled width itself.
        #expect(wide.wkv.weight.shape == [Self.headDim, Self.hiddenSize])
        #expect(wide.wgate.weight.shape == [Self.headDim, Self.hiddenSize])
        #expect(wide.ape.shape == [Self.wideRatio, Self.headDim])
        #expect(wide.norm.weight.shape == [Self.headDim])
    }

    @Test func theCompressorInsideAnIndexerPoolsToTheIndexHeadWidth() throws {
        let indexer = DeepSeekV4Indexer(
            configuration: try Self.configuration(), layer: Self.overlapLayer)

        let indexWidth = Self.indexHeadDim * Self.overlapWidthFactor
        #expect(indexer.compressor.wkv.weight.shape == [indexWidth, Self.hiddenSize])
        #expect(indexer.compressor.ape.shape == [Self.overlapRatio, indexWidth])
        #expect(indexer.compressor.norm.weight.shape == [Self.indexHeadDim])
    }

    @Test func aPooledChunkCoversOneRunOfTokensOfTheCompressRatio() throws {
        for (layer, ratio) in [
            (Self.overlapLayer, Self.overlapRatio), (Self.wideLayer, Self.wideRatio),
        ] {
            let compressor = try Self.loadedCompressor(layer: layer, seed: Self.weightSeed)
            let tokenCount = ratio * Self.pooledChunkCount
            let chunks = try Self.pooled(
                from: compressor, layer: layer,
                input: Self.block(tokenCount: tokenCount, seed: Self.weightSeed), offset: 0)

            #expect(
                chunks.shape == [Self.batchSize, Self.pooledChunkCount, Self.headDim],
                "compress ratio \(ratio)")
            #expect(
                Self.floats(chunks).allSatisfy { $0.isFinite }, "compress ratio \(ratio)")
        }
    }

    @Test func aBlockShorterThanOneChunkPoolsNothing() throws {
        let compressor = try Self.loadedCompressor(
            layer: Self.overlapLayer, seed: Self.weightSeed)
        let chunks = try Self.pooled(
            from: compressor, layer: Self.overlapLayer,
            input: Self.block(tokenCount: Self.overlapRatio - 1, seed: Self.weightSeed),
            offset: 0)

        #expect(chunks.shape == [Self.batchSize, 0, Self.headDim])
    }

    @Test func aBlockKeepsTheTokensOfItsIncompleteLastChunkOutOfEveryPool() throws {
        let compressor = try Self.loadedCompressor(
            layer: Self.overlapLayer, seed: Self.weightSeed)
        let whole = Self.overlapRatio * Self.pooledChunkCount

        let complete = try Self.pooled(
            from: compressor, layer: Self.overlapLayer,
            input: Self.block(tokenCount: whole, seed: Self.weightSeed), offset: 0)
        let withTail = try Self.pooled(
            from: compressor, layer: Self.overlapLayer,
            input: Self.block(
                tokenCount: whole + Self.overlapRatio - 1, seed: Self.weightSeed),
            offset: 0)

        // The tail cannot add a chunk of its own, because no chunk is complete
        // until `compress_ratio` tokens stand in it.
        #expect(withTail.shape == complete.shape)
    }

    @Test func aPooledChunkReadsNoTokenPastItsOwnEnd() throws {
        for (layer, ratio) in [
            (Self.overlapLayer, Self.overlapRatio), (Self.wideLayer, Self.wideRatio),
        ] {
            let compressor = try Self.loadedCompressor(layer: layer, seed: Self.weightSeed)
            let tokenCount = ratio * Self.pooledChunkCount
            let first = Self.block(tokenCount: tokenCount, seed: Self.weightSeed)

            // The same block, with the tokens of the last chunk replaced. Only
            // the last pooled chunk may notice.
            let changed = concatenated(
                [
                    first[0..., 0 ..< (tokenCount - ratio), 0...],
                    Self.block(tokenCount: ratio, seed: Self.weightSeed + 1),
                ], axis: 1)

            let before = Self.floats(
                try Self.pooled(from: compressor, layer: layer, input: first, offset: 0))
            let after = Self.floats(
                try Self.pooled(from: compressor, layer: layer, input: changed, offset: 0))

            let untouched = (Self.pooledChunkCount - 1) * Self.headDim
            #expect(
                Array(before.prefix(untouched)) == Array(after.prefix(untouched)),
                "a chunk that ends before the change must not move, ratio \(ratio)")
            #expect(
                Array(before.suffix(Self.headDim)) != Array(after.suffix(Self.headDim)),
                "the last chunk covers the change, thus it must move, ratio \(ratio)")
        }
    }

    /// The largest gap allowed between two float32 results of the same
    /// arithmetic in a different order.
    private static let tolerance: Float = 1e-5

    /// The offset the rotary test reads, which is one whole chunk on.
    private static let rotaryOffset = overlapRatio

    @Test func theOffsetTurnsOnlyTheRotaryHalfOfEachPooledChunk() throws {
        let compressor = try Self.loadedCompressor(
            layer: Self.overlapLayer, seed: Self.weightSeed)
        let input = Self.block(
            tokenCount: Self.overlapRatio * Self.pooledChunkCount, seed: Self.weightSeed)

        let atZero = Self.floats(
            try Self.pooled(from: compressor, layer: Self.overlapLayer, input: input, offset: 0))
        let atOffset = Self.floats(
            try Self.pooled(
                from: compressor, layer: Self.overlapLayer, input: input,
                offset: Self.rotaryOffset))

        #expect(atZero != atOffset, "the offset must reach the rotary position")
        for chunk in 0 ..< Self.pooledChunkCount {
            let start = chunk * Self.headDim
            let noPosition = Self.headDim - Self.ropeDim
            #expect(
                Array(atZero[start ..< (start + noPosition)])
                    == Array(atOffset[start ..< (start + noPosition)]),
                "the leading values of chunk \(chunk) take no position")

            // A rotation keeps the length of each pair, thus the two offsets
            // agree on every pair magnitude however far apart they turn it.
            for pair in stride(from: start + noPosition, to: start + Self.headDim, by: 2) {
                let zeroLength = hypot(atZero[pair], atZero[pair + 1])
                let offsetLength = hypot(atOffset[pair], atOffset[pair + 1])
                #expect(
                    abs(zeroLength - offsetLength) <= Self.tolerance,
                    "the rotary pair at \(pair) must keep its length")
            }
        }
    }

    // MARK: - The layers that hold a compressor

    @Test func attentionHoldsACompressorOnlyOnACompressorLayer() throws {
        let configuration = try Self.configuration()

        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.plainLayer)
                .compressor == nil,
            "a layer whose compress ratio is 0 must hold no compressor at all")
        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.overlapLayer)
                .compressor != nil)
        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.wideLayer)
                .compressor != nil)
    }

    // MARK: - A prompt past the sliding window

    /// The number of tokens the long-prompt test reads. It stands past the
    /// `sliding_window` of 128 and past ``wideRatio``, thus every compressor of
    /// the synthetic checkpoint would pool at least one chunk out of it.
    private static let longPromptLength = 512

    @Test func aPromptPastTheSlidingWindowGivesFiniteLogits() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        try model.update(
            parameters: ModuleParameters.unflattened(
                Self.checkpoint(for: model, seed: Self.weightSeed)),
            verify: [.all])
        eval(model)

        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])
        let logits = model(tokens, cache: nil)

        #expect(logits.shape == [Self.batchSize, Self.longPromptLength, Self.vocabSize])
        #expect(Self.floats(logits).allSatisfy { $0.isFinite })
    }

    // MARK: - The layers the compressor treats as compressed

    /// The number of decoder layers of the published checkpoint.
    private static let publishedLayerCount = 43

    /// The first layer of the published checkpoint that holds a compressor.
    private static let firstPublishedCompressorLayer = 2

    /// The last layer of the published checkpoint that holds a compressor.
    private static let lastPublishedCompressorLayer = 42

    /// The published `config.json` of `mlx-community/DeepSeek-V4-Flash-4bit`.
    ///
    /// - Returns: The decoded configuration.
    private static func publishedConfiguration() throws -> DeepSeekV4Configuration {
        let url = try #require(
            Bundle.module.url(forResource: "DeepSeek-V4-Flash-4bit-config", withExtension: "json"))
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(contentsOf: url))
    }

    @Test func theCompressedLayersAreTheCompressRopeThetaLayers() throws {
        let configuration = try Self.publishedConfiguration()
        #expect(configuration.numHiddenLayers == Self.publishedLayerCount)
        #expect(
            configuration.compressRopeTheta != configuration.ropeTheta,
            "the premise: the two thetas differ, thus the comparison below can fail")
        #expect(
            configuration.compressRatios.count != configuration.numHiddenLayers,
            "the premise: the ratio list is longer than the layer count")

        var compressorLayers: [Int] = []
        var compressThetaLayers: [Int] = []
        for layer in 0 ..< configuration.numHiddenLayers {
            if configuration.hasCompressor(layer: layer) {
                compressorLayers.append(layer)
            }
            if configuration.ropeTheta(forLayer: layer) == configuration.compressRopeTheta {
                compressThetaLayers.append(layer)
            }
        }

        #expect(compressorLayers == compressThetaLayers)
        #expect(
            compressorLayers
                == Array(Self.firstPublishedCompressorLayer ... Self.lastPublishedCompressorLayer),
            "the published checkpoint names compressor tensors on the 41 layers 2 to 42")
    }

    // MARK: - The load filter

    /// A one-value array, which stands in for a checkpoint tensor whose values
    /// no test reads.
    private static func marker(_ value: Float) -> MLXArray {
        MLXArray([value])
    }

    @Test func sanitizeKeepsTheCompressorOfTheAttentionAndOfTheIndexer() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let layer = Self.overlapLayer
        let sanitized = model.sanitize(weights: [
            "layers.\(layer).attn.compressor.wkv.weight": Self.marker(1),
            "layers.\(layer).attn.compressor.wgate.weight": Self.marker(2),
            "layers.\(layer).attn.compressor.ape": Self.marker(3),
            "layers.\(layer).attn.compressor.norm.weight": Self.marker(4),
            "layers.\(layer).attn.indexer.compressor.wkv.weight": Self.marker(5),
            "layers.\(layer).attn.indexer.wq_b.weight": Self.marker(6),
        ])

        for suffix in [
            "attn.compressor.wkv.weight",
            "attn.compressor.wgate.weight",
            "attn.compressor.ape",
            "attn.compressor.norm.weight",
            "attn.indexer.compressor.wkv.weight",
            "attn.indexer.wq_b.weight",
        ] {
            let path = "model.layers.\(layer).\(suffix)"
            #expect(sanitized[path] != nil, Comment(rawValue: path))
        }
    }
}
