// Copyright © 2026 Apple Inc.
//
// The pooled cache of DeepSeek-V4 compressed sparse attention.
//
// A decode step carries ONE token. A stateless compressor pools nothing out of
// one token, thus the global context would go away at the first decode step.
// ``DeepSeekV4ChunkCache`` is what keeps the pooled chunks across calls, and
// ``DeepSeekV4Cache`` joins it to the sliding window of one layer.
//
// The rule the whole file states is: **the chunks a block pools must not
// depend on how the block was cut**. A prompt arrives in prefill chunks whose
// boundaries are not multiples of the compress ratio, and a decode step then
// adds one token at a time. Every one of those cuts must give the chunks the
// one-call pooling gives.
//
// A ratio-4 layer pools with OVERLAP: chunk `c` reads the tokens of chunk
// `c - 1` as well as its own. A cache that keeps only the incomplete tail
// therefore gives chunk `c` a padded left half at each call boundary, and the
// comparison below fails. That is the defect these tests exist to hold.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4CacheTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

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

    /// The index of a layer that pools with overlap and holds an indexer.
    private static let overlapLayer = 2

    /// The index of a layer that pools without overlap and holds no indexer.
    private static let wideLayer = 3

    /// The compress ratio of ``overlapLayer``. A layer of this ratio pools
    /// with overlap.
    private static let overlapRatio = 4

    /// The compress ratio of ``wideLayer``.
    private static let wideRatio = 16

    /// The sliding window of the synthetic checkpoint.
    private static let slidingWindow = 8

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// - Returns: The decoded configuration.
    private static func configuration() throws -> DeepSeekV4Configuration {
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
              "sliding_window": \(slidingWindow),
              "compress_ratios": [0, 0, \(overlapRatio), \(wideRatio)],
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

    /// The number of sequences every fixture carries.
    private static let batchSize = 1

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

    /// Builds one compressor of the synthetic checkpoint and loads a
    /// repeatable random weight into every parameter it declares.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: The loaded compressor.
    private static func loadedCompressor(layer: Int) throws -> DeepSeekV4Compressor {
        let compressor = DeepSeekV4Compressor(
            configuration: try configuration(), layer: layer, headDim: headDim)
        try compressor.update(
            parameters: ModuleParameters.unflattened(
                checkpoint(for: compressor, seed: weightSeed)),
            verify: [.all])
        eval(compressor)
        return compressor
    }

    /// Builds an empty pooled cache for one layer.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: The cache.
    private static func chunkCache(layer: Int) throws -> DeepSeekV4ChunkCache {
        DeepSeekV4ChunkCache(configuration: try configuration(), layer: layer)
    }

    // MARK: - The chunks a cut block pools

    /// The number of tokens the pooling tests feed. It stands past the sliding
    /// window and past ``wideRatio``, thus both layers pool several chunks.
    private static let pooledTokenCount = 67

    /// The cuts the pooling tests read. None of them is a multiple of both
    /// compress ratios, and the last of them is a run of single tokens, which
    /// is the decode shape.
    private static let feedCuts: [[Int]] = [
        [67],
        [32, 35],
        [1, 3, 5, 7, 11, 40],
        [17, 17, 17, 16],
    ]

    /// Pools one block of tokens through a cache, cut into the given lengths.
    ///
    /// - Parameters:
    ///   - block: The whole block, shape `(batch, tokens, hidden)`.
    ///   - cuts: The length of each call. They must add up to the token count.
    ///   - layer: The index of the decoder layer.
    /// - Returns: The chunks the cache holds after the last call.
    private static func pooledInCuts(
        _ block: MLXArray, cuts: [Int], layer: Int
    ) throws -> MLXArray {
        let compressor = try loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try configuration(), layer: layer)
        let cache = try chunkCache(layer: layer)
        var offset = 0
        var pooled = zeros([batchSize, 0, headDim])
        for cut in cuts {
            pooled = cache.pooled(
                block[0..., offset ..< (offset + cut), 0...],
                through: compressor, rope: rope, offset: offset)
            offset += cut
        }
        eval(pooled)
        return pooled
    }

    /// The largest gap allowed between two float32 poolings of the same tokens
    /// in a different call order.
    private static let tolerance: Float = 1e-5

    @Test func aBlockCutIntoSeveralCallsPoolsTheChunksOfOneCall() throws {
        for layer in [Self.overlapLayer, Self.wideLayer] {
            let block = Self.fixture(
                [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)
            let whole = Self.floats(
                try Self.pooledInCuts(block, cuts: [Self.pooledTokenCount], layer: layer))
            #expect(!whole.isEmpty, "layer \(layer) must pool at least one chunk")

            for cuts in Self.feedCuts {
                let cut = Self.floats(try Self.pooledInCuts(block, cuts: cuts, layer: layer))
                #expect(cut.count == whole.count, "layer \(layer), cuts \(cuts)")
                for (index, pair) in zip(cut, whole).enumerated() {
                    let where_ = "layer \(layer), cuts \(cuts), value \(index)"
                    #expect(
                        abs(pair.0 - pair.1) <= Self.tolerance,
                        Comment(rawValue: "\(where_): got \(pair.0), expected \(pair.1)"))
                }
            }
        }
    }

    @Test func aChunkArrivesOnlyWhenItsWholeRunOfTokensHasArrived() throws {
        let layer = Self.wideLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let cache = try Self.chunkCache(layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)

        for position in 0 ..< Self.pooledTokenCount {
            let pooled = cache.pooled(
                block[0..., position ..< (position + 1), 0...],
                through: compressor, rope: rope, offset: position)
            #expect(
                pooled.dim(1) == (position + 1) / Self.wideRatio,
                "at position \(position) the pool must hold the chunks that have ended")
        }
    }

    /// The position the rewind test goes back to. It stands INSIDE the second
    /// chunk, thus that chunk must go and be pooled again from the tokens the
    /// rewind reads next.
    private static let rewindPosition = overlapRatio + 1

    @Test func theCacheRewindsAndPoolsTheNewTokensOverTheOldOnes() throws {
        let layer = Self.overlapLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let window = Self.slidingWindow
        let rewindTo = Self.rewindPosition

        // The first run reads `first`. The rewind then reads `second` from
        // ``rewindPosition`` onward, thus the chunks that cover that range
        // must answer what the joined block answers.
        let first = Self.fixture(
            [Self.batchSize, window, Self.hiddenSize], seed: Self.weightSeed)
        let second = Self.fixture(
            [Self.batchSize, window, Self.hiddenSize], seed: Self.weightSeed + 1)
        let joined = concatenated(
            [first[0..., 0 ..< rewindTo, 0...], second[0..., rewindTo ..< window, 0...]], axis: 1)

        let cache = try Self.chunkCache(layer: layer)
        _ = cache.pooled(first, through: compressor, rope: rope, offset: 0)
        #expect(cache.rewind(to: rewindTo), "a fresh cache holds every raw row, thus it rewinds")
        let afterRewind = Self.floats(
            cache.pooled(
                second[0..., rewindTo ..< window, 0...],
                through: compressor, rope: rope, offset: rewindTo))

        let never = try Self.chunkCache(layer: layer)
        let straight = Self.floats(
            never.pooled(joined, through: compressor, rope: rope, offset: 0))
        let stale = Self.floats(
            try Self.chunkCache(layer: layer)
                .pooled(first, through: compressor, rope: rope, offset: 0))

        #expect(
            straight != stale,
            "the premise: the two blocks pool different chunks, thus a stale chunk shows")
        #expect(afterRewind.count == straight.count)
        for (index, pair) in zip(afterRewind, straight).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= Self.tolerance,
                Comment(rawValue: "value \(index): got \(pair.0), expected \(pair.1)"))
        }
    }

    // MARK: - The layer cache

    @Test func newCacheGivesAPooledCacheOnlyOnACompressorLayer() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let caches = try model.newCache(parameters: nil)

        #expect(caches.count == Self.layerCount)
        #expect(caches.allSatisfy { $0.isTrimmable }, "every fresh layer cache must rewind")
        #expect(caches[Self.plainLayer] is RotatingKVCache)
        #expect(caches[Self.overlapLayer] is DeepSeekV4Cache)
        #expect(caches[Self.wideLayer] is DeepSeekV4Cache)
        #expect(caches.allSatisfy { $0.maxSize == Self.slidingWindow })
    }

    @Test func theLayerCacheKeepsItsPooledChunksAcrossAStateRoundTrip() throws {
        let configuration = try Self.configuration()
        let layer = Self.overlapLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: configuration, layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)

        let cache = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let keyValues = Self.fixture(
            [Self.batchSize, 1, Self.pooledTokenCount, Self.headDim], seed: Self.weightSeed + 1)
        _ = cache.update(keys: keyValues, values: keyValues)
        let pooled = Self.floats(
            cache.attentionChunks.pooled(
                block, through: compressor, rope: rope, offset: 0))
        #expect(!pooled.isEmpty, "the premise: the cache holds pooled chunks to carry over")

        let restored = DeepSeekV4Cache(configuration: configuration, layer: layer)
        restored.state = cache.state.map { $0[.ellipsis] }
        restored.metaState = cache.metaState

        #expect(restored.offset == cache.offset)
        #expect(
            restored.attentionChunks.carryStart == cache.attentionChunks.carryStart,
            "the carry position must survive the round trip")
        let restoredChunks = try #require(restored.attentionChunks.chunks)
        #expect(
            Self.floats(restoredChunks) == pooled,
            "the pooled chunks must survive the round trip")
    }

    @Test func theLayerCacheCopiesItsPooledChunks() throws {
        let configuration = try Self.configuration()
        let layer = Self.wideLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: configuration, layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)

        let cache = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let pooled = Self.floats(
            cache.attentionChunks.pooled(block, through: compressor, rope: rope, offset: 0))
        #expect(!pooled.isEmpty, "the premise: the cache holds pooled chunks to copy")

        let duplicate = try #require(cache.copy() as? DeepSeekV4Cache)
        let duplicateChunks = try #require(duplicate.attentionChunks.chunks)
        #expect(Self.floats(duplicateChunks) == pooled)
    }

    // MARK: - The whole model

    /// The number of tokens the chunked-prefill test feeds. It stands well
    /// past the sliding window and past ``wideRatio``.
    private static let longPromptLength = 96

    /// The cut the chunked-prefill test feeds the prompt in. It is not a
    /// multiple of either compress ratio.
    private static let prefillCut = 13

    /// The largest gap allowed between two float32 forward passes over the
    /// same tokens with a different prefill cut.
    private static let logitTolerance: Float = 2e-3

    /// Builds the synthetic model with a repeatable random weight in every
    /// parameter.
    ///
    /// - Returns: The loaded model.
    private static func loadedModel() throws -> DeepSeekV4Model {
        let model = DeepSeekV4Model(try configuration())
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint(for: model, seed: weightSeed)),
            verify: [.all])
        eval(model)
        return model
    }

    @Test func aLongPromptGivesOneAnswerWhateverThePrefillCutIs() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])

        let whole = try model.newCache(parameters: nil)
        let wholeLogits = Self.floats(model(tokens, cache: whole)[0..., -1, 0...])

        let cut = try model.newCache(parameters: nil)
        var cutLogits: [Float] = []
        var position = 0
        while position < Self.longPromptLength {
            let end = min(position + Self.prefillCut, Self.longPromptLength)
            cutLogits = Self.floats(
                model(tokens[0..., position ..< end], cache: cut)[0..., -1, 0...])
            position = end
        }

        #expect(cutLogits.count == wholeLogits.count)
        #expect(cutLogits.allSatisfy { $0.isFinite })
        for (index, pair) in zip(cutLogits, wholeLogits).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= Self.logitTolerance,
                "logit \(index): cut gave \(pair.0), one call gave \(pair.1)")
        }
    }

    @Test func aPromptWithNoCacheReadsTheSameSlidingWindowAsACachedRun() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])

        // A run that keeps no cache builds its own mask, thus it is where a
        // model that forgot to name its sliding window attends densely.
        let stateless = Self.floats(model(tokens, cache: nil)[0..., -1, 0...])
        let cached = Self.floats(
            model(tokens, cache: try model.newCache(parameters: nil))[0..., -1, 0...])

        #expect(stateless.count == cached.count)
        #expect(stateless.allSatisfy { $0.isFinite })
        for (index, pair) in zip(stateless, cached).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= Self.logitTolerance,
                Comment(rawValue: "logit \(index): no cache gave \(pair.0), a cache gave \(pair.1)"))
        }
    }

    @Test func aDecodeStepReadsTheChunksThePromptPooled() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])
        let cache = try model.newCache(parameters: nil)
        _ = model(tokens, cache: cache)

        let pooledLayer = try #require(cache[Self.overlapLayer] as? DeepSeekV4Cache)
        let expectedChunks = Self.longPromptLength / Self.overlapRatio
        #expect(
            pooledLayer.attentionChunks.chunkCount == expectedChunks,
            "the prompt must leave \(expectedChunks) pooled chunks behind")

        let next = MLXArray([Int32(1)]).reshaped([Self.batchSize, 1])
        let logits = model(next, cache: cache)
        #expect(logits.shape == [Self.batchSize, 1, Self.vocabSize])
        #expect(Self.floats(logits).allSatisfy { $0.isFinite })
        #expect(
            pooledLayer.attentionChunks.chunkCount == expectedChunks,
            "one decode step past a whole chunk must add no chunk")
        #expect(pooledLayer.offset == Self.longPromptLength + 1)
    }
}
