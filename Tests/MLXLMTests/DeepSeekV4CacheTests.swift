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
              "index_topk": \(indexTopK),
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

    /// Builds one compressor of the synthetic checkpoint and loads a
    /// repeatable random weight into every parameter it declares.
    ///
    /// - Parameter layer: The index of the decoder layer.
    /// - Returns: The loaded compressor.
    private static func loadedCompressor(layer: Int) throws -> DeepSeekV4Compressor {
        let compressor = DeepSeekV4Compressor(
            configuration: try configuration(), layer: layer, headDim: headDim)
        try DeepSeekV4RepeatableWeights.load(
            into: compressor, seed: weightSeed, low: fixtureLow, high: fixtureHigh)
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
        let chunks = try pooled(
            block, cuts: cuts, through: chunkCache(layer: layer),
            compressor: loadedCompressor(layer: layer), layer: layer)
        eval(chunks)
        return chunks
    }

    /// Pools one block of tokens through a given branch, cut into the given
    /// lengths.
    ///
    /// - Parameters:
    ///   - block: The whole block, shape `(batch, tokens, hidden)`.
    ///   - cuts: The length of each call. They must add up to the token count.
    ///   - branch: The branch that pools the block.
    ///   - compressor: The compressor of the branch.
    ///   - layer: The index of the decoder layer.
    /// - Returns: The chunks the branch holds after the last call.
    private static func pooled(
        _ block: MLXArray, cuts: [Int], through branch: DeepSeekV4ChunkCache,
        compressor: DeepSeekV4Compressor, layer: Int
    ) throws -> MLXArray {
        let rope = DeepSeekV4RoPE(configuration: try configuration(), layer: layer)
        var offset = 0
        var pooled = zeros([batchSize, 0, headDim])
        for cut in cuts {
            pooled = branch.pooled(
                block[0..., offset ..< (offset + cut), 0...],
                through: compressor, rope: rope, offset: offset)
            offset += cut
        }
        return pooled
    }

    /// The largest gap allowed between two float32 poolings of the same tokens
    /// in a different call order.
    private static let tolerance: Float = 1e-5

    @Test func aBlockCutIntoSeveralCallsPoolsTheChunksOfOneCall() throws {
        for layer in [Self.overlapLayer, Self.wideLayer] {
            let block = Self.fixture(
                [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)
            let whole = DeepSeekV4TestArrays.floats(
                try Self.pooledInCuts(block, cuts: [Self.pooledTokenCount], layer: layer))
            #expect(!whole.isEmpty, "layer \(layer) must pool at least one chunk")

            for cuts in Self.feedCuts {
                let cut = DeepSeekV4TestArrays.floats(
                    try Self.pooledInCuts(block, cuts: cuts, layer: layer))
                #expect(cut.count == whole.count, "layer \(layer), cuts \(cuts)")
                for (index, pair) in zip(cut, whole).enumerated() {
                    let place = "layer \(layer), cuts \(cuts), value \(index)"
                    #expect(
                        abs(pair.0 - pair.1) <= Self.tolerance,
                        Comment(rawValue: "\(place): got \(pair.0), expected \(pair.1)"))
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

    /// The number of whole chunks the no-work test feeds before its
    /// single-token steps. Their tokens stand past the sliding window, thus
    /// the cache keeps only the rows it needs for the next chunk.
    private static let leadingChunkCount = 3

    @Test(arguments: [overlapLayer, wideLayer])
    func aStepThatEndsNoChunkGivesBackTheChunksTheCacheHeld(layer: Int) throws {
        let ratio = try Self.configuration().compressRatio(ofLayer: layer)
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let cache = try Self.chunkCache(layer: layer)
        let prefix = Self.leadingChunkCount * ratio
        let block = Self.fixture(
            [Self.batchSize, prefix + ratio, Self.hiddenSize], seed: Self.weightSeed)

        var held = cache.pooled(
            block[0..., 0 ..< prefix, 0...], through: compressor, rope: rope, offset: 0)
        for position in prefix ..< (prefix + ratio) {
            let pooled = cache.pooled(
                block[0..., position ..< (position + 1), 0...],
                through: compressor, rope: rope, offset: position)
            let endsChunk = (position + 1).isMultiple(of: ratio)
            #expect(
                (pooled === held) == !endsChunk,
                "layer \(layer), position \(position): only a step that ends a chunk pools")
            held = pooled
        }
        #expect(held.dim(1) == Self.leadingChunkCount + 1)
    }

    @Test func theCacheKeepsTheProjectedRowsOfTheRunInProgress() throws {
        let layer = Self.overlapLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let cache = try Self.chunkCache(layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)
        _ = cache.pooled(block, through: compressor, rope: rope, offset: 0)

        let kept = block[0..., cache.carryStart..., 0...]
        let rows = try #require(cache.projectedRows, "the run in progress leaves rows behind")
        let pairs = [
            (rows.values, compressor.wkv(kept)),
            (rows.gates, compressor.wgate(kept)),
        ].map { (DeepSeekV4TestArrays.floats($0.0), DeepSeekV4TestArrays.floats($0.1)) }
        for (got, expected) in pairs {
            #expect(got.count == expected.count, "the cache keeps one projected row per token")
            for (index, pair) in zip(got, expected).enumerated() {
                #expect(
                    abs(pair.0 - pair.1) <= Self.tolerance,
                    Comment(rawValue: "value \(index): got \(pair.0), expected \(pair.1)"))
            }
        }
    }

    /// The axis a `(batch, tokens, width)` tensor holds its tokens on.
    private static let tokenAxis = 1

    /// The number of tokens one decode step carries.
    private static let decodeStepLength = 1

    /// The tokens of one decode step.
    ///
    /// - Parameters:
    ///   - block: The block, shape `(batch, tokens, hidden)`.
    ///   - position: The position of the token inside the block.
    /// - Returns: The token, shape `(batch, 1, hidden)`.
    private static func decodeStep(of block: MLXArray, at position: Int) -> MLXArray {
        block[0..., position ..< (position + decodeStepLength), 0...]
    }

    /// The position after the last projected row of a cache.
    ///
    /// - Parameter cache: The cache to read.
    /// - Returns: ``DeepSeekV4ChunkCache/carryStart`` plus the number of projected rows.
    private static func projectedEnd(of cache: DeepSeekV4ChunkCache) -> Int {
        cache.carryStart + (cache.projectedRows?.values.dim(tokenAxis) ?? 0)
    }

    @Test(arguments: [overlapLayer, wideLayer])
    func aStepProjectsTheRowsOfARunOnlyWhenItEndsTheRun(layer: Int) throws {
        let ratio = try Self.configuration().compressRatio(ofLayer: layer)
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let cache = try Self.chunkCache(layer: layer)
        let prefix = Self.leadingChunkCount * ratio
        let block = Self.fixture(
            [Self.batchSize, prefix + ratio, Self.hiddenSize], seed: Self.weightSeed)

        _ = cache.pooled(
            block[0..., 0 ..< prefix, 0...], through: compressor, rope: rope, offset: 0)
        for position in prefix ..< (prefix + ratio) {
            _ = cache.pooled(
                Self.decodeStep(of: block, at: position),
                through: compressor, rope: rope, offset: position)
            // The last step ends the run, thus there the projected rows must
            // reach the new token. Before it they must stop at the last chunk end.
            let lastChunkEnd = ((position + Self.decodeStepLength) / ratio) * ratio
            #expect(
                Self.projectedEnd(of: cache) == lastChunkEnd,
                "layer \(layer), position \(position): only a step that ends a run projects")
        }
    }

    /// Pools a block from rows that were projected one token at a time, which
    /// is what each decode step did before the cache kept the rows of a run
    /// unprojected.
    ///
    /// - Parameters:
    ///   - block: The block, shape `(batch, tokens, hidden)`.
    ///   - compressor: The compressor of the layer.
    ///   - layer: The index of the decoder layer.
    /// - Returns: The chunks of every whole run of the block.
    private static func pooledFromOneTokenProjections(
        _ block: MLXArray, compressor: DeepSeekV4Compressor, layer: Int
    ) throws -> MLXArray {
        let rope = DeepSeekV4RoPE(configuration: try configuration(), layer: layer)
        let wholeCount = (block.dim(tokenAxis) / compressor.chunkWidth) * compressor.chunkWidth
        let rows = (0 ..< wholeCount).map { position in
            compressor.project(decodeStep(of: block, at: position))
        }
        let joined = DeepSeekV4Compressor.ProjectedRows(
            values: concatenated(rows.map(\.values), axis: tokenAxis),
            gates: concatenated(rows.map(\.gates), axis: tokenAxis))
        return compressor.pooled(joined, rope: rope, offset: 0, dtype: block.dtype)
    }

    @Test(arguments: [overlapLayer, wideLayer])
    func aRunOfOneTokenStepsPoolsTheChunksOfRowsProjectedOneTokenAtATime(layer: Int) throws {
        let compressor = try Self.loadedCompressor(layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.pooledTokenCount, Self.hiddenSize], seed: Self.weightSeed)
        let stepped = DeepSeekV4TestArrays.floats(
            try Self.pooled(
                block,
                cuts: Array(repeating: Self.decodeStepLength, count: Self.pooledTokenCount),
                through: Self.chunkCache(layer: layer), compressor: compressor, layer: layer))
        let reference = DeepSeekV4TestArrays.floats(
            try Self.pooledFromOneTokenProjections(block, compressor: compressor, layer: layer))

        #expect(!reference.isEmpty, "the premise: layer \(layer) pools at least one chunk")
        #expect(stepped.count == reference.count, "layer \(layer)")
        for (index, pair) in zip(stepped, reference).enumerated() {
            let place = "layer \(layer), value \(index)"
            #expect(
                abs(pair.0 - pair.1) <= Self.tolerance,
                Comment(rawValue: "\(place): got \(pair.0), expected \(pair.1)"))
        }
    }

    /// The number of tokens the round trip test inside a run reads.
    private static let roundTripTokenCount = 72

    /// The position at which the round trip test inside a run saves its state.
    /// It is not a multiple of ``overlapRatio``, thus the cache then holds the
    /// rows of a run that has not ended.
    private static let roundTripSavePosition = 66

    @Test func aStateSavedInsideARunPoolsTheChunksOfAnUnbrokenRun() throws {
        let configuration = try Self.configuration()
        let layer = Self.overlapLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: configuration, layer: layer)
        let block = Self.fixture(
            [Self.batchSize, Self.roundTripTokenCount, Self.hiddenSize], seed: Self.weightSeed)
        let leading = (Self.roundTripSavePosition / Self.overlapRatio) * Self.overlapRatio
        #expect(
            leading < Self.roundTripSavePosition,
            "the premise: the state is saved inside a run that has not ended")

        let cache = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let keyValues = Self.fixture(
            [Self.batchSize, 1, Self.roundTripSavePosition, Self.headDim],
            seed: Self.weightSeed + 1)
        _ = cache.update(keys: keyValues, values: keyValues)
        _ = cache.attentionChunks.pooled(
            block[0..., 0 ..< leading, 0...], through: compressor, rope: rope, offset: 0)
        for position in leading ..< Self.roundTripSavePosition {
            _ = cache.attentionChunks.pooled(
                Self.decodeStep(of: block, at: position),
                through: compressor, rope: rope, offset: position)
        }

        let restored = DeepSeekV4Cache(configuration: configuration, layer: layer)
        try restored.validatePromptCacheRestore(
            state: cache.state.map { $0[.ellipsis] }, metaState: cache.metaState)
        restored.restorePromptCache(
            state: cache.state.map { $0[.ellipsis] }, metaState: cache.metaState)

        let rest = block[0..., Self.roundTripSavePosition..., 0...]
        let unbroken = DeepSeekV4TestArrays.floats(
            cache.attentionChunks.pooled(
                rest, through: compressor, rope: rope, offset: Self.roundTripSavePosition))
        let afterRestore = DeepSeekV4TestArrays.floats(
            restored.attentionChunks.pooled(
                rest, through: compressor, rope: rope, offset: Self.roundTripSavePosition))
        #expect(
            unbroken.count == (Self.roundTripTokenCount / Self.overlapRatio) * Self.headDim,
            "the premise: the rest of the block ends the run the state was saved inside")
        #expect(afterRestore == unbroken, "the rows of the run must survive the round trip")
    }

    /// The number of tokens each block of the rewind test inside a run holds.
    /// The block ends one run past the sliding window.
    private static let runRewindTokenCount = slidingWindow + overlapRatio

    /// The position the rewind test inside a run goes back to. It stands
    /// inside the run that the single-token steps have not ended.
    private static let runRewindPosition = slidingWindow + decodeStepLength

    @Test func theCacheRewindsInsideTheRunItHasNotProjected() throws {
        let layer = Self.overlapLayer
        let compressor = try Self.loadedCompressor(layer: layer)
        let rope = DeepSeekV4RoPE(configuration: try Self.configuration(), layer: layer)
        let length = Self.runRewindTokenCount
        let rewindTo = Self.runRewindPosition
        let first = Self.fixture([Self.batchSize, length, Self.hiddenSize], seed: Self.weightSeed)
        let second = Self.fixture(
            [Self.batchSize, length, Self.hiddenSize], seed: Self.weightSeed + 1)
        let joined = concatenated(
            [first[0..., 0 ..< rewindTo, 0...], second[0..., rewindTo ..< length, 0...]],
            axis: Self.tokenAxis)

        // The first run stops one step before the run ends, thus its last
        // tokens are not projected yet when the rewind comes.
        let cache = try Self.chunkCache(layer: layer)
        let stoppedAt = length - Self.decodeStepLength
        let steps = Array(
            repeating: Self.decodeStepLength,
            count: (stoppedAt - Self.slidingWindow) / Self.decodeStepLength)
        _ = try Self.pooled(
            first[0..., 0 ..< stoppedAt, 0...], cuts: [Self.slidingWindow] + steps,
            through: cache, compressor: compressor, layer: layer)
        #expect(
            Self.projectedEnd(of: cache) < rewindTo,
            "the premise: the rewind position stands past the projected rows")
        #expect(cache.rewind(to: rewindTo), "the rows from the carry position onward are held")
        let afterRewind = DeepSeekV4TestArrays.floats(
            cache.pooled(
                second[0..., rewindTo ..< length, 0...],
                through: compressor, rope: rope, offset: rewindTo))

        let straight = DeepSeekV4TestArrays.floats(
            try Self.chunkCache(layer: layer)
                .pooled(joined, through: compressor, rope: rope, offset: 0))
        let stale = DeepSeekV4TestArrays.floats(
            try Self.chunkCache(layer: layer)
                .pooled(first, through: compressor, rope: rope, offset: 0))
        #expect(
            straight != stale,
            "the premise: the two blocks pool different chunks, thus a stale row shows")
        #expect(afterRewind.count == straight.count)
        for (index, pair) in zip(afterRewind, straight).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= Self.tolerance,
                Comment(rawValue: "value \(index): got \(pair.0), expected \(pair.1)"))
        }
    }

    /// The number of raw rows each branch of the earlier state format carries.
    private static let rawRowCount = 3

    @Test func aPromptCacheOfTheRawRowFormatFailsClosed() throws {
        let configuration = try Self.configuration()
        let layer = Self.overlapLayer
        let source = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let keyValues = Self.fixture(
            [Self.batchSize, 1, Self.slidingWindow, Self.headDim], seed: Self.weightSeed)
        _ = source.update(keys: keyValues, values: keyValues)

        // The earlier format: each branch held the pooled chunks, the RAW block
        // rows and the carry position, and the meta state was the meta state of
        // the window alone.
        let rawRowBranch = [
            Self.fixture([Self.batchSize, 1, Self.headDim], seed: Self.weightSeed + 1),
            Self.fixture(
                [Self.batchSize, Self.rawRowCount, Self.hiddenSize], seed: Self.weightSeed + 2),
            MLXArray([Int32(0)]),
        ]
        let template = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let error = #expect(throws: KVCacheError.self) {
            try template.validatePromptCacheRestore(
                state: source.local.state + rawRowBranch + rawRowBranch,
                metaState: source.local.metaState)
        }
        #expect(error?.message == DeepSeekV4Cache.earlierFormatMessage)
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
        #expect(
            cache.rewind(to: rewindTo), "a fresh cache holds every projected row, thus it rewinds")
        let afterRewind = DeepSeekV4TestArrays.floats(
            cache.pooled(
                second[0..., rewindTo ..< window, 0...],
                through: compressor, rope: rope, offset: rewindTo))

        let never = try Self.chunkCache(layer: layer)
        let straight = DeepSeekV4TestArrays.floats(
            never.pooled(joined, through: compressor, rope: rope, offset: 0))
        let stale = DeepSeekV4TestArrays.floats(
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
        let pooled = DeepSeekV4TestArrays.floats(
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
            DeepSeekV4TestArrays.floats(restoredChunks) == pooled,
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
        let pooled = DeepSeekV4TestArrays.floats(
            cache.attentionChunks.pooled(block, through: compressor, rope: rope, offset: 0))
        #expect(!pooled.isEmpty, "the premise: the cache holds pooled chunks to copy")

        let duplicate = try #require(cache.copy() as? DeepSeekV4Cache)
        let duplicateChunks = try #require(duplicate.attentionChunks.chunks)
        #expect(DeepSeekV4TestArrays.floats(duplicateChunks) == pooled)
    }

    // MARK: - The dtype of the indexer pool

    /// The activation dtype of the real checkpoint, which the indexer pool
    /// tests feed.
    private static let activationDType = DType.bfloat16

    /// Builds the compressor of ``overlapLayer`` with every weight in
    /// ``activationDType``, as the real checkpoint holds it.
    ///
    /// A compressor with float32 weights pools float32 chunks from any block,
    /// because its norm and its rotation promote the chunks to float32.
    ///
    /// - Returns: The loaded compressor.
    private static func activationCompressor() throws -> DeepSeekV4Compressor {
        let compressor = try loadedCompressor(layer: overlapLayer)
        let weights = compressor.parameters().flattened().map { key, value in
            (key, value.asType(activationDType))
        }
        try compressor.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        return compressor
    }

    /// Pools a block of ``pooledTokenCount`` tokens in the activation dtype
    /// through one branch of ``overlapLayer``, cut into the given lengths.
    ///
    /// - Parameters:
    ///   - branch: The branch that pools the block.
    ///   - cuts: The length of each call. They must add up to
    ///     ``pooledTokenCount``.
    /// - Returns: The chunks the branch holds after the last call.
    private static func activationPooled(
        through branch: DeepSeekV4ChunkCache, cuts: [Int]
    ) throws -> MLXArray {
        let block = fixture([batchSize, pooledTokenCount, hiddenSize], seed: weightSeed)
            .asType(activationDType)
        return try pooled(
            block, cuts: cuts, through: branch, compressor: activationCompressor(),
            layer: overlapLayer)
    }

    @Test(arguments: feedCuts)
    func theIndexerBranchKeepsItsPoolInTheScoreDTypeWithTheBitsOfACast(cuts: [Int]) throws {
        let cache = DeepSeekV4Cache(configuration: try Self.configuration(), layer: Self.overlapLayer)
        let indexerBranch = try #require(cache.indexerChunks, "the overlap layer holds an indexer")
        let kept = try Self.activationPooled(through: indexerBranch, cuts: cuts)
        let activation = try Self.activationPooled(through: cache.attentionChunks, cuts: cuts)

        #expect(
            activation.dtype == Self.activationDType,
            "the premise: the attention branch keeps the activation dtype")
        #expect(kept.dtype == DeepSeekV4Indexer.scoreDType, "cuts \(cuts)")
        #expect(
            DeepSeekV4TestArrays.floats(kept) == DeepSeekV4TestArrays.floats(activation),
            "cuts \(cuts): the indexer pool holds the bits of a cast of the activation pool")
    }

    @Test func theIndexerBranchKeepsItsScoreDTypePoolAcrossAStateRoundTrip() throws {
        let configuration = try Self.configuration()
        let layer = Self.overlapLayer
        let cache = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let keyValues = Self.fixture(
            [Self.batchSize, 1, Self.pooledTokenCount, Self.headDim], seed: Self.weightSeed + 1)
        _ = cache.update(keys: keyValues, values: keyValues)
        let indexerBranch = try #require(cache.indexerChunks, "the overlap layer holds an indexer")
        let pooled = DeepSeekV4TestArrays.floats(
            try Self.activationPooled(through: indexerBranch, cuts: [Self.pooledTokenCount]))
        #expect(!pooled.isEmpty, "the premise: the cache holds pooled chunks to carry over")

        let restored = DeepSeekV4Cache(configuration: configuration, layer: layer)
        try restored.validatePromptCacheRestore(
            state: cache.state.map { $0[.ellipsis] }, metaState: cache.metaState)
        restored.restorePromptCache(
            state: cache.state.map { $0[.ellipsis] }, metaState: cache.metaState)

        let restoredChunks = try #require(restored.indexerChunks?.chunks)
        #expect(restoredChunks.dtype == DeepSeekV4Indexer.scoreDType)
        #expect(
            DeepSeekV4TestArrays.floats(restoredChunks) == pooled,
            "the indexer pool must survive the round trip")
    }

    @Test func aCopyOfTheCachePoolsItsIndexerBranchInTheScoreDType() throws {
        let cache = DeepSeekV4Cache(configuration: try Self.configuration(), layer: Self.overlapLayer)
        let duplicate = try #require(cache.copy() as? DeepSeekV4Cache)
        let indexerBranch = try #require(
            duplicate.indexerChunks, "the copy of the overlap layer holds an indexer")

        let kept = try Self.activationPooled(through: indexerBranch, cuts: [Self.pooledTokenCount])
        #expect(kept.dtype == DeepSeekV4Indexer.scoreDType)
    }

    /// The format names of the earlier prompt caches that kept the rows of a
    /// run in a projected form: the format whose indexer pool held the
    /// activation dtype, and the format whose indexer pool held the score dtype.
    private static let projectedRunFormats = [
        "DeepSeekV4Cache.projectedRows.v2", "DeepSeekV4Cache.scoreDTypeIndexerPool.v3",
    ]

    @Test(arguments: projectedRunFormats)
    func aPromptCacheOfAProjectedRunFormatFailsClosed(format: String) throws {
        let configuration = try Self.configuration()
        let layer = Self.overlapLayer
        let source = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let keyValues = Self.fixture(
            [Self.batchSize, 1, Self.slidingWindow, Self.headDim], seed: Self.weightSeed)
        _ = source.update(keys: keyValues, values: keyValues)

        let template = DeepSeekV4Cache(configuration: configuration, layer: layer)
        let error = #expect(throws: KVCacheError.self) {
            try template.validatePromptCacheRestore(
                state: source.state,
                metaState: source.local.metaState + [format])
        }
        #expect(error?.message == DeepSeekV4Cache.earlierFormatMessage)
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
        try DeepSeekV4RepeatableWeights.load(
            into: model, seed: weightSeed, low: fixtureLow, high: fixtureHigh)
        return model
    }

    /// Records one failure for each logit of a run that is farther than
    /// ``logitTolerance`` from the logit of the reference run.
    ///
    /// - Parameters:
    ///   - actual: The logits of the run under test.
    ///   - expected: The logits of the reference run.
    ///   - actualSource: The name of the run under test, for the message.
    ///   - expectedSource: The name of the reference run, for the message.
    ///   - sourceLocation: The location the failures point at.
    private static func expectSameLogits(
        _ actual: [Float],
        _ expected: [Float],
        actualSource: String,
        expectedSource: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(actual.count == expected.count, sourceLocation: sourceLocation)
        for (index, pair) in zip(actual, expected).enumerated() {
            #expect(
                abs(pair.0 - pair.1) <= logitTolerance,
                Comment(
                    rawValue: "logit \(index): \(actualSource) gave \(pair.0), "
                        + "\(expectedSource) gave \(pair.1)"),
                sourceLocation: sourceLocation)
        }
    }

    /// The logits of a prompt read in the generation shape, beside the
    /// logits of one call over the same prompt.
    private struct SteppedRun {
        /// The last logits of the run that prefills and then steps.
        let steppedLogits: [Float]

        /// The last logits of one call over the whole prompt.
        let wholeLogits: [Float]

        /// The cache of the run that prefills and then steps.
        let steppedCache: [KVCache]
    }

    /// Reads a prompt in the generation shape, one prefill call and then one
    /// call for each of its last tokens, and reads it again in one call.
    ///
    /// - Parameters:
    ///   - model: The model that reads the prompt.
    ///   - tokens: The prompt, shape `(batch, promptLength)`.
    ///   - promptLength: The number of tokens in the prompt.
    ///   - stepCount: The number of last tokens the run takes one at a time.
    /// - Returns: The logits of both runs and the cache of the stepped run.
    private static func steppedRun(
        model: DeepSeekV4Model, tokens: MLXArray, promptLength: Int, stepCount: Int
    ) throws -> SteppedRun {
        let wholeLogits = DeepSeekV4TestArrays.floats(
            model(tokens, cache: try model.newCache(parameters: nil))[0..., -1, 0...])

        let stepped = try model.newCache(parameters: nil)
        let prefillEnd = promptLength - stepCount
        _ = model(tokens[0..., 0 ..< prefillEnd], cache: stepped)
        var steppedLogits: [Float] = []
        for position in prefillEnd ..< promptLength {
            steppedLogits = DeepSeekV4TestArrays.floats(
                model(tokens[0..., position ..< (position + 1)], cache: stepped)[0..., -1, 0...])
        }
        return SteppedRun(
            steppedLogits: steppedLogits, wholeLogits: wholeLogits, steppedCache: stepped)
    }

    @Test func aLongPromptGivesOneAnswerWhateverThePrefillCutIs() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])

        let whole = try model.newCache(parameters: nil)
        let wholeLogits = DeepSeekV4TestArrays.floats(
            model(tokens, cache: whole)[0..., -1, 0...])

        let cut = try model.newCache(parameters: nil)
        var cutLogits: [Float] = []
        var position = 0
        while position < Self.longPromptLength {
            let end = min(position + Self.prefillCut, Self.longPromptLength)
            cutLogits = DeepSeekV4TestArrays.floats(
                model(tokens[0..., position ..< end], cache: cut)[0..., -1, 0...])
            position = end
        }

        #expect(cutLogits.allSatisfy { $0.isFinite })
        Self.expectSameLogits(
            cutLogits, wholeLogits, actualSource: "cut", expectedSource: "one call")
    }

    /// The number of tokens the decode-step test takes ONE at a time, after a
    /// prefill of the tokens before them.
    ///
    /// It is a multiple of neither compress ratio, thus the single-token calls
    /// carry a chunk of each compressed layer over a call boundary.
    private static let decodeStepCount = 11

    @Test func aPrefillFollowedByOneTokenStepsGivesTheAnswerOfOneCall() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])

        // The generation shape: one prefill call, then one call for each
        // token. `aLongPromptGivesOneAnswerWhateverThePrefillCutIs` cuts the
        // prompt into blocks of 13 and never reads a block of one, which is
        // the block every decode step carries.
        let run = try Self.steppedRun(
            model: model, tokens: tokens, promptLength: Self.longPromptLength,
            stepCount: Self.decodeStepCount)

        #expect(run.steppedLogits.allSatisfy { $0.isFinite })
        Self.expectSameLogits(
            run.steppedLogits, run.wholeLogits, actualSource: "decode steps",
            expectedSource: "one call")
    }

    /// The number of tokens the small-pool decode test reads. The prompt
    /// stands past the sliding window, and its last step pools exactly
    /// `index_topk` chunks of ``overlapLayer``, thus every decode step of
    /// that layer reads its whole pool with no mask.
    private static let smallPoolLength = 12

    /// The number of tokens the small-pool decode test takes one at a time.
    private static let smallPoolStepCount = 3

    /// The `index_topk` of the synthetic checkpoint.
    private static let indexTopK = 3

    @Test func aDecodeStepOverAPoolInsideTheBudgetGivesTheAnswerOfOneCall() throws {
        // A decode step reads every pooled chunk with no mask when the pool
        // holds `index_topk` chunks or fewer. A prefill block of more than
        // one token reads the same chunks through the mask, thus the two
        // must agree.
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.smallPoolLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.smallPoolLength])
        let run = try Self.steppedRun(
            model: model, tokens: tokens, promptLength: Self.smallPoolLength,
            stepCount: Self.smallPoolStepCount)

        let pooledLayer = try #require(run.steppedCache[Self.overlapLayer] as? DeepSeekV4Cache)
        let chunkCount = pooledLayer.attentionChunks.chunkCount
        #expect(
            chunkCount > 0 && chunkCount <= Self.indexTopK,
            "the premise: the pool holds chunks, and no more than the budget")
        Self.expectSameLogits(
            run.steppedLogits, run.wholeLogits, actualSource: "decode steps",
            expectedSource: "one call")
    }

    @Test func aPromptWithNoCacheReadsTheSameSlidingWindowAsACachedRun() throws {
        let model = try Self.loadedModel()
        let tokens = MLXArray((0 ..< Self.longPromptLength).map { Int32($0 % Self.vocabSize) })
            .reshaped([Self.batchSize, Self.longPromptLength])

        // A run that keeps no cache builds its own mask, thus it is where a
        // model that forgot to name its sliding window attends densely.
        let stateless = DeepSeekV4TestArrays.floats(model(tokens, cache: nil)[0..., -1, 0...])
        let cached = DeepSeekV4TestArrays.floats(
            model(tokens, cache: try model.newCache(parameters: nil))[0..., -1, 0...])

        #expect(stateless.allSatisfy { $0.isFinite })
        Self.expectSameLogits(
            stateless, cached, actualSource: "no cache", expectedSource: "a cache")
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
        #expect(DeepSeekV4TestArrays.floats(logits).allSatisfy { $0.isFinite })
        #expect(
            pooledLayer.attentionChunks.chunkCount == expectedChunks,
            "one decode step past a whole chunk must add no chunk")
        #expect(pooledLayer.offset == Self.longPromptLength + 1)
    }

    // MARK: - The arrays the generation loop evaluates

    /// The pooled layer caches a prompt of ``longPromptLength`` tokens leaves
    /// behind: one layer with an indexer, and one layer without.
    ///
    /// - Returns: The cache of ``overlapLayer`` and the cache of ``wideLayer``.
    private static func promptedLayerCaches() throws -> [DeepSeekV4Cache] {
        let model = try loadedModel()
        let tokens = MLXArray((0 ..< longPromptLength).map { Int32($0 % vocabSize) })
            .reshaped([batchSize, longPromptLength])
        let cache = try model.newCache(parameters: nil)
        eval(model(tokens, cache: cache))
        return try [overlapLayer, wideLayer].map { layer in
            try #require(cache[layer] as? DeepSeekV4Cache)
        }
    }

    @Test func theEvaluationStateMakesNoNewArray() throws {
        for layerCache in try Self.promptedLayerCaches() {
            let first = layerCache.evaluationState
            let second = layerCache.evaluationState

            #expect(!first.isEmpty, "the premise: the prompt left arrays to evaluate")
            #expect(first.count == second.count)
            #expect(
                zip(first, second).allSatisfy { $0 === $1 },
                "each read must give the arrays the cache holds, not new ones")
        }
    }

    @Test func theEvaluationStateHoldsEveryArrayTheNextStepReads() throws {
        for layerCache in try Self.promptedLayerCaches() {
            let evaluated = layerCache.evaluationState
            let chunkViews = [layerCache.attentionChunks.chunks, layerCache.indexerChunks?.chunks]
                .compactMap { $0 }
            let kept = layerCache.innerState() + chunkViews

            #expect(!chunkViews.isEmpty, "the premise: the prompt pooled chunks")
            for array in kept {
                #expect(
                    evaluated.contains { $0 === array },
                    "a kept array that stays lazy fails on the decode thread")
            }
        }
    }
}
