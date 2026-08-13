// Copyright © 2026 Apple Inc.
//
// The causal contract of ``DeepSeekV4Indexer``.
//
// The indexer picks the pooled chunks each query reads. A chunk covers a run
// of `chunkWidth` raw positions, thus a query may read a chunk only after the
// whole chunk stands behind it. An off-by-one here lets a query read a token
// of its own future, which no later mask takes back, thus the property test
// below is the guard of this file.
//
// The property test states the rule again in plain Swift rather than calling
// the production helper, thus a wrong rule in the module cannot make the test
// agree with it.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct DeepSeekV4IndexerTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - The synthetic checkpoint

    /// The width of the residual stream of the synthetic checkpoint.
    private static let hiddenSize = 8

    /// The rank of the low-rank query projection of the synthetic checkpoint.
    private static let queryLoraRank = 6

    /// The number of indexer heads of the synthetic checkpoint.
    private static let indexHeadCount = 2

    /// The width of one indexer head of the synthetic checkpoint.
    private static let indexHeadDim = 4

    /// The number of chunks the synthetic indexer keeps for each query.
    private static let indexTopK = 3

    /// The number of trailing head dimensions that take a position.
    private static let ropeDim = 2

    /// The number of decoder layers of the synthetic checkpoint.
    private static let layerCount = 4

    /// The compress ratio of a layer that holds an indexer. It is also the
    /// number of raw positions one pooled chunk covers.
    private static let indexerRatio = 4

    /// The index of a layer whose compress ratio is ``indexerRatio``, thus a
    /// layer that holds an indexer.
    private static let indexerLayer = 2

    /// The index of a layer whose compress ratio is more than 0 and is not
    /// ``indexerRatio``, thus a layer that holds a compressor and no indexer.
    private static let compressorOnlyLayer = 3

    /// The index of a layer whose compress ratio is 0, thus a layer that holds
    /// neither a compressor nor an indexer.
    private static let plainLayer = 0

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// - Returns: The decoded configuration.
    private static func configuration() throws -> DeepSeekV4Configuration {
        let json = """
            {
              "hidden_size": \(hiddenSize),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 8,
              "qk_rope_head_dim": \(ropeDim),
              "q_lora_rank": \(queryLoraRank),
              "o_groups": 2,
              "o_lora_rank": 4,
              "num_hidden_layers": \(layerCount),
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "compress_rope_theta": 160000.0,
              "compress_ratios": [0, 0, \(indexerRatio), 128],
              "index_n_heads": \(indexHeadCount),
              "index_head_dim": \(indexHeadDim),
              "index_topk": \(indexTopK)
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
    private static let weightSeed: UInt64 = 20_260_813

    /// Builds a repeatable random tensor.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - seed: The seed of this tensor.
    /// - Returns: The tensor, in float32.
    private static func fixture(_ shape: [Int], seed: UInt64) -> MLXArray {
        MLXRandom.uniform(low: fixtureLow, high: fixtureHigh, shape, key: MLXRandom.key(seed))
    }

    /// Builds one indexer of the synthetic checkpoint and loads a repeatable
    /// random weight into every parameter it declares.
    ///
    /// The load runs through `update(parameters:verify:[.all])`, which is the
    /// verification `MLXLMCommon.loadWeights` applies, thus this builder also
    /// states that the indexer declares exactly the parameters a checkpoint
    /// fills.
    ///
    /// - Parameter seed: The seed the weight filler starts at.
    /// - Returns: The loaded indexer.
    private static func loadedIndexer(seed: UInt64) throws -> DeepSeekV4Indexer {
        let indexer = DeepSeekV4Indexer(configuration: try configuration(), layer: indexerLayer)
        var checkpoint: [String: MLXArray] = [:]
        var next = seed
        for (key, value) in indexer.parameters().flattened() {
            next += 1
            checkpoint[key] = fixture(value.shape, seed: next)
        }
        try indexer.update(parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(indexer)
        return indexer
    }

    /// The chunks one indexer picks for one block of tokens.
    ///
    /// - Parameters:
    ///   - indexer: The indexer under test.
    ///   - batch: The number of sequences in the block.
    ///   - tokens: The number of tokens in the block.
    ///   - chunks: The number of pooled chunks the compressor holds.
    ///   - offset: The absolute position of the first token of the block.
    ///   - seed: The seed the input filler starts at.
    /// - Returns: The selection mask, shape `(batch, 1, tokens, chunks)`.
    private static func selection(
        from indexer: DeepSeekV4Indexer,
        batch: Int,
        tokens: Int,
        chunks: Int,
        offset: Int,
        seed: UInt64
    ) throws -> MLXArray {
        let angles = DeepSeekV4RoPE(configuration: try configuration(), layer: indexerLayer)
            .cosSin(offset: offset, length: tokens)
        let mask = indexer(
            fixture([batch, tokens, hiddenSize], seed: seed),
            queryResidual: fixture([batch, tokens, queryLoraRank], seed: seed + 1),
            pooledKeys: fixture([batch, chunks, indexHeadDim], seed: seed + 2),
            cos: angles.cos.expandedDimensions(axes: [0, 1]),
            sin: angles.sin.expandedDimensions(axes: [0, 1]),
            offset: offset)
        eval(mask)
        return mask
    }

    /// The number of chunks a query at one absolute position may read.
    ///
    /// This states the block-causal rule again, in plain Swift, so that the
    /// tests below never measure the module against itself.
    ///
    /// - Parameters:
    ///   - position: The absolute position of the query.
    ///   - chunks: The number of pooled chunks the compressor holds.
    /// - Returns: The number of chunks that stand wholly behind the query.
    private static func visibleChunkCount(position: Int, chunks: Int) -> Int {
        (0 ..< chunks).filter { ($0 + 1) * indexerRatio <= position + 1 }.count
    }

    /// The shapes the property tests walk.
    private static let batchSizes = [1, 2]

    /// The block lengths the property tests walk.
    private static let tokenCounts = [1, 4, 9]

    /// The block offsets the property tests walk.
    private static let offsets = [0, 3, 10]

    /// The pooled-chunk counts the property tests walk. The list crosses
    /// ``indexTopK`` in both directions, thus it reads both the ranked path
    /// and the every-chunk-fits path.
    private static let chunkCounts = [1, 3, 5, 9]

    /// The chunks one selection mask picked, by sequence and by token.
    ///
    /// - Parameters:
    ///   - mask: The selection mask, shape `(batch, 1, tokens, chunks)`.
    ///   - batch: The number of sequences in the block.
    ///   - tokens: The number of tokens in the block.
    ///   - chunks: The number of pooled chunks the compressor holds.
    /// - Returns: One list of chunk indices for each sequence and token.
    private static func pickedChunks(
        _ mask: MLXArray, batch: Int, tokens: Int, chunks: Int
    ) -> [[[Int]]] {
        let flags = mask.asType(.int32).reshaped(-1).asArray(Int32.self)
        return (0 ..< batch).map { sequence in
            (0 ..< tokens).map { token in
                (0 ..< chunks).filter { chunk in
                    flags[(sequence * tokens + token) * chunks + chunk] != 0
                }
            }
        }
    }

    // MARK: - The causal contract

    @Test func everyPickedChunkStandsWhollyBehindItsQuery() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)
        var picked = 0

        for batch in Self.batchSizes {
            for tokens in Self.tokenCounts {
                for offset in Self.offsets {
                    for chunks in Self.chunkCounts {
                        let mask = try Self.selection(
                            from: indexer, batch: batch, tokens: tokens, chunks: chunks,
                            offset: offset, seed: Self.weightSeed + UInt64(chunks + tokens))
                        #expect(mask.shape == [batch, 1, tokens, chunks])

                        let selected = Self.pickedChunks(
                            mask, batch: batch, tokens: tokens, chunks: chunks)
                        for sequence in 0 ..< batch {
                            for token in 0 ..< tokens {
                                let position = offset + token
                                for chunk in selected[sequence][token] {
                                    picked += 1
                                    #expect(
                                        (chunk + 1) * Self.indexerRatio - 1 <= position,
                                        "chunk \(chunk) reaches past query position \(position)")
                                }
                            }
                        }
                    }
                }
            }
        }

        #expect(picked > 0, "no shape picked a chunk, thus the test proved nothing")
    }

    @Test func theCountIsTheTopKBudgetOrTheVisibleCount() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)

        for batch in Self.batchSizes {
            for tokens in Self.tokenCounts {
                for offset in Self.offsets {
                    for chunks in Self.chunkCounts {
                        let mask = try Self.selection(
                            from: indexer, batch: batch, tokens: tokens, chunks: chunks,
                            offset: offset, seed: Self.weightSeed + UInt64(offset + chunks))
                        let selected = Self.pickedChunks(
                            mask, batch: batch, tokens: tokens, chunks: chunks)

                        for sequence in 0 ..< batch {
                            for token in 0 ..< tokens {
                                let visible = Self.visibleChunkCount(
                                    position: offset + token, chunks: chunks)
                                #expect(
                                    selected[sequence][token].count
                                        == min(Self.indexTopK, visible),
                                    "position \(offset + token) of \(chunks) chunks")
                            }
                        }
                    }
                }
            }
        }
    }

    /// The number of tokens the first-positions test reads. It reaches one
    /// position past the end of the first chunk.
    private static let firstBlockTokens = 5

    /// The number of pooled chunks the first-positions test reads.
    private static let firstBlockChunks = 4

    @Test func theFirstPositionsPickNothingUntilAWholeChunkStandsBehindThem() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)
        let mask = try Self.selection(
            from: indexer, batch: 1, tokens: Self.firstBlockTokens,
            chunks: Self.firstBlockChunks, offset: 0, seed: Self.weightSeed)
        let selected = Self.pickedChunks(
            mask, batch: 1, tokens: Self.firstBlockTokens, chunks: Self.firstBlockChunks)[0]

        #expect(selected[0].isEmpty)
        #expect(selected[1].isEmpty)
        #expect(selected[2].isEmpty)
        #expect(selected[3] == [0])
        #expect(selected[4] == [0])
    }

    // MARK: - The projections the checkpoint fills

    @Test func theProjectionsTakeTheirWidthFromTheIndexKeys() throws {
        let indexer = DeepSeekV4Indexer(
            configuration: try Self.configuration(), layer: Self.indexerLayer)

        #expect(
            indexer.wqB.weight.shape
                == [Self.indexHeadCount * Self.indexHeadDim, Self.queryLoraRank])
        #expect(indexer.weightsProj.weight.shape == [Self.indexHeadCount, Self.hiddenSize])
    }

    // MARK: - The layers that hold an indexer

    /// The published `config.json` of `mlx-community/DeepSeek-V4-Flash-4bit`.
    ///
    /// - Returns: The decoded configuration.
    private static func publishedConfiguration() throws -> DeepSeekV4Configuration {
        let url = try #require(
            Bundle.module.url(forResource: "DeepSeek-V4-Flash-4bit-config", withExtension: "json"))
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(contentsOf: url))
    }

    /// The first layer of the published checkpoint that holds an indexer.
    private static let firstPublishedIndexerLayer = 2

    /// The last layer of the published checkpoint that holds an indexer.
    private static let lastPublishedIndexerLayer = 42

    /// The number of layers of the published checkpoint that hold an indexer.
    private static let publishedIndexerLayerCount = 21

    @Test func thePublishedCheckpointHoldsAnIndexerOnTheEvenLayersFromTwoUp() throws {
        let configuration = try Self.publishedConfiguration()
        let expected = stride(
            from: Self.firstPublishedIndexerLayer,
            through: Self.lastPublishedIndexerLayer,
            by: 2)
        let indexerLayers = (0 ..< configuration.numHiddenLayers).filter {
            configuration.hasIndexer(layer: $0)
        }

        #expect(indexerLayers == Array(expected))
        #expect(indexerLayers.count == Self.publishedIndexerLayerCount)
    }

    @Test func aCompressorLayerWithoutTheIndexerRatioHoldsNoIndexer() throws {
        let configuration = try Self.configuration()

        #expect(configuration.hasIndexer(layer: Self.indexerLayer))
        #expect(!configuration.hasIndexer(layer: Self.compressorOnlyLayer))
        #expect(configuration.hasCompressor(layer: Self.compressorOnlyLayer))
        #expect(!configuration.hasIndexer(layer: Self.plainLayer))
        #expect(!configuration.hasIndexer(layer: Self.layerCount))
    }

    @Test func attentionHoldsAnIndexerOnlyOnAnIndexerLayer() throws {
        let configuration = try Self.configuration()

        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.indexerLayer)
                .indexer != nil)
        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.compressorOnlyLayer)
                .indexer == nil)
        #expect(
            DeepSeekV4Attention(configuration: configuration, layer: Self.plainLayer)
                .indexer == nil)
    }

    // MARK: - The load filter

    /// A one-value array, which stands in for a checkpoint tensor whose values
    /// no test reads.
    private static func marker(_ value: Float) -> MLXArray {
        MLXArray([value])
    }

    @Test func sanitizeKeepsTheIndexerProjections() throws {
        let model = DeepSeekV4Model(try Self.configuration())
        let layer = Self.indexerLayer
        let sanitized = model.sanitize(weights: [
            "layers.\(layer).attn.indexer.wq_b.weight": Self.marker(1),
            "layers.\(layer).attn.indexer.weights_proj.weight": Self.marker(2),
            "layers.\(layer).attn.wq_a.weight": Self.marker(3),
        ])

        #expect(sanitized["model.layers.\(layer).attn.indexer.wq_b.weight"] != nil)
        #expect(sanitized["model.layers.\(layer).attn.indexer.weights_proj.weight"] != nil)
        #expect(sanitized["model.layers.\(layer).attn.wq_a.weight"] != nil)
    }
}
