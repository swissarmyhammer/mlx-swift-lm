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

    /// The width of one attention head of the synthetic checkpoint.
    private static let attentionHeadDim = 8

    /// The `config.json` of the synthetic checkpoint.
    ///
    /// - Parameter indexHeadWidth: The `index_head_dim` of the checkpoint.
    /// - Returns: The decoded configuration.
    private static func configuration(
        indexHeadWidth: Int = indexHeadDim
    ) throws -> DeepSeekV4Configuration {
        let json = """
            {
              "hidden_size": \(hiddenSize),
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": \(attentionHeadDim),
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
              "index_head_dim": \(indexHeadWidth),
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
        try DeepSeekV4RepeatableWeights.load(
            into: indexer, seed: seed, low: fixtureLow, high: fixtureHigh, order: .flattened)
        return indexer
    }

    /// The arguments one indexer call reads for one block of tokens.
    private struct BlockInputs {
        /// The block input, shape `(batch, tokens, hidden)`.
        let x: MLXArray

        /// The low-rank query, shape `(batch, tokens, qLoraRank)`.
        let queryResidual: MLXArray

        /// The pooled keys, shape `(batch, chunks, indexHeadDim)`.
        let pooledKeys: MLXArray

        /// The rotary position of the indexer layer.
        let rope: DeepSeekV4RoPE

        /// The absolute position of the first token of the block.
        let offset: Int
    }

    /// Builds the repeatable arguments of one indexer call.
    ///
    /// - Parameters:
    ///   - batch: The number of sequences in the block.
    ///   - tokens: The number of tokens in the block.
    ///   - chunks: The number of pooled chunks the compressor holds.
    ///   - offset: The absolute position of the first token of the block.
    ///   - seed: The seed the input filler starts at.
    /// - Returns: The arguments.
    private static func blockInputs(
        batch: Int, tokens: Int, chunks: Int, offset: Int, seed: UInt64
    ) throws -> BlockInputs {
        BlockInputs(
            x: fixture([batch, tokens, hiddenSize], seed: seed),
            queryResidual: fixture([batch, tokens, queryLoraRank], seed: seed + 1),
            pooledKeys: fixture([batch, chunks, indexHeadDim], seed: seed + 2),
            rope: DeepSeekV4RoPE(configuration: try configuration(), layer: indexerLayer),
            offset: offset)
    }

    /// The chunks each query of one block reads: the chunks one indexer
    /// ranks, less the ranked chunks the query may not read.
    ///
    /// The ranking keeps its whole budget, and the attention reads only the
    /// ranked chunks that ``DeepSeekV4Math/pickedChunkVisibility(_:offset:chunkWidth:)``
    /// accepts. This helper reads the two together, as the attention does.
    ///
    /// - Parameters:
    ///   - indexer: The indexer under test.
    ///   - batch: The number of sequences in the block.
    ///   - tokens: The number of tokens in the block.
    ///   - chunks: The number of pooled chunks the compressor holds. It must
    ///     be more than ``indexTopK``.
    ///   - offset: The absolute position of the first token of the block.
    ///   - seed: The seed the input filler starts at.
    /// - Returns: One sorted list of chunk indices for each sequence and
    ///   token.
    private static func readChunks(
        from indexer: DeepSeekV4Indexer,
        batch: Int,
        tokens: Int,
        chunks: Int,
        offset: Int,
        seed: UInt64
    ) throws -> [[[Int]]] {
        let inputs = try blockInputs(
            batch: batch, tokens: tokens, chunks: chunks, offset: offset, seed: seed)
        let ranked = indexer.rankedChunks(
            inputs.x, queryResidual: inputs.queryResidual, pooledKeys: inputs.pooledKeys,
            rope: inputs.rope, offset: inputs.offset)
        let visible = DeepSeekV4Math.pickedChunkVisibility(
            ranked, offset: offset, chunkWidth: indexer.chunkWidth)
        #expect(ranked.shape == [batch, tokens, indexTopK])
        #expect(visible.shape == ranked.shape)
        let indices = DeepSeekV4TestArrays.integers(ranked)
        let flags = DeepSeekV4TestArrays.integers(visible)
        return (0 ..< batch).map { sequence in
            (0 ..< tokens).map { token in
                let row = (sequence * tokens + token) * indexTopK
                return (row ..< (row + indexTopK))
                    .filter { flags[$0] != 0 }
                    .map { Int(indices[$0]) }
                    .sorted()
            }
        }
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

    /// The block offsets the property tests walk. The last one puts every
    /// chunk of ``chunkCounts`` wholly behind the first query, which is the
    /// shape of a decode step.
    private static let offsets = [0, 3, 10, 40]

    /// The pooled-chunk counts the property tests walk. Each one is more
    /// than ``indexTopK``, which is the only case where the indexer ranks. A
    /// pool inside the budget reads every visible chunk, and
    /// `DeepSeekV4MathHelpersTests` holds that rule.
    private static let chunkCounts = [5, 9]

    // MARK: - The causal contract

    @Test func everyReadChunkStandsWhollyBehindItsQuery() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)
        var picked = 0

        for batch in Self.batchSizes {
            for tokens in Self.tokenCounts {
                for offset in Self.offsets {
                    for chunks in Self.chunkCounts {
                        let selected = try Self.readChunks(
                            from: indexer, batch: batch, tokens: tokens, chunks: chunks,
                            offset: offset, seed: Self.weightSeed + UInt64(chunks + tokens))
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
                        let selected = try Self.readChunks(
                            from: indexer, batch: batch, tokens: tokens, chunks: chunks,
                            offset: offset, seed: Self.weightSeed + UInt64(offset + chunks))

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

    /// The number of pooled chunks the first-positions test reads. It is more
    /// than ``indexTopK``, thus the indexer ranks.
    private static let firstBlockChunks = 4

    @Test func theFirstPositionsPickNothingUntilAWholeChunkStandsBehindThem() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)
        let selected = try Self.readChunks(
            from: indexer, batch: 1, tokens: Self.firstBlockTokens,
            chunks: Self.firstBlockChunks, offset: 0, seed: Self.weightSeed)[0]

        #expect(selected[0].isEmpty)
        #expect(selected[1].isEmpty)
        #expect(selected[2].isEmpty)
        #expect(selected[3] == [0])
        #expect(selected[4] == [0])
    }

    // MARK: - The query tiles of the ranking

    /// The block length the tiled-ranking test reads. It holds more than one
    /// tile of ``tiledRankingRowsPerTile`` rows and ends on a partial tile.
    private static let tiledRankingTokens = 9

    /// The pooled-chunk count the tiled-ranking test reads.
    private static let tiledRankingChunks = 9

    /// The offset of the tiled-ranking block. Some of its queries have fewer
    /// than ``indexTopK`` chunks behind them, and some have more.
    private static let tiledRankingOffset = 10

    /// The number of sequences the tiled-ranking test reads.
    private static let tiledRankingBatch = 2

    /// The number of query rows each tile of the tiled-ranking test holds.
    private static let tiledRankingRowsPerTile = 2

    /// The bytes of one float32 score.
    private static let scoreBytes = MemoryLayout<Float>.size

    /// The axis a `(batch, tokens, ...)` tensor holds its tokens on.
    private static let tokenAxis = 1

    /// The axis a `(batch, heads, tokens, width)` tensor holds its heads on.
    private static let headAxis = 1

    /// The axis a `(batch, tokens, heads, width)` tensor holds its heads on.
    private static let batchMajorHeadAxis = 2

    /// The axis a stack of matrices holds its rows on.
    private static let matrixRowAxis = -2

    /// The last axis of a tensor.
    private static let lastAxis = -1

    /// The scores of each pooled chunk against each query of one block, by
    /// the arithmetic the indexer read before it went in query tiles.
    ///
    /// The test states that arithmetic again, step by step, so that a change
    /// of the module cannot make the test agree with it.
    ///
    /// - Parameters:
    ///   - indexer: The indexer under test.
    ///   - inputs: The arguments of the call.
    /// - Returns: The scores, shape `(batch, tokens, chunks)`, as one flat
    ///   list in row-major order.
    private static func referenceScores(
        of indexer: DeepSeekV4Indexer, inputs: BlockInputs
    ) -> [Float] {
        let batch = inputs.x.dim(0)
        let tokens = inputs.x.dim(tokenAxis)
        let projected = indexer.wqB(inputs.queryResidual)
            .reshaped(batch, tokens, indexHeadCount, indexHeadDim)
            .swappedAxes(headAxis, batchMajorHeadAxis)
        let queries = inputs.rope(projected, offset: DeepSeekV4RoPE.position(inputs.offset))
            .asType(.float32)
        let keys = inputs.pooledKeys.asType(.float32)
            .expandedDimensions(axis: headAxis)
            .swappedAxes(lastAxis, matrixRowAxis)
        let perHead = maximum(queries.matmul(keys), 0) * indexer.scale
        let headWeights =
            (indexer.weightsProj(inputs.x).asType(.float32)
            * indexer.headWeightScale)
            .swappedAxes(lastAxis, matrixRowAxis)
            .expandedDimensions(axis: lastAxis)
        return DeepSeekV4TestArrays.floats((perHead * headWeights).sum(axis: headAxis))
    }

    /// The chunks one query reads by the reference rule: the visible chunks
    /// of the highest reference score, at most ``indexTopK`` of them.
    ///
    /// - Parameters:
    ///   - scores: The reference scores of the query, one for each chunk.
    ///   - position: The absolute position of the query.
    /// - Returns: The sorted chunk indices.
    private static func referenceChunks(scores: ArraySlice<Float>, position: Int) -> [Int] {
        let visible = visibleChunkCount(position: position, chunks: scores.count)
        let first = scores.startIndex
        return (0 ..< visible)
            .sorted { scores[first + $0] > scores[first + $1] }
            .prefix(indexTopK)
            .sorted()
    }

    @Test func aTiledRankingReadsTheChunksOfTheHighestReferenceScore() throws {
        let indexer = try Self.loadedIndexer(seed: Self.weightSeed)
        let tokens = Self.tiledRankingTokens
        let chunks = Self.tiledRankingChunks
        let inputs = try Self.blockInputs(
            batch: Self.tiledRankingBatch, tokens: tokens, chunks: chunks,
            offset: Self.tiledRankingOffset, seed: Self.weightSeed)
        let rowBytes = Self.indexHeadCount * chunks * Self.scoreBytes
        let byteLimit = Self.tiledRankingBatch * Self.tiledRankingRowsPerTile * rowBytes

        let ranked = indexer.rankedChunks(
            inputs.x, queryResidual: inputs.queryResidual, pooledKeys: inputs.pooledKeys,
            rope: inputs.rope, offset: inputs.offset, scoreByteLimit: byteLimit)
        let picks = DeepSeekV4TestArrays.integers(ranked)
        let scores = Self.referenceScores(of: indexer, inputs: inputs)

        #expect(ranked.shape == [Self.tiledRankingBatch, tokens, Self.indexTopK])
        for row in 0 ..< Self.tiledRankingBatch * tokens {
            let position = Self.tiledRankingOffset + row % tokens
            let read = picks[(row * Self.indexTopK) ..< ((row + 1) * Self.indexTopK)]
                .map(Int.init)
                .filter { ($0 + 1) * Self.indexerRatio <= position + 1 }
                .sorted()
            let expected = Self.referenceChunks(
                scores: scores[(row * chunks) ..< ((row + 1) * chunks)], position: position)
            #expect(read == expected, "row \(row) at position \(position)")
        }
    }

    /// The query heads of the real checkpoint.
    private static let publishedIndexHeads = 64

    /// The width of one indexer head of the real checkpoint.
    private static let publishedIndexHeadDim = 128

    /// The token count of the compiled-sum parity test.
    private static let paritySumTokens = 7

    /// The chunk count of the compiled-sum parity test.
    private static let paritySumChunks = 33

    /// The seed of the per-head scores of the compiled-sum parity test.
    private static let paritySumSeed: UInt64 = 20_261_014

    @Test func theCompiledHeadSumGivesTheBitsOfTheOpPath() {
        let scale = MLXArray(1 / Float(Self.publishedIndexHeadDim).squareRoot())
        let scores = MLXRandom.normal(
            [1, Self.publishedIndexHeads, Self.paritySumTokens, Self.paritySumChunks],
            key: MLXRandom.key(Self.paritySumSeed))
        let headWeights = MLXRandom.normal(
            [1, Self.publishedIndexHeads, Self.paritySumTokens, 1],
            key: MLXRandom.key(Self.paritySumSeed + 1))

        let compiled = DeepSeekV4Indexer.compiledHeadWeightedSum(scores, headWeights, scale)
        let opPath = DeepSeekV4Indexer.headWeightedSum(
            scores, headWeights: headWeights, scale: scale)

        #expect(compiled.shape == [1, Self.paritySumTokens, Self.paritySumChunks])
        #expect(DeepSeekV4TestArrays.floats(compiled) == DeepSeekV4TestArrays.floats(opPath))
    }

    /// The pooled-chunk count of a 32k context at a compress ratio of 4.
    private static let longContextChunks = 8192

    /// The score bytes one tile may hold in the tile-size test: 1 GiB.
    private static let oneGibibyte = 1 << 30

    /// The rows of one 1 GiB tile of 64 heads over 8192 float32 scores.
    private static let longContextRowsPerTile = 512

    @Test func aScoreTileHoldsTheRowsThatFitTheByteLimit() {
        #expect(
            DeepSeekV4Indexer.queryRowsPerScoreTile(
                batch: 1, heads: Self.publishedIndexHeads, chunks: Self.longContextChunks,
                scoreByteLimit: Self.oneGibibyte) == Self.longContextRowsPerTile)
    }

    @Test func aScoreTileHoldsOneRowWhenOneRowBreaksTheByteLimit() {
        #expect(
            DeepSeekV4Indexer.queryRowsPerScoreTile(
                batch: 1, heads: Self.publishedIndexHeads, chunks: Self.longContextChunks,
                scoreByteLimit: Self.scoreBytes) == 1)
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

    /// The number of tokens the equal-width forward test reads: one whole
    /// chunk more than the indexer keeps, thus the indexer ranks the chunks.
    private static let equalWidthTokenCount = indexerRatio * (indexTopK + 1)

    /// The seed of the input the equal-width forward test reads.
    private static let equalWidthInputSeed: UInt64 = 20_261_013

    /// A layer whose indexer heads are as wide as its attention heads.
    ///
    /// The attention compressor and the indexer compressor then pool chunks
    /// of the same width at the same stride, thus the layer turns the two
    /// kinds of pooled rows with one period table. The test fails when the
    /// layer cannot build, or when the forward gives a value that is not
    /// finite.
    @Test func anIndexerAsWideAsTheAttentionHeadsBuildsAndRunsAForward() throws {
        let configuration = try Self.configuration(indexHeadWidth: Self.attentionHeadDim)
        let attention = DeepSeekV4Attention(
            configuration: configuration, layer: Self.indexerLayer)
        try DeepSeekV4RepeatableWeights.load(
            into: attention, seed: Self.weightSeed, low: Self.fixtureLow, high: Self.fixtureHigh,
            order: .flattened)
        let input = Self.fixture(
            [1, Self.equalWidthTokenCount, Self.hiddenSize], seed: Self.equalWidthInputSeed)

        let output = attention(input, mask: .causal, cache: nil)
        let everyValueIsFinite = DeepSeekV4TestArrays.floats(output).allSatisfy(\.isFinite)

        #expect(attention.indexer != nil)
        #expect(output.shape == [1, Self.equalWidthTokenCount, Self.hiddenSize])
        #expect(everyValueIsFinite)
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
