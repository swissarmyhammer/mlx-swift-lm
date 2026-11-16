// Copyright © 2026 Apple Inc.

// Unit tests of the DFlash2 block drafter: its configuration, its layers, its
// context, its proposals and its quantization at load. The speculation tests
// with a target live in DFlash2SpeculationTests.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Fixture

/// A tiny DFlash2 drafter for the tiny Qwen 3.5 target of ``DFlash2Fixture/target(seed:)``.
enum DFlash2Fixture {

    /// The width of the hidden state of the target and of the drafter.
    static let hiddenSize = 32

    /// The tokens of the vocabulary of the target and of the drafter.
    static let vocabularySize = 64

    /// The decoder layers of the target: linear, attention, linear, attention.
    static let targetLayers = 4

    /// One full-attention layer for each two target layers.
    static let fullAttentionInterval = 2

    /// The first full-attention layer of the target, the last of its interval.
    static let firstAttentionLayer = fullAttentionInterval - 1

    /// The target layers the drafter reads: the first attention layer and the linear layer
    /// after it.
    static let targetLayerIds = [firstAttentionLayer, firstAttentionLayer + 1]

    /// The tokens of a verify the checkpoint was trained for, the anchor included.
    static let trainedBlockSize = 8

    /// The sliding window of the drafter, small so a test fills it.
    static let slidingWindow = 12

    /// How far below the end of the vocabulary the mask token sits.
    static let maskTokenDistanceFromEnd = 2

    /// The token id that fills the block after the anchor.
    static let maskTokenId = vocabularySize - maskTokenDistanceFromEnd

    /// The seed of the target weights.
    static let targetSeed: UInt64 = 35

    /// The seed of the drafter weights.
    static let drafterSeed: UInt64 = 5

    /// The weights of one quantization group of ``fourBit``: the smallest input width of the
    /// tiny drafter, so a group divides every one of them.
    static let fourBitGroupSize = hiddenSize

    /// The bits of one weight of ``fourBit``, those of ``DFlash2DraftModel/fourBit``.
    static let fourBitBits = DFlash2DraftModel.fourBit.bits

    /// A 4-bit quantization whose groups divide every input width of the tiny drafter, as
    /// ``DFlash2DraftModel/fourBit`` divides those of the released one.
    static let fourBit = BaseConfiguration.Quantization(
        groupSize: fourBitGroupSize, bits: fourBitBits)

    /// The JSON of the drafter configuration, in the layout of the z-lab checkpoints.
    ///
    /// - Parameters:
    ///   - layerType: The type of every drafter layer.
    ///   - isCausal: The `is_causal` value.
    ///   - window: The sliding window.
    /// - Returns: The JSON text.
    static func json(
        layerType: String = "sliding_attention", isCausal: String = "false",
        window: Int = slidingWindow
    ) -> String {
        """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": \(isCausal),
          "dflash_config": {
            "block_size": \(trainedBlockSize),
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": \(maskTokenId),
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": \(targetLayerIds)
          },
          "head_dim": 16,
          "hidden_size": \(hiddenSize),
          "intermediate_size": 64,
          "layer_types": ["\(layerType)", "\(layerType)"],
          "max_position_embeddings": 1024,
          "model_type": "qwen3",
          "num_attention_heads": 2,
          "num_hidden_layers": 2,
          "num_key_value_heads": 1,
          "num_target_layers": \(targetLayers),
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
          "sliding_window": \(window),
          "vocab_size": \(vocabularySize)
        }
        """
    }

    /// The drafter configuration.
    static func configuration() throws -> DFlash2Configuration {
        try JSONDecoder.json5().decode(DFlash2Configuration.self, from: Data(json().utf8))
    }

    /// The drafter on seeded random weights.
    ///
    /// - Parameter quantization: The quantization the drafter applies at load, or `nil`.
    static func drafter(
        quantization: BaseConfiguration.Quantization? = nil
    ) throws -> DFlash2DraftModel {
        let configuration = try configuration()
        let drafter = withRandomState(MLXRandom.RandomState(seed: drafterSeed)) {
            DFlash2DraftModel(configuration, quantization: quantization)
        }
        eval(drafter)
        return drafter
    }

    /// The hybrid Qwen 3.5 target on seeded random weights.
    ///
    /// - Parameters:
    ///   - seed: The seed of the target weights.
    ///   - hiddenSize: The width of the hidden state of the target.
    static func target(
        seed: UInt64 = targetSeed, hiddenSize: Int = hiddenSize
    ) throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, hiddenLayers: targetLayers,
                    fullAttentionInterval: fullAttentionInterval, hiddenSize: hiddenSize,
                    vocabularySize: vocabularySize, tieWordEmbeddings: false
                ).utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35TextModel(configuration)
        }
        eval(target)
        return target
    }

    /// The width of one row of target hidden states the drafter reads.
    static var targetHiddenWidth: Int { targetLayerIds.count * hiddenSize }

    /// `count` rows of seeded random target hidden states, shape `[1, count, width]`.
    static func targetHidden(_ count: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([1, count, targetHiddenWidth], key: MLXRandom.key(seed))
    }
}

// MARK: - Configuration

@Suite
struct DFlash2ConfigurationTests {

    @Test("the configuration of the Qwen3.8-27B drafter decodes")
    func decodesReleaseCheckpoint() throws {
        let json = """
            {
              "architectures": ["DFlash2DraftModel"],
              "is_causal": false,
              "dflash_config": {
                "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                "mask_token_id": 248070, "selector_rank": 256, "selector_top_k": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]
              },
              "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
              "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention"],
              "max_position_embeddings": 262144, "model_type": "qwen3",
              "num_attention_heads": 32, "num_hidden_layers": 5, "num_key_value_heads": 8,
              "num_target_layers": 64, "rms_norm_eps": 1e-06,
              "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
              "sliding_window": 2048, "tie_word_embeddings": false, "vocab_size": 248320
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            DFlash2Configuration.self, from: Data(json.utf8))
        #expect(configuration.hiddenSize == 5120)
        #expect(configuration.kvHeads == 8)
        #expect(configuration.ropeTheta == 10_000_000)
        #expect(configuration.contextWindow == 2047)
        #expect(configuration.numTargetLayers == 64)
        #expect(configuration.dflash.blockSize == 8)
        #expect(configuration.dflash.maskTokenId == 248070)
        #expect(configuration.dflash.targetLayerIds == [5, 19, 33, 47, 61])
    }

    @Test("the mlx-community layout reads its window from dflash_config")
    func readsTheMLXCommunityLayout() throws {
        let json = """
            {
              "architectures": ["DFlash2DraftModel"], "model_type": "qwen3",
              "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 6,
              "vocab_size": 248320, "num_attention_heads": 32, "num_key_value_heads": 8,
              "head_dim": 128, "rms_norm_eps": 1e-06, "max_position_embeddings": 262144,
              "rope_parameters": { "rope_type": "default", "rope_theta": 10000000.0 },
              "block_size": 8, "is_causal": false,
              "dflash_config": {
                "mask_token_id": 248077, "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
                "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256,
                "selector_top_k": 16, "use_swa": true, "swa_window_size": 2048
              }
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            DFlash2Configuration.self, from: Data(json.utf8))
        #expect(configuration.dflash.blockSize == 8)
        #expect(configuration.slidingWindow == 2048)
        #expect(configuration.numTargetLayers == nil)
    }

    @Test(
        "a drafter that is causal or not sliding is refused",
        arguments: [
            DFlash2Fixture.json(isCausal: "true"),
            DFlash2Fixture.json(layerType: "full_attention"),
        ])
    func refusesUnsupportedShapes(json: String) {
        #expect(throws: DecodingError.self) {
            try JSONDecoder.json5().decode(DFlash2Configuration.self, from: Data(json.utf8))
        }
    }

    @Test("the identity of a drafter names its quantization")
    func identityNamesQuantization() throws {
        let plain = try DFlash2Fixture.drafter()
        let fourBit = try DFlash2Fixture.drafter(quantization: DFlash2Fixture.fourBit)
        #expect(plain.drafterID != fourBit.drafterID)
    }
}

// MARK: - Geometry

@Suite
struct DFlash2GeometryTests {

    private static let target = DFlash2Geometry(
        layerCount: 40, hiddenSize: 2048, vocabularySize: 248320)
    private static let layerIds = [1, 6, 11, 16, 22, 27, 32, 37]

    private static func drafter(
        _ layers: Int?, hidden: Int = 2048, vocabulary: Int = 248320
    ) -> DFlash2Geometry {
        DFlash2Geometry(layerCount: layers, hiddenSize: hidden, vocabularySize: vocabulary)
    }

    @Test("a drafter of the same shape accepts the target")
    func acceptsTheSameShape() {
        #expect(Self.drafter(40).accepts(Self.target, layerIds: Self.layerIds))
    }

    @Test("a drafter without a depth accepts a target that has its layers")
    func acceptsWithoutDepth() {
        #expect(Self.drafter(nil).accepts(Self.target, layerIds: Self.layerIds))
    }

    @Test(
        "a drafter of another shape refuses the target",
        arguments: [
            DFlash2GeometryTests.drafter(64),
            DFlash2GeometryTests.drafter(40, hidden: 5120),
            DFlash2GeometryTests.drafter(40, vocabulary: 202048),
        ])
    func refusesOtherShapes(drafter: DFlash2Geometry) {
        #expect(!drafter.accepts(Self.target, layerIds: Self.layerIds))
    }

    @Test("a drafter refuses a target without a layer it reads")
    func refusesMissingLayer() {
        #expect(!Self.drafter(nil).accepts(Self.target, layerIds: Self.layerIds + [40]))
    }
}

// MARK: - Layers

@Suite
struct DFlash2LayerTests {

    @Test("the dynamic conv matches a naive two-tap conv")
    func dynamicConvMatchesNaive() throws {
        let (hiddenSize, kernelSize, groupSize, length) = (16, 2, 4, 5)
        let groups = hiddenSize / groupSize
        let conv = DFlash2DynamicConv(
            hiddenSize: hiddenSize, kernelSize: kernelSize, groupSize: groupSize)
        let key = MLXRandom.key(7)
        let base = MLXRandom.normal([2, kernelSize, hiddenSize], key: key)
        let projection = MLXRandom.normal(
            [2 * kernelSize * groups, hiddenSize], key: MLXRandom.key(8))
        try conv.update(
            parameters: ModuleParameters.unflattened([
                ("base_kernel", base), ("kernel_projection.weight", projection),
            ]),
            verify: [])
        let hidden = MLXRandom.normal([1, length, hiddenSize], key: MLXRandom.key(9))

        let (prepared, kernel) = conv.prepare(hidden)
        let dynamic = conv.kernelProjection(hidden).reshaped(
            1, length, 2, kernelSize, groups)

        func naive(baseRow: Int, dynamic: MLXArray) -> MLXArray {
            var output = MLXArray.zeros([1, length, hiddenSize])
            for tap in 0 ..< kernelSize {
                let shifted =
                    tap == 0
                    ? hidden
                    : concatenated(
                        [
                            MLXArray.zeros([1, tap, hiddenSize]),
                            hidden[0..., ..<(length - tap), 0...],
                        ], axis: 1)
                let perChannel = repeated(
                    dynamic[0..., 0..., tap, 0...], count: groupSize, axis: -1)
                output = output + (base[baseRow, tap] + perChannel) * shifted
            }
            return output
        }

        let expectedPrepared = naive(baseRow: 0, dynamic: dynamic[0..., 0..., 0, 0..., 0...])
        let expectedFinished = naive(baseRow: 1, dynamic: kernel)
        #expect(allClose(prepared, expectedPrepared, atol: 1e-4).item(Bool.self))
        #expect(
            allClose(conv.finish(hidden, kernel: kernel), expectedFinished, atol: 1e-4)
                .item(Bool.self))
    }

    @Test("the greedy selector takes the path the anchor points at over the top logit")
    func selectorPrefersCoherentPath() throws {
        // vocabulary 12, top 3, rank 4, hidden 8. Position 0's top logit is a decoy; the
        // anchor's edge makes token 5 win there.
        let configuration = try JSONDecoder.json5().decode(
            DFlash2Configuration.self,
            from: Data(
                DFlash2Fixture.json()
                    .replacingOccurrences(
                        of: "\"selector_top_k\": 4", with: "\"selector_top_k\": 3"
                    )
                    .replacingOccurrences(of: "\"selector_rank\": 8", with: "\"selector_rank\": 4")
                    .replacingOccurrences(
                        of: "\"vocab_size\": \(DFlash2Fixture.vocabularySize)",
                        with: "\"vocab_size\": 12"
                    )
                    .replacingOccurrences(
                        of: "\"hidden_size\": \(DFlash2Fixture.hiddenSize)",
                        with: "\"hidden_size\": 8"
                    )
                    .utf8))
        let selector = DFlash2CandidateSelector(configuration)
        let (vocabulary, rank, hidden) = (12, 4, 8)
        let (anchor, decoy, coherent) = (9, 7, 5)
        var projection = [Float](repeating: 0, count: rank * hidden)
        for index in 0 ..< rank { projection[index * hidden + index] = 1 }
        var predecessor = [Float](repeating: 0, count: vocabulary * rank)
        predecessor[anchor * rank] = 1
        var successor = [Float](repeating: 0, count: vocabulary * rank)
        successor[coherent * rank] = 50
        try selector.update(
            parameters: ModuleParameters.unflattened([
                ("hidden_projection.weight", MLXArray(projection, [rank, hidden])),
                ("predecessor_codebook.weight", MLXArray(predecessor, [vocabulary, rank])),
                ("successor_codebook.weight", MLXArray(successor, [vocabulary, rank])),
            ]),
            verify: [])

        var logits = [Float](repeating: -30, count: 2 * vocabulary)
        logits[decoy] = 10
        logits[coherent] = 9
        logits[1] = 8
        logits[vocabulary + 2] = 10
        logits[vocabulary + 3] = 9
        logits[vocabulary + 4] = 8
        let tokens = selector.select(
            hidden: MLXArray.ones([1, 2, hidden]), logits: MLXArray(logits, [1, 2, vocabulary]),
            anchor: MLXArray([Int32(anchor)]))
        #expect(tokens.asArray(Int32.self) == [Int32(coherent), 2])
    }
}

// MARK: - Context

@Suite
struct DFlash2ContextTests {

    /// The array of an array mask, or `nil` for any other mask mode.
    private static func array(of mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray? {
        if case .array(let array) = mask { array } else { nil }
    }

    @Test("every block query sees every context row inside the window")
    func maskIsNoneInsideTheWindow() {
        let mask = DFlash2DraftModel.visibilityMask(contextRows: 4, width: 3, window: 8)
        #expect(Self.array(of: mask) == nil)
    }

    @Test("a block query hides the context rows its window has passed")
    func maskHidesRowsPastTheWindow() throws {
        // Five context rows at positions 0...4, the block at 5...7, a window of 6:
        // query 5 sees rows 0...4, query 6 rows 1...4 and query 7 rows 2...4.
        let array = try #require(
            Self.array(of: DFlash2DraftModel.visibilityMask(contextRows: 5, width: 3, window: 6)))
        let expected: [[Bool]] = [
            [true, true, true, true, true, true, true, true],
            [false, true, true, true, true, true, true, true],
            [false, false, true, true, true, true, true, true],
        ]
        #expect(array.asArray(Bool.self) == expected.flatMap { $0 })
    }

    @Test("appended context rows keep the drafter state aligned and the window bounded")
    func appendKeepsStateAlignedAndBounded() throws {
        let drafter = try DFlash2Fixture.drafter()
        let target = try DFlash2Fixture.target()
        var state = drafter.makeState(parameters: nil)
        let chunks = [7, 9, 1, 3]
        for (index, count) in chunks.enumerated() {
            let appended = drafter.appendDrafterState(
                target: target, tokens: MLXArray.zeros([1, count], dtype: .int32),
                targetHidden: DFlash2Fixture.targetHidden(count, seed: UInt64(index)),
                positionDeltas: nil, state: &state, sampler: ArgMaxSampler())
            #expect(appended)
        }
        let total = chunks.reduce(0, +)
        #expect(state.nextPosition == total)
        #expect(state.isAligned)
        for case let cache as RotatingKVCache in state.cache {
            let (keys, _) = try #require(cache.logicalView(tail: drafter.contextWindow))
            #expect(keys.dim(2) == drafter.contextWindow)
        }
    }

    @Test("rows of another width are refused")
    func refusesRowsOfAnotherWidth() throws {
        let drafter = try DFlash2Fixture.drafter()
        var state = drafter.makeState(parameters: nil)
        let appended = drafter.appendDrafterState(
            target: try DFlash2Fixture.target(), tokens: MLXArray.zeros([1, 2], dtype: .int32),
            targetHidden: MLXArray.zeros([1, 2, DFlash2Fixture.hiddenSize]),
            positionDeltas: nil, state: &state, sampler: ArgMaxSampler())
        #expect(!appended)
        #expect(state.nextPosition == 0)
    }
}

// MARK: - Proposals

@Suite
struct DFlash2ProposalTests {

    /// The context rows and block widths of a few rounds, past the window.
    private static let rounds: [(rows: Int, width: Int)] = [
        (14, 8), (3, 8), (1, 4), (5, 5), (2, 8),
    ]

    /// Drafts the rounds and gives back each round's drafts.
    private static func draftRounds(
        _ drafter: DFlash2DraftModel, sampler: any LogitSampler = ArgMaxSampler()
    ) throws -> [[Int32]] {
        let target = try DFlash2Fixture.target()
        var state = drafter.makeState(parameters: nil)
        var anchor: Int32 = 7
        return try rounds.enumerated().map { index, round in
            #expect(
                drafter.appendDrafterState(
                    target: target, tokens: MLXArray.zeros([1, round.rows], dtype: .int32),
                    targetHidden: DFlash2Fixture.targetHidden(round.rows, seed: UInt64(index)),
                    positionDeltas: nil, state: &state, sampler: ArgMaxSampler()))
            let drafts = drafter.draftBlock(
                target: target, lastToken: MLXArray([anchor]),
                lastHidden: MLXArray.zeros([1, 1, 1]),
                sharedKV: [:], positionDeltas: nil, queryOffset: state.nextPosition,
                blockSize: round.width, state: &state, sampler: sampler)
            #expect(drafts.shape == [1, round.width - 1])
            let tokens = drafts.asArray(Int32.self)
            anchor = try #require(tokens.first)
            return tokens
        }
    }

    @Test("the compiled proposal drafts what the eager proposal drafts")
    func compiledMatchesEager() throws {
        let compiled = try DFlash2Fixture.drafter()
        let eager = try DFlash2Fixture.drafter()
        eager.compiledTracesEnabled = false

        #expect(try Self.draftRounds(compiled) == Self.draftRounds(eager))
        #expect(compiled.compiledTraceCount > 0)
        #expect(eager.compiledTraceCount == 0)
    }

    @Test("replacing modules drops every compiled trace")
    func invalidationDropsTraces() throws {
        let drafter = try DFlash2Fixture.drafter()
        _ = try Self.draftRounds(drafter)
        drafter.invalidateCompiledTraces()
        #expect(drafter.compiledTraceCount == 0)
    }

    @Test("a draft never takes a suppressed token")
    func draftsSkipSuppressedTokens() throws {
        let allowed = Set(Int32(40) ..< 60)
        let suppressed = Set(0 ..< DFlash2Fixture.vocabularySize).subtracting(allowed.map(Int.init))
        let sampler = SuppressTokensSampler(
            base: ArgMaxSampler(),
            suppressor: try #require(SuppressTokensProcessor(tokenIds: suppressed)))
        let drafts = try Self.draftRounds(try DFlash2Fixture.drafter(), sampler: sampler)
        #expect(Set(drafts.flatMap { $0 }).isSubset(of: allowed))
    }

    @Test("the block size is the trained block, capped by the in-place verify")
    func maximumBlockSizeIsCapped() throws {
        let drafter = try DFlash2Fixture.drafter()
        #expect(
            drafter.maximumBlockSize
                == min(DFlash2Fixture.trainedBlockSize, GatedDeltaDecodeKernel.mostTokens))
    }

    @Test("a target of another shape is refused")
    func refusesAnotherTarget() throws {
        let drafter = try DFlash2Fixture.drafter()
        let wider = try DFlash2Fixture.target(hiddenSize: 2 * DFlash2Fixture.hiddenSize)
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(with: wider)
        }
    }
}

// MARK: - Quantization at load

@Suite
struct DFlash2QuantizationTests {

    /// The drafter's own weights, as a bf16 checkpoint stores them.
    private static func checkpoint() throws -> ModelCheckpoint {
        let source = try DFlash2Fixture.drafter()
        var weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        // The checkpoint stores the selector codebooks as bare tensors.
        for name in ["predecessor_codebook", "successor_codebook"] {
            let key = "candidate_selector.\(name)"
            weights[key] = weights.removeValue(forKey: "\(key).weight")
        }
        return ModelCheckpoint(weights: weights)
    }

    @Test("a 4-bit drafter quantizes its linear layers at load and keeps its codebooks")
    func quantizesLinearLayersAtLoad() throws {
        let drafter = try DFlash2Fixture.drafter(quantization: DFlash2Fixture.fourBit)
        let prepared = try drafter.prepareCheckpoint(Self.checkpoint())
        try applyCheckpoint(prepared, to: drafter)

        let linears = drafter.leafModules().flattened().map(\.1).filter { $0 is Linear }
        #expect(!linears.isEmpty)
        #expect(linears.allSatisfy { $0 is QuantizedLinear })
        #expect(
            drafter.leafModules().flattened().map(\.1).contains { $0 is Embedding })
    }

    @Test("a 4-bit drafter drafts as a drafter quantized after load")
    func quantizedAtLoadDraftsAsQuantizedAfter() throws {
        let atLoad = try DFlash2Fixture.drafter(quantization: DFlash2Fixture.fourBit)
        try applyCheckpoint(atLoad.prepareCheckpoint(Self.checkpoint()), to: atLoad)
        let after = try DFlash2Fixture.drafter()
        quantize(
            model: after, groupSize: DFlash2Fixture.fourBit.groupSize,
            bits: DFlash2Fixture.fourBit.bits
        ) { _, module in module is Linear }
        eval(atLoad, after)

        let target = try DFlash2Fixture.target()
        func drafts(_ drafter: DFlash2DraftModel) -> [Int32] {
            var state = drafter.makeState(parameters: nil)
            _ = drafter.appendDrafterState(
                target: target, tokens: MLXArray.zeros([1, 6], dtype: .int32),
                targetHidden: DFlash2Fixture.targetHidden(6, seed: 1), positionDeltas: nil,
                state: &state, sampler: ArgMaxSampler())
            return drafter.draftBlock(
                target: target, lastToken: MLXArray([Int32(3)]),
                lastHidden: MLXArray.zeros([1, 1, 1]), sharedKV: [:], positionDeltas: nil,
                queryOffset: state.nextPosition, blockSize: DFlash2Fixture.trainedBlockSize,
                state: &state, sampler: ArgMaxSampler()
            ).asArray(Int32.self)
        }
        #expect(drafts(atLoad) == drafts(after))
    }
}
