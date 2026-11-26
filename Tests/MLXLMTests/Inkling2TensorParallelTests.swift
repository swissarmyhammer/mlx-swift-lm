// Copyright © 2026 Apple Inc.
//
// The shard plan of Inkling 2: the heads and expert width each process keeps, the axis each
// parameter splits on, and the errors for counts a group does not divide. No test here needs a
// distributed group of more than one process.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// The text configurations and plans that the suites of this file shard.
enum Inkling2ShardFixture {

    /// The text stack of the Inkling 2 configuration `json`.
    static func textConfiguration(_ json: String) throws -> Inkling2TextConfiguration {
        try JSONDecoder().decode(Inkling2Configuration.self, from: Data(json.utf8))
            .textConfiguration
    }

    /// The text stack of the tiny model of ``Inkling2ModelTests``.
    static func tinyTextConfiguration() throws -> Inkling2TextConfiguration {
        try textConfiguration(Inkling2ModelTests.tinyJSON)
    }

    /// The plan of the tiny model across `groupSize` processes.
    static func tinyPlan(groupSize: Int) throws -> Inkling2TensorParallelPlan {
        try Inkling2TensorParallelPlan(
            configuration: tinyTextConfiguration(), groupSize: groupSize)
    }
}

struct Inkling2TensorParallelPlanTests {

    /// The group size of the plans.
    private static let groupSize = 2

    /// The sliding-window query heads of the test configuration, unlike the full layers' 24.
    private static let slidingAttentionHeads = 16

    /// The sliding-window key and value heads of the test configuration, unlike the full
    /// layers' 8.
    private static let slidingKVHeads = 4

    /// The `text_config` of Inkling 2 Mini with its own sliding-window head counts.
    private static func makeConfiguration() throws -> Inkling2TextConfiguration {
        let json = Inkling2ConfigurationTests.miniJSON
            .replacingOccurrences(
                of: "\"swa_num_attention_heads\": 24",
                with: "\"swa_num_attention_heads\": \(slidingAttentionHeads)"
            )
            .replacingOccurrences(
                of: "\"swa_num_key_value_heads\": 8",
                with: "\"swa_num_key_value_heads\": \(slidingKVHeads)")
        return try Inkling2ShardFixture.textConfiguration(json)
    }

    private static func makePlan(groupSize: Int = groupSize) throws -> Inkling2TensorParallelPlan {
        try Inkling2TensorParallelPlan(configuration: makeConfiguration(), groupSize: groupSize)
    }

    @Test("each process keeps its share of the full-attention query and key-value heads")
    func splitsFullAttentionHeads() throws {
        let heads = try Self.makePlan().heads(sliding: false)
        #expect(heads == (heads: 12, kvHeads: 4))
    }

    @Test("each process keeps its share of the sliding-window query and key-value heads")
    func splitsSlidingAttentionHeads() throws {
        let heads = try Self.makePlan().heads(sliding: true)
        #expect(
            heads
                == (
                    heads: Self.slidingAttentionHeads / Self.groupSize,
                    kvHeads: Self.slidingKVHeads / Self.groupSize
                ))
    }

    @Test("each process keeps its share of the inner width of every expert")
    func splitsExpertWidth() throws {
        #expect(try Self.makePlan().expertWidth == 576)
    }

    @Test("the static taps of a dynamic convolution split on their channel axis")
    func splitsDynamicStaticTapsOnChannels() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: false)
        #expect(
            TensorParallel.split(of: "k_sconv.static_weight", in: splits)
                == TensorParallel.Split(axis: 1, segments: .count(1)))
    }

    @Test("the tap projection of a dynamic convolution splits by key-value head")
    func splitsDynamicTapProjection() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: false)
        #expect(
            TensorParallel.split(of: "v_sconv.weight_proj.scales", in: splits)
                == TensorParallel.Split(axis: 0, segments: .count(1)))
    }

    @Test("a static convolution splits its channels by key-value head")
    func splitsStaticConvolutionChannels() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: true)
        #expect(
            TensorParallel.split(of: "k_sconv.conv1d.weight", in: splits)
                == TensorParallel.Split(axis: 0, segments: .count(1)))
    }

    @Test("the query clip scales split by query head")
    func splitsQueryClipScales() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: true)
        #expect(
            TensorParallel.split(of: "q_l2_clip.scale", in: splits)
                == TensorParallel.Split(axis: 0, segments: .count(1)))
    }

    @Test("the relative logit table, which every head shares, stays whole")
    func keepsRelativeLogitTable() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: false)
        #expect(TensorParallel.split(of: "rel_logits_proj.proj", in: splits) == nil)
    }

    @Test("the output projection is not split in place")
    func keepsOutputProjection() throws {
        let splits = try Self.makePlan().attentionSplits(sliding: true)
        #expect(TensorParallel.split(of: "o_proj.weight", in: splits) == nil)
    }

    @Test("the interleaved gate and up rows of each expert split in one contiguous run")
    func splitsGateUpRows() {
        #expect(
            TensorParallel.split(
                of: "gate_up_proj.weight", in: Inkling2TensorParallelPlan.expertSplits)
                == TensorParallel.Split(axis: 1, segments: .count(1)))
    }

    @Test("the down projection of each expert splits its inputs")
    func splitsDownInputs() {
        #expect(
            TensorParallel.split(
                of: "down_proj.scales", in: Inkling2TensorParallelPlan.expertSplits)
                == TensorParallel.Split(axis: 2, segments: .count(1)))
    }

    @Test("the global scale of each expert stays whole")
    func keepsExpertGlobalScale() {
        #expect(
            TensorParallel.split(
                of: "gate_up_proj.global_scale", in: Inkling2TensorParallelPlan.expertSplits)
                == nil)
    }

    @Test("key and value heads that the group does not divide throw")
    func throwsForIndivisibleKVHeads() throws {
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "kvHeads", count: 8, groupSize: 3)
        ) {
            try Self.makePlan(groupSize: 3)
        }
    }
}

/// The plan checks of a mixed-quantization layer, as in Inkling 2 Mini oQ4e: 8-bit affine
/// attention in groups of 64 beside 4-bit experts.
struct Inkling2WholeShareTests {

    /// The processes of the plan.
    private static let groupSize = 2

    /// The 8-bit affine format of the attention.
    private static let attentionBits = 8
    private static let attentionGroupSize = 64

    /// A group wider than the share of the output projection's 128 inputs on each process.
    private static let wideGroupSize = 128

    /// A 4-bit affine expert format of groups as wide as the tiny expert width, 32.
    private static let wideExpertFormat = Inkling2ExpertFormat(
        mode: .affine, bits: 4, groupSize: 32, hasGlobalScale: false)

    /// The decoder layers of a tiny model, with NVFP4 experts in groups of 16, and every
    /// attention projection whose inputs `groupSize` divides quantized to 8-bit affine.
    private static func makeLayers(attentionGroupSize groupSize: Int) throws
        -> [Inkling2DecoderLayer]
    {
        let model = try Inkling2ModelTests.makeModel()
        let layers = model.loraLayers.compactMap { $0 as? Inkling2DecoderLayer }
        for layer in layers {
            quantize(model: layer.attention) { _, module in
                guard let linear = module as? Linear, linear.weight.dim(1) % groupSize == 0
                else { return nil }
                return (groupSize: groupSize, bits: attentionBits, mode: .affine)
            }
        }
        return layers
    }

    private static func makePlan() throws -> Inkling2TensorParallelPlan {
        try Inkling2ShardFixture.tinyPlan(groupSize: groupSize)
    }

    @Test("8-bit affine attention beside nvfp4 experts splits across two processes")
    func acceptsMixedQuantization() throws {
        let plan = try Self.makePlan()
        let layers = try Self.makeLayers(attentionGroupSize: Self.attentionGroupSize)
        #expect(throws: Never.self) {
            for layer in layers { try layer.requireWholeShares(plan) }
        }
    }

    @Test("attention groups wider than each process's share throw, naming the output projection")
    func rejectsWideAttentionGroups() throws {
        let plan = try Self.makePlan()
        let layer = try #require(
            try Self.makeLayers(attentionGroupSize: Self.wideGroupSize).first)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "self_attn.o_proj quantization groups (affine, 8-bit, 128 inputs each)",
                count: 1, groupSize: Self.groupSize)
        ) {
            try layer.requireWholeShares(plan)
        }
    }

    @Test("expert groups wider than each process's share throw, naming the down projection")
    func rejectsWideExpertGroups() throws {
        let plan = try Self.makePlan()
        let layer = try #require(
            try Self.makeLayers(attentionGroupSize: Self.attentionGroupSize).first)
        let experts = Inkling2Experts(
            try Inkling2ShardFixture.tinyTextConfiguration(), gateUp: Self.wideExpertFormat,
            down: Self.wideExpertFormat)
        try layer.mlp.update(
            modules: ModuleChildren(values: ["experts": .value(experts)]), verify: [])
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity:
                    "mlp.experts.down_proj quantization groups (affine, 4-bit, 32 inputs each)",
                count: 1, groupSize: Self.groupSize)
        ) {
            try layer.requireWholeShares(plan)
        }
    }
}

struct TensorParallelSplitTests {

    /// The splits of the tests.
    private static let splits = [
        "q_proj": TensorParallel.Split(axis: 0, segments: .count(1)),
        "conv.static_weight": TensorParallel.Split(axis: 1, segments: .count(1)),
    ]

    @Test("a key that names a child module splits each of its parameters")
    func childModuleKeyMatches() {
        #expect(TensorParallel.split(of: "q_proj.scales", in: Self.splits) == Self.splits["q_proj"])
    }

    @Test("a key that names a parameter splits that parameter")
    func parameterKeyMatches() {
        #expect(
            TensorParallel.split(of: "conv.static_weight", in: Self.splits)
                == Self.splits["conv.static_weight"])
    }

    @Test("a parameter that no key names stays whole")
    func unnamedParameterStaysWhole() {
        #expect(TensorParallel.split(of: "q_proj_extra.weight", in: Self.splits) == nil)
    }

    /// The depth of the nested keys of ``nestedSplits``, so an arbitrary pick of a matching key
    /// finds the deepest one rarely.
    private static let nestingDepth = 8

    /// Keys `a`, `a.b`, `a.b.b`, ..., each splitting on the axis of its depth.
    private static let nestedSplits = Dictionary(
        uniqueKeysWithValues: (0 ..< nestingDepth).map { depth in
            (
                (["a"] + Array(repeating: "b", count: depth)).joined(separator: "."),
                TensorParallel.Split(axis: depth, segments: .count(1))
            )
        })

    @Test("of keys that nest, the deepest that names the parameter decides its split")
    func deepestNestedKeyDecides() {
        let path = (["a"] + Array(repeating: "b", count: Self.nestingDepth)).joined(separator: ".")
        #expect(
            TensorParallel.split(of: path, in: Self.nestedSplits)?.axis == Self.nestingDepth - 1)
    }
}

struct Inkling2SingleProcessShardTests {

    /// The prompt tokens.
    private static let tokens = [1, 5, 3, 7]

    @Test("sharding across a group of one process keeps the logits")
    func groupOfOneKeepsLogits() throws {
        let model = try Inkling2ModelTests.makeModel(bfloat16: true)
        let input = MLXArray(Self.tokens).expandedDimensions(axis: 0)
        let before = model(input, cache: nil).asType(.float32).asArray(Float.self)

        try model.shard(group: MLXDistributed.initialize())

        #expect(model(input, cache: nil).asType(.float32).asArray(Float.self) == before)
    }

    @Test("the key-value heads of each layer visit are the heads its layer keeps")
    func kvHeadsFollowKeptHeads() throws {
        let model = try Inkling2ModelTests.makeModel()
        let groupSize = 2
        for case let layer as Inkling2DecoderLayer in model.loraLayers {
            layer.attention.adoptShardedHeads(
                heads: layer.attention.heads / groupSize,
                kvHeads: layer.attention.kvHeads / groupSize)
        }
        let text = try Inkling2ShardFixture.tinyTextConfiguration()

        #expect(
            model.kvHeads
                == text.layerVisits.map {
                    (text.isSliding($0) ? text.slidingKVHeads : text.kvHeads) / groupSize
                })
    }
}

/// Each process's share of a layer, run in one process: the partial outputs of the shares must
/// sum to the output of the whole layer, which the group's sum computes.
struct Inkling2ShareSumTests {

    /// The processes the shares stand for.
    private static let groupSize = 2

    /// The seed of the layer input.
    private static let inputSeed: UInt64 = 7

    /// The tokens of the layer input.
    private static let length = 5

    /// The largest difference float32 sums in another order leave.
    private static let tolerance: Float = 1e-4

    /// The decoder layer `index` of a tiny float32 model.
    private static func layer(_ index: Int) throws -> Inkling2DecoderLayer {
        let layers = try Inkling2ModelTests.makeModel().loraLayers
        return try #require(layers[index] as? Inkling2DecoderLayer)
    }

    /// A layer `index` with the weights of `source`, whatever the shared random state holds.
    private static func copy(_ source: Inkling2DecoderLayer, index: Int) throws
        -> Inkling2DecoderLayer
    {
        let layer = try layer(index)
        _ = try layer.update(parameters: source.parameters(), verify: .all)
        return layer
    }

    private static func makePlan() throws -> Inkling2TensorParallelPlan {
        try Inkling2ShardFixture.tinyPlan(groupSize: groupSize)
    }

    /// Keeps part `rank` of each parameter of `module` that `splits` names.
    private static func keepShare(
        of module: Module, _ splits: [String: TensorParallel.Split], rank: Int
    ) {
        TensorParallelShares.keep(rank: rank, of: groupSize, in: module, splits)
    }

    private static func layerInput() -> MLXArray {
        MLXRandom.normal([1, length, 64], key: MLXRandom.key(inputSeed))
    }

    private static func attend(_ attention: Inkling2Attention, _ x: MLXArray) -> MLXArray {
        attention(
            queryInput: x, keyInput: x, valueInput: x, cache: nil, convolutions: [nil, nil],
            convolutionMask: nil, masks: Inkling2DecodeMasks()
        ).0
    }

    @Test(
        "the attention shares of both layer kinds sum to the whole attention",
        arguments: [0, 3])
    func attentionSharesSum(layerIndex: Int) throws {
        let plan = try Self.makePlan()
        let x = Self.layerInput()
        let whole = try Self.layer(layerIndex)
        let expected = Self.attend(whole.attention, x)

        var sum = MLXArray.zeros(like: expected)
        for rank in 0 ..< Self.groupSize {
            let attention = try Self.copy(whole, index: layerIndex).attention
            Self.keepShare(
                of: attention, plan.attentionSplits(sliding: whole.isSliding), rank: rank)
            let heads = plan.heads(sliding: whole.isSliding)
            attention.adoptShardedHeads(heads: heads.heads, kvHeads: heads.kvHeads)
            let output = attention.outputProj.weight.split(parts: Self.groupSize, axis: -1)[rank]
            try attention.update(
                modules: ModuleChildren(values: ["o_proj": .value(Linear(weight: output))]),
                verify: [])
            sum = sum + Self.attend(attention, x)
        }

        #expect(maxAbsoluteDifference(sum, expected) < Self.tolerance)
    }

    @Test("the expert shares sum to the whole mixture of experts")
    func expertSharesSum() throws {
        let x = Self.layerInput()
        let whole = try Self.layer(0)
        let expected = whole.mlp(x)

        var sum = MLXArray.zeros(like: expected)
        for rank in 0 ..< Self.groupSize {
            let mlp = try Self.copy(whole, index: 0).mlp
            Self.keepShare(of: mlp.experts, Inkling2TensorParallelPlan.expertSplits, rank: rank)
            sum = sum + mlp(x)
        }

        #expect(maxAbsoluteDifference(sum, expected) < Self.tolerance)
    }
}
