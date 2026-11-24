// Copyright © 2026 Apple Inc.
//
// The shard plan of a Qwen 3.5 model: the head counts each process keeps, the
// segments of the fused gated delta input projection, and the typed error a
// group size that does not divide a count raises. No test here needs a
// distributed group of more than one process.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

struct TensorParallelShareTests {

    @Test("a count that the group size divides splits evenly")
    func shareDividesCount() throws {
        #expect(try TensorParallel.share(of: 8, named: "heads", across: 2) == 4)
    }

    @Test("a count that the group size does not divide throws, naming the count")
    func shareThrowsForIndivisibleCount() {
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "kvHeads", count: 4, groupSize: 3)
        ) {
            try TensorParallel.share(of: 4, named: "kvHeads", across: 3)
        }
    }

    @Test("the indivisible error states the reason")
    func indivisibleDescription() {
        let error = TensorParallelShardingError.indivisible(
            quantity: "kvHeads", count: 4, groupSize: 3)
        #expect(error.description == "kvHeads 4 not divisible by 3")
    }
}

struct Qwen35TensorParallelPlanTests {

    /// The group size of the plans.
    private static let groupSize = 2

    /// The full-attention query heads of the test configuration.
    private static let attentionHeads = 8

    /// The full-attention key and value heads of the test configuration.
    private static let kvHeads = 2

    /// The gated delta key heads of the test configuration.
    private static let linearKeyHeads = 4

    /// The gated delta value heads of the test configuration.
    private static let linearValueHeads = 8

    /// The width of each gated delta key head of the test configuration.
    private static let linearKeyHeadDim = 16

    /// A dense configuration with head counts a group of two divides.
    private static func makeConfiguration() throws -> Qwen35TextConfiguration {
        var configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 0).utf8))
        configuration.attentionHeads = attentionHeads
        configuration.kvHeads = kvHeads
        configuration.linearNumKeyHeads = linearKeyHeads
        configuration.linearNumValueHeads = linearValueHeads
        configuration.linearKeyHeadDim = linearKeyHeadDim
        return configuration
    }

    @Test("each process keeps its share of the full-attention query heads")
    func splitsAttentionHeads() throws {
        let plan = try Qwen35TensorParallelPlan(
            configuration: Self.makeConfiguration(), groupSize: Self.groupSize)
        #expect(plan.attentionHeads == Self.attentionHeads / Self.groupSize)
    }

    @Test("each process keeps its share of the full-attention key and value heads")
    func splitsKVHeads() throws {
        let plan = try Qwen35TensorParallelPlan(
            configuration: Self.makeConfiguration(), groupSize: Self.groupSize)
        #expect(plan.kvHeads == Self.kvHeads / Self.groupSize)
    }

    @Test("each process keeps its share of the gated delta key heads")
    func splitsLinearKeyHeads() throws {
        let plan = try Qwen35TensorParallelPlan(
            configuration: Self.makeConfiguration(), groupSize: Self.groupSize)
        #expect(plan.linearKeyHeads == Self.linearKeyHeads / Self.groupSize)
    }

    @Test("each process keeps its share of the gated delta value heads")
    func splitsLinearValueHeads() throws {
        let plan = try Qwen35TensorParallelPlan(
            configuration: Self.makeConfiguration(), groupSize: Self.groupSize)
        #expect(plan.linearValueHeads == Self.linearValueHeads / Self.groupSize)
    }

    @Test("the fused q, k, v input projection splits at the end of q and of k")
    func segmentsGatedDeltaInput() throws {
        let plan = try Qwen35TensorParallelPlan(
            configuration: Self.makeConfiguration(), groupSize: Self.groupSize)
        let keyDim = Self.linearKeyHeads * Self.linearKeyHeadDim
        #expect(plan.gatedDeltaInputSegments == .indices([keyDim, 2 * keyDim]))
    }

    @Test("key and value heads that the group does not divide throw")
    func throwsForIndivisibleKVHeads() throws {
        var configuration = try Self.makeConfiguration()
        let groupSize = 3
        configuration.attentionHeads = 2 * groupSize
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "kvHeads", count: Self.kvHeads, groupSize: groupSize)
        ) {
            try Qwen35TensorParallelPlan(configuration: configuration, groupSize: groupSize)
        }
    }

    @Test("an MLP width that the group does not divide throws")
    func throwsForIndivisibleIntermediateSize() throws {
        var configuration = try Self.makeConfiguration()
        configuration.intermediateSize = 33
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "intermediateSize", count: 33, groupSize: Self.groupSize)
        ) {
            try Qwen35TensorParallelPlan(configuration: configuration, groupSize: Self.groupSize)
        }
    }

    @Test("a mixture-of-experts model is not supported yet")
    func rejectsMixtureOfExperts() throws {
        var configuration = try Self.makeConfiguration()
        configuration.numExperts = 4
        #expect(
            throws: TensorParallelShardingError.unsupported(feature: "mixture-of-experts layers")
        ) {
            try Qwen35TensorParallelPlan(configuration: configuration, groupSize: Self.groupSize)
        }
    }
}

struct TensorParallelStockProjectionTests {

    /// A projection subclass whose weights do not split as a stock projection's do.
    private final class ScaledLinear: Linear {}

    @Test("a stock projection passes the check")
    func acceptsStockProjection() throws {
        try TensorParallel.requireStockProjection(Linear(4, 4), at: "proj")
    }

    @Test("a projection subclass is not supported")
    func rejectsProjectionSubclass() {
        #expect(throws: TensorParallelShardingError.unsupported(feature: "ScaledLinear at proj")) {
            try TensorParallel.requireStockProjection(ScaledLinear(4, 4), at: "proj")
        }
    }
}

struct Qwen35SingleProcessShardTests {

    /// The seed of the random weights.
    private static let weightSeed: UInt64 = 20_261_124

    /// The prompt tokens.
    private static let tokens = [1, 5, 3, 7]

    /// The logits of `model` for ``tokens``.
    private static func logits(of model: Qwen35TextModel) -> [Float] {
        let input = LMInput.Text(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        return model(input, cache: nil, state: nil).logits.asType(.float32).asArray(Float.self)
    }

    @Test("sharding across a group of one process keeps the logits")
    func groupOfOneKeepsLogits() throws {
        MLXRandom.seed(Self.weightSeed)
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(mtpLayers: 0, hiddenLayers: 2, fullAttentionInterval: 2).utf8
            ))
        let model = Qwen35TextModel(configuration)
        let before = Self.logits(of: model)

        try model.shard(group: MLXDistributed.initialize())

        #expect(Self.logits(of: model) == before)
    }
}
