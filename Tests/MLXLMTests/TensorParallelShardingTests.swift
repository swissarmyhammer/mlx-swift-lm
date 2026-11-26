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
    ///
    /// - Parameters:
    ///   - fullAttentionInterval: One layer of each interval is full attention.
    ///   - hiddenSize: The width of the residual stream.
    ///   - intermediateSize: The MLP width, or nil for the width of the fixture.
    static func makeConfiguration(
        fullAttentionInterval: Int = 1, hiddenSize: Int = Qwen35FixtureJSON.defaultHiddenSize,
        intermediateSize: Int? = nil
    ) throws -> Qwen35TextConfiguration {
        var configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, fullAttentionInterval: fullAttentionInterval,
                    hiddenSize: hiddenSize
                ).utf8))
        configuration.intermediateSize = intermediateSize ?? configuration.intermediateSize
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

/// The input shard of a projection, whose output is the part of one process of a sum that the
/// caller makes once.
struct TensorParallelInputShardTests {

    /// The processes of the splits.
    private static let groupSize = 2

    /// The inputs of the projection: two quantization groups on each process.
    private static let inputs = 128

    /// The outputs of the projection.
    private static let outputs = 8

    /// The quantization of the quantized projection.
    private static let quantizationGroupSize = 32
    private static let quantizationBits = 4

    /// The seed of the weights and of the input.
    private static let seed: UInt64 = 20_261_127

    /// The tokens of the input.
    private static let length = 3

    /// The largest difference float32 sums in another order leave.
    private static let tolerance: Float = 1e-4

    /// A projection with a bias on seeded weights, quantized to 4-bit affine when `quantized`.
    private static func makeProjection(quantized: Bool) -> Linear {
        let projection = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Linear(inputs, outputs, bias: true)
        }
        guard quantized else { return projection }
        return QuantizedLinear(
            projection, groupSize: quantizationGroupSize, bits: quantizationBits)
    }

    @Test(
        "the input shards of a float and of a quantized projection, the bias on rank 0, sum to it",
        arguments: [false, true])
    func inputShardsSumToWholeProjection(quantized: Bool) throws {
        let whole = Self.makeProjection(quantized: quantized)
        let x = MLXRandom.normal([1, Self.length, Self.inputs], key: MLXRandom.key(Self.seed))
        let inputs = x.split(parts: Self.groupSize, axis: -1)

        let sum = try TensorParallelShares.sum(over: Self.groupSize) { rank in
            let shard = try TensorParallel.makeInputShard(
                whole, at: "proj", rank: rank, groupSize: Self.groupSize)
            return shard(inputs[rank])
        }

        #expect(maxAbsDiff(sum, whole(x)) < Self.tolerance)
    }
}

struct TensorParallelWholeShareTests {

    /// The processes of the splits.
    private static let groupSize = 2

    /// The rows of the projections.
    private static let rows = 8

    /// The 8-bit affine format of the attention of Inkling 2 Mini oQ4e.
    private static let eightBits = 8
    private static let affineGroupSize = 64

    /// The 4-bit microscaled formats.
    private static let fourBits = 4
    private static let mxfp4GroupSize = 32
    private static let nvfp4GroupSize = 16

    @Test("a row split of whole rows on every process passes")
    func acceptsWholeRows() {
        let projection = QuantizedLinear(
            Self.affineGroupSize, Self.rows, bias: false, groupSize: Self.affineGroupSize,
            bits: Self.eightBits)
        #expect(throws: Never.self) {
            try TensorParallel.requireWholeRows(
                of: projection, at: "self_attn.q_proj", across: Self.groupSize)
        }
    }

    @Test("a segment of rows that the group does not divide throws, naming the layer")
    func rejectsIndivisibleRowSegment() {
        let processes = 4
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "linear_attn.in_proj_qkv rows", count: 6, groupSize: processes)
        ) {
            try TensorParallel.requireWholeRows(
                12, segments: .indices([6]), at: "linear_attn.in_proj_qkv", across: processes)
        }
    }

    @Test("8-bit affine inputs that split into whole groups of 64 pass")
    func acceptsWholeAffineGroups() {
        let projection = QuantizedLinear(
            4 * Self.affineGroupSize, Self.rows, bias: false, groupSize: Self.affineGroupSize,
            bits: Self.eightBits)
        #expect(throws: Never.self) {
            try TensorParallel.requireWholeInputs(
                of: projection, at: "self_attn.o_proj", across: Self.groupSize)
        }
    }

    @Test("mxfp4 inputs that would split a group of 32 throw, naming the layer")
    func rejectsSplitMXFP4Group() {
        let projection = QuantizedLinear(
            Self.mxfp4GroupSize, Self.rows, bias: false, groupSize: Self.mxfp4GroupSize,
            bits: Self.fourBits, mode: .mxfp4)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "mlp.down_proj quantization groups (mxfp4, 4-bit, 32 inputs each)",
                count: 1, groupSize: Self.groupSize)
        ) {
            try TensorParallel.requireWholeInputs(
                of: projection, at: "mlp.down_proj", across: Self.groupSize)
        }
    }

    @Test("nvfp4 inputs that would split a group of 16 throw, naming the layer")
    func rejectsSplitNVFP4Group() {
        let quantization = TensorParallel.QuantizationGroups(
            mode: .nvfp4, bits: Self.fourBits, groupSize: Self.nvfp4GroupSize)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity:
                    "mlp.experts.down_proj quantization groups (nvfp4, 4-bit, 16 inputs each)",
                count: 3, groupSize: Self.groupSize)
        ) {
            try TensorParallel.requireWholeInputs(
                3 * Self.nvfp4GroupSize, quantization: quantization,
                at: "mlp.experts.down_proj", across: Self.groupSize)
        }
    }

    @Test("float inputs that the group does not divide throw, naming the layer")
    func rejectsIndivisibleFloatInputs() {
        let processes = 4
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "self_attn.o_proj inputs", count: 6, groupSize: processes)
        ) {
            try TensorParallel.requireWholeInputs(
                of: Linear(6, Self.rows), at: "self_attn.o_proj", across: processes)
        }
    }
}

struct Qwen35WholeShareTests {

    /// The processes of the plan.
    private static let groupSize = 2

    /// The width of the residual stream, two mxfp4 groups.
    private static let hiddenSize = 64

    /// The format of Qwen3.8-27B-mxfp4.
    private static let mxfp4GroupSize = 32
    private static let mxfp4Bits = 4

    /// The first layer of a model whose every layer is full attention, and of one whose every
    /// layer is gated delta.
    private static let layerKinds = [1, 2]

    /// A configuration of `intermediateSize` MLP width whose layers are full attention every
    /// `fullAttentionInterval` layers.
    private static func makeConfiguration(intermediateSize: Int, fullAttentionInterval: Int)
        throws -> Qwen35TextConfiguration
    {
        try Qwen35TensorParallelPlanTests.makeConfiguration(
            fullAttentionInterval: fullAttentionInterval, hiddenSize: hiddenSize,
            intermediateSize: intermediateSize)
    }

    /// Layer 0 of `configuration`, with every projection quantized to mxfp4.
    private static func makeLayer(_ configuration: Qwen35TextConfiguration) -> Qwen35DecoderLayer {
        let layer = Qwen35DecoderLayer(configuration, layerIdx: 0)
        quantize(model: layer, groupSize: mxfp4GroupSize, bits: mxfp4Bits, mode: .mxfp4)
        return layer
    }

    @Test(
        "an mxfp4 layer whose every split keeps whole groups passes",
        arguments: layerKinds)
    func acceptsWholeMXFP4Groups(fullAttentionInterval: Int) throws {
        let configuration = try Self.makeConfiguration(
            intermediateSize: 2 * Self.hiddenSize, fullAttentionInterval: fullAttentionInterval)
        let plan = try Qwen35TensorParallelPlan(
            configuration: configuration, groupSize: Self.groupSize)
        let layer = Self.makeLayer(configuration)
        #expect(throws: Never.self) { try layer.requireWholeShares(plan) }
    }

    @Test("an mxfp4 MLP of one group per row throws, naming the down projection")
    func rejectsSplitMXFP4DownGroup() throws {
        let configuration = try Self.makeConfiguration(
            intermediateSize: Self.mxfp4GroupSize, fullAttentionInterval: 1)
        let plan = try Qwen35TensorParallelPlan(
            configuration: configuration, groupSize: Self.groupSize)
        let layer = Self.makeLayer(configuration)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "mlp.down_proj quantization groups (mxfp4, 4-bit, 32 inputs each)",
                count: 1, groupSize: Self.groupSize)
        ) {
            try layer.requireWholeShares(plan)
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
