// Copyright © 2026 Apple Inc.
//
// The shard plan of a Qwen 3.5 mixture-of-experts model: each process keeps a slice of the inner
// width of every expert and of the shared expert, and the router stays whole. The shares of one
// block, made in one process, must sum to the output of the whole block, which the all-sum of a
// group computes.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The configurations of the suites of this file.
enum Qwen35MoEShardFixture {

    /// The processes of the plans.
    static let groupSize = 2

    /// The width of the residual stream, two groups of the quantization of the tests.
    static let hiddenSize = 64

    /// The experts of each block.
    static let experts = 4

    /// The experts each token uses.
    static let expertsPerToken = 2

    /// A configuration whose head counts a group of two divides, with MoE blocks whose experts
    /// and shared expert are `expertWidth` and `sharedExpertWidth` wide.
    ///
    /// - Parameters:
    ///   - expertWidth: The inner width of each expert.
    ///   - sharedExpertWidth: The inner width of the shared expert.
    ///   - fullAttentionInterval: One layer of each interval is full attention.
    static func makeConfiguration(
        expertWidth: Int, sharedExpertWidth: Int, fullAttentionInterval: Int = 1
    ) throws -> Qwen35TextConfiguration {
        var configuration = try Qwen35TensorParallelPlanTests.makeConfiguration(
            fullAttentionInterval: fullAttentionInterval, hiddenSize: hiddenSize)
        configuration.numExperts = experts
        configuration.numExpertsPerTok = expertsPerToken
        configuration.moeIntermediateSize = expertWidth
        configuration.sharedExpertIntermediateSize = sharedExpertWidth
        return configuration
    }
}

/// The share of one process of a Qwen 3.5 MoE block, made in one process.
enum Qwen35MoEShares {

    /// The rows of the gate and up projections of the shared expert, which split as those of a
    /// dense MLP do.
    private static let sharedExpertRows: [String: TensorParallel.Split] = [
        "gate_proj": TensorParallel.Split(axis: 0, segments: .count(1)),
        "up_proj": TensorParallel.Split(axis: 0, segments: .count(1)),
    ]

    /// The path of the down projection of the shared expert, for the error.
    private static let sharedExpertDownPath = "mlp.shared_expert.down_proj"

    /// Keeps the share of process `rank` of `groupSize` processes of `block`: the in-place splits
    /// of the plan, the rows of the shared expert, and the input shard of its down projection,
    /// whose output the block sums with the routed experts.
    ///
    /// - Parameters:
    ///   - rank: The process whose share the block keeps.
    ///   - groupSize: The number of processes.
    ///   - block: The block whose parameters split, in place.
    static func keep(rank: Int, of groupSize: Int, in block: Qwen35SparseMoeBlock) throws {
        TensorParallelShares.keep(
            rank: rank, of: groupSize, in: block, Qwen35TensorParallelPlan.expertSplits)
        let sharedExpert = block.sharedExpert
        TensorParallelShares.keep(rank: rank, of: groupSize, in: sharedExpert, sharedExpertRows)
        let down = try TensorParallel.makeInputShard(
            sharedExpert.downProj, at: sharedExpertDownPath, rank: rank, groupSize: groupSize)
        sharedExpert.update(modules: ModuleChildren(values: ["down_proj": .value(down)]))
    }

    /// The sum over every process of ``Qwen35MoEShardFixture/groupSize`` of the output on `x` of
    /// its share of a block that `makeBlock` makes, which the all-sum of the block computes.
    ///
    /// - Parameters:
    ///   - makeBlock: Makes the whole block, the same one on each call.
    ///   - x: The input of the block.
    /// - Returns: The sum of the outputs of the shares.
    static func sum(
        of makeBlock: () throws -> Qwen35SparseMoeBlock, on x: MLXArray
    ) throws -> MLXArray {
        let groupSize = Qwen35MoEShardFixture.groupSize
        return try TensorParallelShares.sum(over: groupSize) { rank in
            let block = try makeBlock()
            try keep(rank: rank, of: groupSize, in: block)
            return block(x)
        }
    }
}

struct Qwen35MoETensorParallelPlanTests {

    /// An expert width a group of two divides.
    private static let width = 32

    /// A width no group of two divides.
    private static let oddWidth = 33

    private static func makeConfiguration(
        expertWidth: Int = width, sharedExpertWidth: Int = width
    ) throws -> Qwen35TextConfiguration {
        try Qwen35MoEShardFixture.makeConfiguration(
            expertWidth: expertWidth, sharedExpertWidth: sharedExpertWidth)
    }

    private static func makePlan(_ configuration: Qwen35TextConfiguration) throws
        -> Qwen35TensorParallelPlan
    {
        try Qwen35TensorParallelPlan(
            configuration: configuration, groupSize: Qwen35MoEShardFixture.groupSize)
    }

    @Test("a mixture-of-experts model splits across two processes")
    func splitsMixtureOfExperts() throws {
        let configuration = try Self.makeConfiguration()
        #expect(throws: Never.self) { try Self.makePlan(configuration) }
    }

    @Test("an expert width that the group does not divide throws")
    func throwsForIndivisibleExpertWidth() throws {
        let configuration = try Self.makeConfiguration(expertWidth: Self.oddWidth)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "moeIntermediateSize", count: Self.oddWidth,
                groupSize: Qwen35MoEShardFixture.groupSize)
        ) {
            try Self.makePlan(configuration)
        }
    }

    @Test("a shared expert width that the group does not divide throws")
    func throwsForIndivisibleSharedExpertWidth() throws {
        let configuration = try Self.makeConfiguration(sharedExpertWidth: Self.oddWidth)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "sharedExpertIntermediateSize", count: Self.oddWidth,
                groupSize: Qwen35MoEShardFixture.groupSize)
        ) {
            try Self.makePlan(configuration)
        }
    }

    @Test("the dense MLP width of an MoE model, which no layer uses, need not split")
    func ignoresDenseWidthOfMixtureOfExperts() throws {
        var configuration = try Self.makeConfiguration()
        configuration.intermediateSize = Self.oddWidth
        #expect(throws: Never.self) { try Self.makePlan(configuration) }
    }

    @Test("the gate and up rows of each expert split on their row axis")
    func splitsExpertGateUpRows() {
        #expect(
            TensorParallel.split(
                of: "switch_mlp.up_proj.scales", in: Qwen35TensorParallelPlan.expertSplits)
                == TensorParallel.Split(axis: 1, segments: .count(1)))
    }

    @Test("the down projection of each expert splits its inputs")
    func splitsExpertDownInputs() {
        #expect(
            TensorParallel.split(
                of: "switch_mlp.down_proj.weight", in: Qwen35TensorParallelPlan.expertSplits)
                == TensorParallel.Split(axis: 2, segments: .count(1)))
    }

    @Test(
        "the shared expert splits as a dense MLP does, not in place with the experts",
        arguments: ["gate_proj", "up_proj", "down_proj"])
    func keepsSharedExpertOutOfExpertSplits(projection: String) {
        #expect(
            TensorParallel.split(
                of: "shared_expert.\(projection).weight", in: Qwen35TensorParallelPlan.expertSplits)
                == nil)
    }

    @Test(
        "the router and the shared expert gate stay whole",
        arguments: ["gate", "shared_expert_gate"])
    func keepsRouterWhole(router: String) {
        #expect(
            TensorParallel.split(of: "\(router).weight", in: Qwen35TensorParallelPlan.expertSplits)
                == nil)
    }
}

/// The plan checks of an mxfp4 MoE layer.
struct Qwen35MoEWholeShareTests {

    /// The width of the experts whose down projection has a bias.
    private static let biasedExpertWidth = 16

    @Test("experts whose down projection has a bias are not supported")
    func rejectsExpertDownBias() {
        let experts = SwitchGLU(
            inputDims: Self.biasedExpertWidth, hiddenDims: Self.biasedExpertWidth,
            numExperts: Qwen35MoEShardFixture.experts, bias: true)
        #expect(
            throws: TensorParallelShardingError.unsupported(
                feature: "a bias of the expert down projection at mlp.switch_mlp.down_proj")
        ) {
            try TensorParallel.requireStockExperts(experts, at: "mlp.switch_mlp")
        }
    }

    /// The format of the experts.
    private static let mxfp4GroupSize = 32
    private static let mxfp4Bits = 4

    /// The mxfp4 groups of a width whose share on each process keeps whole groups.
    private static let wholeShareGroups = Qwen35MoEShardFixture.groupSize

    /// A width whose share on each process keeps whole mxfp4 groups.
    private static let wholeShareWidth = wholeShareGroups * mxfp4GroupSize

    /// Layer 0 of a model of `expertWidth` experts, with every projection quantized to mxfp4.
    private static func makeLayer(expertWidth: Int) throws -> (
        Qwen35DecoderLayer, Qwen35TensorParallelPlan
    ) {
        let configuration = try Qwen35MoEShardFixture.makeConfiguration(
            expertWidth: expertWidth, sharedExpertWidth: wholeShareWidth)
        let layer = Qwen35DecoderLayer(configuration, layerIdx: 0)
        quantize(model: layer, groupSize: mxfp4GroupSize, bits: mxfp4Bits, mode: .mxfp4)
        let plan = try Qwen35TensorParallelPlan(
            configuration: configuration, groupSize: Qwen35MoEShardFixture.groupSize)
        return (layer, plan)
    }

    @Test("an mxfp4 MoE layer whose every split keeps whole groups passes")
    func acceptsWholeMXFP4Groups() throws {
        let (layer, plan) = try Self.makeLayer(expertWidth: Self.wholeShareWidth)
        #expect(throws: Never.self) { try layer.requireWholeShares(plan) }
    }

    @Test("mxfp4 experts of one group per row throw, naming the expert down projection")
    func rejectsSplitExpertDownGroup() throws {
        let (layer, plan) = try Self.makeLayer(expertWidth: Self.mxfp4GroupSize)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity:
                    "mlp.switch_mlp.down_proj quantization groups (mxfp4, 4-bit, 32 inputs each)",
                count: 1, groupSize: Qwen35MoEShardFixture.groupSize)
        ) {
            try layer.requireWholeShares(plan)
        }
    }
}

/// Each process's share of an MoE block, run in one process: the partial outputs of the shares
/// must sum to the output of the whole block.
struct Qwen35MoEShareSumTests {

    /// The seed of the block weights.
    private static let weightSeed: UInt64 = 20_261_126

    /// The seed of the block input.
    private static let inputSeed: UInt64 = 11

    /// The tokens of the block input, more than one so the block runs its prefill form.
    private static let length = 5

    /// The inner width of the experts and of the shared expert, two quantization groups on each
    /// process.
    private static let width = 128

    /// The quantization of a quantized block: 4-bit, in groups of 32 inputs.
    private static let quantizationGroupSize = 32
    private static let quantizationBits = 4

    /// The largest difference float32 sums in another order leave.
    private static let tolerance: Float = 1e-4

    /// The format of the weights of a block.
    enum BlockFormat: CaseIterable, CustomTestStringConvertible {
        /// Float weights.
        case float
        /// Affine quantized weights.
        case affine
        /// mxfp4 quantized weights.
        case mxfp4

        /// The quantization mode of the format, or nil for float weights.
        var mode: QuantizationMode? {
            switch self {
            case .float: nil
            case .affine: .affine
            case .mxfp4: .mxfp4
            }
        }

        var testDescription: String { "\(self)" }
    }

    /// A block on the weights of ``weightSeed``, in `format`.
    private static func makeBlock(_ format: BlockFormat) throws -> Qwen35SparseMoeBlock {
        let configuration = try Qwen35MoEShardFixture.makeConfiguration(
            expertWidth: width, sharedExpertWidth: width)
        let block = withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            Qwen35SparseMoeBlock(configuration)
        }
        if let mode = format.mode {
            quantize(
                model: block, groupSize: quantizationGroupSize, bits: quantizationBits,
                mode: mode)
        }
        eval(block)
        return block
    }

    private static func blockInput() -> MLXArray {
        MLXRandom.normal(
            [1, length, Qwen35MoEShardFixture.hiddenSize], key: MLXRandom.key(inputSeed))
    }

    @Test(
        "the expert shares of a float, an affine and an mxfp4 block sum to the whole block",
        arguments: BlockFormat.allCases)
    func expertSharesSum(format: BlockFormat) throws {
        let x = Self.blockInput()
        let expected = try Self.makeBlock(format)(x)

        let sum = try Qwen35MoEShares.sum(of: { try Self.makeBlock(format) }, on: x)

        #expect(maxAbsDiff(sum, expected) < Self.tolerance)
    }

    @Test("a block whose experts sum across a group of one keeps its decode output")
    func groupOfOneKeepsDecodeOutput() throws {
        let block = try Self.makeBlock(.affine)
        let token = Self.blockInput()[0..., ..<1]
        let before = block(token)

        block.adoptShardedExperts(group: try MLXDistributed.initialize())

        #expect(maxAbsDiff(block(token), before) == 0)
    }
}

struct Qwen35MoESingleProcessShardTests {

    /// The seed of the random weights.
    private static let weightSeed: UInt64 = 20_261_125

    /// The prompt tokens.
    private static let tokens = [1, 5, 3, 7]

    @Test("sharding an MoE model across a group of one process keeps the logits")
    func groupOfOneKeepsLogits() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, numExperts: Qwen35MoEShardFixture.experts,
                    expertsPerToken: Qwen35MoEShardFixture.expertsPerToken, hiddenLayers: 2,
                    fullAttentionInterval: 2
                ).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            Qwen35TextModel(configuration)
        }
        let input = LMInput.Text(tokens: MLXArray(Self.tokens).expandedDimensions(axis: 0))
        let before = model(input, cache: nil, state: nil).logits

        try model.shard(group: MLXDistributed.initialize())

        #expect(maxAbsDiff(model(input, cache: nil, state: nil).logits, before) == 0)
    }
}
