// Copyright © 2026 Apple Inc.
//
// Tensor parallel sharding of Hadamard-rotated projections. A row split keeps the rotation whole,
// because every process reads the whole input. An input split keeps the signs of the inputs of
// each process, which is the rotation of those inputs alone when each share holds whole blocks.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A rotated projection and its input, shared by the suites of this file.
enum RotatedProjectionFixture {

    /// The processes of the splits.
    static let groupSize = 2

    /// The inputs of the projection: two blocks and four quantization groups on each process.
    static let inputs = 128

    /// The outputs of the projection.
    static let outputs = 16

    /// The Hadamard block of the projection.
    static let blockSize = 32

    /// The quantization of the projection.
    static let quantizationGroupSize = 32
    static let quantizationBits = 4

    /// The seed of the weights, the signs and the input.
    private static let seed: UInt64 = 20_261_126

    /// The tokens of the input.
    private static let length = 3

    /// A rotated projection over `blockSize` blocks, on seeded weights and signs.
    static func makeProjection(blockSize: Int = blockSize) -> HadamardQuantizedLinear {
        let (weightKey, signKey) = MLXRandom.split(key: MLXRandom.key(seed))
        let (weight, scales, biases) = quantized(
            MLXRandom.normal([outputs, inputs], key: weightKey),
            groupSize: quantizationGroupSize, bits: quantizationBits)
        let signs = MLXRandom.bernoulli([inputs], key: signKey).asType(.float32) * 2 - 1
        return HadamardQuantizedLinear(
            weight: weight, scales: scales, biases: biases, signs: signs,
            groupSize: quantizationGroupSize, bits: quantizationBits,
            rotation: SignedBlockHadamard(blockSize: blockSize))
    }

    /// The input of the projection.
    static func input() -> MLXArray {
        MLXRandom.normal([1, length, inputs], key: MLXRandom.key(seed + 1))
    }
}

struct HadamardShardedToAllTests {

    /// The largest difference float32 sums of rotated, dequantized products in another order
    /// leave.
    private static let tolerance: Float = 1e-3

    @Test("the input shares of a rotated projection, each with its own signs, sum to the whole")
    func inputSharesSumToWholeProjection() throws {
        let whole = RotatedProjectionFixture.makeProjection()
        let x = RotatedProjectionFixture.input()
        let groupSize = RotatedProjectionFixture.groupSize
        let inputs = x.split(parts: groupSize, axis: -1)

        var sum = MLXArray.zeros(like: whole(x))
        for rank in 0 ..< groupSize {
            let share = try ShardedToAllHadamardQuantizedLinear(
                whole, at: "self_attn.o_proj", rank: rank, groupSize: groupSize,
                group: MLXDistributed.initialize())
            sum = sum + share.partialProduct(inputs[rank])
        }

        #expect(maxAbsDiff(sum, whole(x)) < Self.tolerance)
    }

    @Test("a rotated projection sharded across a group of one keeps its output")
    func groupOfOneKeepsOutput() throws {
        let whole = RotatedProjectionFixture.makeProjection()
        let x = RotatedProjectionFixture.input()

        let sharded = try TensorParallel.makeShardedToAll(
            whole, at: "self_attn.o_proj", group: MLXDistributed.initialize())

        #expect(maxAbsDiff(sharded(x), whole(x)) == 0)
    }

    @Test("inputs whose share would split a Hadamard block throw, naming the projection")
    func rejectsSplitBlock() {
        let blockSize = RotatedProjectionFixture.inputs
        let projection = RotatedProjectionFixture.makeProjection(blockSize: blockSize)
        #expect(
            throws: TensorParallelShardingError.indivisible(
                quantity: "self_attn.o_proj Hadamard blocks (\(blockSize) inputs each)",
                count: 1, groupSize: RotatedProjectionFixture.groupSize)
        ) {
            try TensorParallel.requireWholeInputs(
                of: projection, at: "self_attn.o_proj",
                across: RotatedProjectionFixture.groupSize)
        }
    }
}

struct HadamardRowSplitTests {

    /// A module that holds one rotated projection, as an attention holds its query projection.
    private final class Owner: Module {
        @ModuleInfo(key: "q_proj") var queryProj: Linear

        init(_ projection: Linear) {
            _queryProj.wrappedValue = projection
        }
    }

    @Test("a rotated projection passes the stock projection check")
    func acceptsRotatedProjection() {
        #expect(throws: Never.self) {
            try TensorParallel.requireStockProjection(
                RotatedProjectionFixture.makeProjection(), at: "self_attn.q_proj")
        }
    }

    @Test("a row split of a rotated projection keeps its signs whole")
    func rowSplitKeepsSigns() throws {
        let owner = Owner(RotatedProjectionFixture.makeProjection())
        let rows = ["q_proj": TensorParallel.Split(axis: 0, segments: .count(1))]

        #expect(try TensorParallel.wholeRotationSigns(in: owner, rows) == ["q_proj.signs"])
    }

    @Test("an input split of a rotated projection in place is not supported")
    func rejectsInPlaceInputSplit() {
        let owner = Owner(RotatedProjectionFixture.makeProjection())
        let inputs = ["q_proj": TensorParallel.Split(axis: 1, segments: .count(1))]

        #expect(
            throws: TensorParallelShardingError.unsupported(
                feature: "an input split of the rotated projection q_proj")
        ) {
            try TensorParallel.wholeRotationSigns(in: owner, inputs)
        }
    }
}

struct PrismHadamardQwen35ShardTests {

    /// The processes of the plan.
    private static let groupSize = 2

    @Test("every decoder layer of a rotated pack makes its shards for two processes")
    func rotatedLayersMakeShards() async throws {
        let model = try #require(
            try await LLMTypeRegistry.shared.createModel(
                configuration: Data(HadamardQuantizedTests.packConfig.utf8),
                modelType: "prism_hadamard_qwen35") as? PrismHadamardQwen35Model)
        let plan = try Qwen35TensorParallelPlan(
            configuration: model.configuration, groupSize: Self.groupSize)
        let group = try MLXDistributed.initialize()

        #expect(throws: Never.self) {
            for layer in model.languageModel.model.layers {
                _ = try layer.makeShards(plan, group: group)
            }
        }
    }
}
