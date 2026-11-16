// Copyright © 2026 Apple Inc.
//
// The four input projections of a Qwen3.5 gated delta net run as one
// ``FusedQuantizedProjectionGroup`` that fuses every call, a prefill too, and
// that a rollback switch turns off. The gated delta net of each twin drops the
// compiled decode that read its fused projection when a parameter update
// turns the fusion off, and when its rollback switch flips.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen35GDNInputProjectionGroupTests: XCTestCase {

    private typealias Fixture = MultiRowProjectionFixture

    /// A module with two sibling projections that read one input.
    private final class ProjectionPair: Module {
        /// The checkpoint key of ``first``.
        static let firstKey = "first"
        /// The checkpoint key of ``second``.
        static let secondKey = "second"

        @ModuleInfo(key: "first") var first: Linear
        @ModuleInfo(key: "second") var second: Linear

        /// - Parameters:
        ///   - inputWidth: The width of the shared input.
        ///   - outputWidths: The output widths of the two projections.
        init(inputWidth: Int, outputWidths: (Int, Int)) {
            _first.wrappedValue = Linear(inputWidth, outputWidths.0, bias: false)
            _second.wrappedValue = Linear(inputWidth, outputWidths.1, bias: false)
            super.init()
        }

        /// The two projections, in the order of the keys.
        var projections: [Linear] { [first, second] }
    }

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 20_261_120

    /// The seed of the input rows.
    private static let rowSeed: UInt64 = 20_261_121

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_122

    /// The seed of the decode tokens.
    private static let followingSeed: UInt64 = 20_261_123

    /// The output widths of the test pair: unequal, so a wrong split shows.
    private static let pairOutputWidths = (1024, 512)

    /// The input rows of a prefill: far past the rows MLX gives to its
    /// vector kernels.
    private static let prefillRows = 64

    /// The prompt tokens the cache holds before the decode steps.
    private static let promptLength = 8

    /// The decode steps after the prompt.
    private static let decodeSteps = 3

    /// The factor of the scales of a parameter update.
    private static let scaleUpdateFactor: Float = 1.25

    // MARK: - Group

    /// The pair on seeded weights, quantized.
    private func quantizedPair() throws -> ProjectionPair {
        try Fixture.seededQuantized(seed: Self.weightSeed) {
            ProjectionPair(inputWidth: Fixture.hiddenSize, outputWidths: Self.pairOutputWidths)
        }
    }

    /// The group of the pair's two projections.
    private func pairGroup(
        fusedRows: FusedQuantizedProjectionGroup.FusedRows, isEnabled: Bool = true
    ) -> FusedQuantizedProjectionGroup {
        FusedQuantizedProjectionGroup(
            keys: [ProjectionPair.firstKey, ProjectionPair.secondKey], fusedRows: fusedRows,
            isEnabled: isEnabled)
    }

    func testEveryCallGroupComputesAPrefillWithTheFusedMatmul() throws {
        let pair = try quantizedPair()
        let group = pairGroup(fusedRows: .everyCall)
        XCTAssertTrue(try group.prepare(pair.projections, in: pair))
        let x = MLXRandom.normal(
            [1, Self.prefillRows, Fixture.hiddenSize], key: MLXRandom.key(Self.rowSeed)
        ).asType(.bfloat16)
        let fused = try XCTUnwrap(fuseQuantizedLinearProjections(pair.projections)).fused
        let want = split(
            MultiRowQuantizedMatvec.projection(x, by: fused),
            indices: [Self.pairOutputWidths.0], axis: -1)

        let got = group(x, pair.projections)

        eval(got, want)
        XCTAssertEqual(zip(got, want).map(BitwiseComparison.mismatchCount), [0, 0])
    }

    func testDisabledGroupKeepsTheSeparateProjections() throws {
        let pair = try quantizedPair()
        let group = pairGroup(fusedRows: .everyCall, isEnabled: false)

        XCTAssertFalse(try group.prepare(pair.projections, in: pair))
    }

    // MARK: - Gated delta net projections

    /// The output widths of the test gated delta net projections: the mixed
    /// qkv, `z`, `b` and `a`.
    private static let gatedDeltaWidths = (qkv: 96, z: 64, b: 4, a: 4)

    /// The value heads of the test gated delta net: `z` splits into this many
    /// heads.
    private static let gatedDeltaHeads = 4

    /// The tokens of the test input rows.
    private static let gatedDeltaTokens = 3

    func testGatedDeltaProjectionSplitsZIntoHeads() throws {
        let widths = Self.gatedDeltaWidths
        let linears = withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            [widths.qkv, widths.z, widths.b, widths.a].map {
                Linear(Fixture.hiddenSize, $0, bias: false)
            }
        }
        let projections = Qwen35GatedDeltaInputProjection.ordered(
            qkv: linears[0], z: linears[1], b: linears[2], a: linears[3])
        let x = MLXRandom.normal(
            [1, Self.gatedDeltaTokens, Fixture.hiddenSize], key: MLXRandom.key(Self.rowSeed))
        let headWidth = widths.z / Self.gatedDeltaHeads
        let want = [
            linears[0](x),
            linears[1](x).reshaped(1, Self.gatedDeltaTokens, Self.gatedDeltaHeads, headWidth),
            linears[2](x), linears[3](x),
        ]

        let got = Qwen35GatedDeltaInputProjection.project(
            x, by: Qwen35GatedDeltaInputProjection.makeGroup(), projections, batch: 1,
            sequence: Self.gatedDeltaTokens, heads: Self.gatedDeltaHeads, headWidth: headWidth)

        let parts = [got.qkv, got.z, got.b, got.a]
        eval(parts, want)
        XCTAssertEqual(
            zip(parts, want).map(BitwiseComparison.mismatchCount), [0, 0, 0, 0])
    }

    // MARK: - Text model

    /// The text model on seeded weights, quantized and prepared, after a
    /// prompt and compiled decode steps.
    ///
    /// - Parameter fusionEnabled: The rollback switch of the gated delta net
    ///   during the decode.
    private func decodedLLMModel(fusionEnabled: Bool = true) throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(Fixture.textConfigurationJSON.utf8))
        let model = try Fixture.seededQuantized(
            seed: Self.weightSeed, prepare: { try $0.prepare() },
            { Qwen35TextModel(configuration) })
        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = fusionEnabled
        decode(vocabulary: configuration.vocabularySize) { inputs, cache in
            model.model.forward(inputs, cache: cache, applyFinalNorm: true)
        }
        return model
    }

    /// The gated delta net of the first layer of `model`.
    private func gatedDeltaNet(of model: Qwen35TextModel) throws -> Qwen35GatedDeltaNet {
        try XCTUnwrap(model.model.layers[0].linearAttn)
    }

    func testLLMGatedDeltaNetParameterUpdateDropsTheCompiledDecode() throws {
        let model = try decodedLLMModel()
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
        let gdn = try gatedDeltaNet(of: model)
        let (key, scales) = try XCTUnwrap(
            gdn.parameters().flattened().first { key, _ in key == "in_proj_qkv.scales" })

        gdn.update(
            parameters: ModuleParameters.unflattened([key: scales * Self.scaleUpdateFactor]))

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testLLMGatedDeltaNetRollbackSwitchDropsTheCompiledDecode() throws {
        let model = try decodedLLMModel()
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)

        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = false

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    func testLLMGatedDeltaNetRollbackSwitchReenableDropsTheCompiledDecode() throws {
        let model = try decodedLLMModel(fusionEnabled: false)
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)

        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = true

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }

    // MARK: - Vision-language model

    /// The language model of the vision-language model on seeded weights,
    /// quantized and prepared, after a prompt and compiled decode steps.
    ///
    /// - Parameter fusionEnabled: The rollback switch of the gated delta net
    ///   during the decode.
    private func decodedVLMModel(fusionEnabled: Bool = true) throws -> Qwen35Language.Model {
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(Fixture.textConfigurationJSON.utf8))
        let model = try Fixture.seededQuantized(
            seed: Self.weightSeed, prepare: { try $0.prepareFusedProjections() },
            { Qwen35Language.Model(configuration) })
        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = fusionEnabled
        decode(vocabulary: configuration.vocabularySize) { inputs, cache in
            model(inputs, cache: cache)
        }
        return model
    }

    /// The first gated delta net of `model`.
    private func gatedDeltaNet(of model: Qwen35Language.Model) throws
        -> Qwen35Language.GatedDeltaNet
    {
        try XCTUnwrap(model.modules().compactMap { $0 as? Qwen35Language.GatedDeltaNet }.first)
    }

    func testVLMGatedDeltaNetRollbackSwitchDropsTheCompiledDecode() throws {
        let model = try decodedVLMModel()
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)

        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = false

        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)
    }

    func testVLMGatedDeltaNetRollbackSwitchReenableDropsTheCompiledDecode() throws {
        let model = try decodedVLMModel(fusionEnabled: false)
        XCTAssertGreaterThan(model.compiledDecodeSegmentCount, 0)

        try gatedDeltaNet(of: model).fusedInputProjectionEnabled = true

        XCTAssertEqual(model.compiledDecodeSegmentCount, 0)
    }

    // MARK: - Decode

    /// Runs a seeded prompt, then seeded decode steps one token at a time,
    /// through `forward` on the caches of the fixture layers: a gated delta
    /// net, then attention.
    ///
    /// - Parameters:
    ///   - vocabulary: The vocabulary of the model.
    ///   - forward: The model forward over tokens of shape `(1, tokens)`.
    private func decode(
        vocabulary: Int, _ forward: (MLXArray, [KVCache?]) -> MLXArray
    ) {
        let cache: [KVCache?] = [MambaCache(), KVCacheSimple()]
        let prompt = seededTokens(
            Self.promptLength, vocabularySize: vocabulary, seed: Self.promptSeed)[.newAxis]
        let following = seededTokens(
            Self.decodeSteps, vocabularySize: vocabulary, seed: Self.followingSeed)[.newAxis]
        eval(forward(prompt, cache))
        for step in 0 ..< Self.decodeSteps {
            eval(forward(following[0..., step ..< (step + 1)], cache))
        }
    }
}
