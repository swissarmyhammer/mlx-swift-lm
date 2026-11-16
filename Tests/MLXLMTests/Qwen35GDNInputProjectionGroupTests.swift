// Copyright © 2026 Apple Inc.
//
// The four input projections of a Qwen3.5 gated delta net run as one
// ``FusedQuantizedProjectionGroup`` that fuses every call, a prefill too, and
// that a rollback switch turns off. The text model's gated delta net drops the
// compiled decode that read its fused projection when a parameter update
// turns the fusion off, as the vision-language twin does.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

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

    // MARK: - Text model

    func testLLMGatedDeltaNetParameterUpdateDropsTheCompiledDecode() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(Fixture.textConfigurationJSON.utf8))
        let model = try Fixture.seededQuantized(
            seed: Self.weightSeed, prepare: { try $0.prepare() },
            { Qwen35TextModel(configuration) })
        let cache: [KVCache?] = [MambaCache(), KVCacheSimple()]
        let vocabulary = configuration.vocabularySize
        let prompt = seededTokens(
            Self.promptLength, vocabularySize: vocabulary, seed: Self.promptSeed)[.newAxis]
        let following = seededTokens(
            Self.decodeSteps, vocabularySize: vocabulary, seed: Self.followingSeed)[.newAxis]
        eval(model.model.forward(prompt, cache: cache, applyFinalNorm: true))
        for step in 0 ..< Self.decodeSteps {
            eval(
                model.model.forward(
                    following[0..., step ..< (step + 1)], cache: cache, applyFinalNorm: true))
        }
        XCTAssertGreaterThan(model.model.compiledDecodeSegmentCount, 0)
        let gdn = try XCTUnwrap(model.model.layers[0].linearAttn)
        let (key, scales) = try XCTUnwrap(
            gdn.parameters().flattened().first { key, _ in key == "in_proj_qkv.scales" })

        gdn.update(
            parameters: ModuleParameters.unflattened([key: scales * Self.scaleUpdateFactor]))

        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)
    }
}
