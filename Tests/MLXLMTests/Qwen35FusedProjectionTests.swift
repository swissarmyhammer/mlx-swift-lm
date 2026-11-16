// Copyright © 2026 Apple Inc.
//
// The loader fuses the q/k/v projections of each full-attention layer and the
// gate/up projections of each dense MLP of a quantized Qwen3.5 model into one
// matmul each, when it prepares the model. The fused model gives the bits of
// the unfused one on the general path, in the compiled decode and in the
// compiled verify. It keeps the checkpoint keys, and it falls back to the
// separate projections when an adapter or another quantization replaces one.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen35FusedProjectionTests: XCTestCase {

    private typealias Fixture = MultiRowProjectionFixture

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 20_261_116

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_117

    /// The seed of the tokens after the prompt.
    private static let followingSeed: UInt64 = 20_261_118

    /// The seed of the input rows of a projection check.
    private static let rowSeed: UInt64 = 20_261_119

    /// The prompt tokens the cache holds before the decode steps.
    private static let promptLength = 8

    /// The decode steps after the prompt.
    private static let decodeSteps = 3

    /// The tokens of a speculative verify: the bonus token and three drafts.
    private static let verifyTokens = 4

    /// The token counts of a projection check: one decode row, a verify, and
    /// a prefill that MLX gives to its split-K matrix kernel, whose split
    /// depends on the output width.
    private static let projectionTokenCounts = [1, 7, 64]

    /// The bits of a projection quantized apart from its siblings.
    private static let otherBits = 8

    /// The rank of the test adapter.
    private static let adapterRank = 4

    /// Each number of the test adapter: nonzero, so the adapter changes the
    /// output.
    private static let adapterValue: Float = 0.01

    /// The factor of the scales of a parameter update.
    private static let scaleUpdateFactor: Float = 1.25

    /// The `intermediate_size` of the fixture configuration: the rows of a
    /// gate projection.
    private static let intermediateSize = 1024

    /// The vocabulary of the fixture configuration.
    private static let vocabularySize = Qwen35MTPFixture.vocabularySize

    // MARK: - Models

    private func llmConfiguration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(Fixture.textConfigurationJSON.utf8))
    }

    private func vlmConfiguration() throws -> MLXVLM.Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.TextConfiguration.self,
            from: Data(Fixture.textConfigurationJSON.utf8))
    }

    /// The text model on seeded weights, quantized, prepared as the loader
    /// prepares it when `prepared` is true.
    private func llmModel(prepared: Bool) throws -> Qwen35TextModel {
        let configuration = try llmConfiguration()
        return try Fixture.seededQuantized(
            seed: Self.weightSeed,
            prepare: { model in
                if prepared {
                    try model.prepare()
                }
            },
            { Qwen35TextModel(configuration) })
    }

    /// The language model of the vision-language model on seeded weights,
    /// quantized, with its projections fused when `prepared` is true.
    private func vlmModel(prepared: Bool) throws -> Qwen35Language.Model {
        let configuration = try vlmConfiguration()
        return try Fixture.seededQuantized(
            seed: Self.weightSeed,
            prepare: { model in
                if prepared {
                    try model.prepareFusedProjections()
                }
            },
            { Qwen35Language.Model(configuration) })
    }

    /// The seeded input rows of a projection check, shape `(1, tokens, hidden)`.
    private func rows(_ tokens: Int) -> MLXArray {
        MLXRandom.normal([1, tokens, Fixture.hiddenSize], key: MLXRandom.key(Self.rowSeed))
            .asType(.bfloat16)
    }

    /// `count` seeded tokens, shape `(1, count)`.
    private func tokens(_ count: Int, seed: UInt64) -> MLXArray {
        seededTokens(count, vocabularySize: Self.vocabularySize, seed: seed)[.newAxis]
    }

    private func assertBitIdentical(
        _ got: [MLXArray], _ want: [MLXArray], _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, want.count, "\(label): outputs", file: file, line: line)
        for (index, (got, want)) in zip(got, want).enumerated() {
            eval(got, want)
            XCTAssertEqual(
                got.shape, want.shape, "\(label) \(index): shape", file: file, line: line)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0, "\(label) \(index): bits differ",
                file: file, line: line)
        }
    }

    // MARK: - One projection

    func testLLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        let attention = try XCTUnwrap(
            try llmModel(prepared: false).modules().compactMap { $0 as? Qwen35Attention }.first)
        let inputs = Self.projectionTokenCounts.map(rows)
        let separate = inputs.map { attention.projectPreRope($0) }

        XCTAssertTrue(try attention.prepareFusedProjection())

        for (input, want) in zip(inputs, separate) {
            let got = attention.projectPreRope(input)
            assertBitIdentical(
                [got.0, got.1, got.2, got.3], [want.0, want.1, want.2, want.3],
                "LLM q/k/v of \(input.dim(1)) tokens")
        }
    }

    func testLLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        let mlp = try XCTUnwrap(
            try llmModel(prepared: false).modules().compactMap { $0 as? Qwen3NextMLP }.first)
        let inputs = Self.projectionTokenCounts.map(rows)
        let separate = inputs.map { mlp($0) }

        XCTAssertTrue(try mlp.prepareFusedProjection())

        assertBitIdentical(inputs.map { mlp($0) }, separate, "LLM MLP")
    }

    func testVLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        let attention = try XCTUnwrap(
            try vlmModel(prepared: false).modules().compactMap { $0 as? Qwen35Language.Attention }
                .first)
        let inputs = Self.projectionTokenCounts.map(rows)
        let separate = inputs.map { attention.projectPreNorm($0) }

        XCTAssertTrue(try attention.prepareFusedProjection())

        for (input, want) in zip(inputs, separate) {
            let got = attention.projectPreNorm(input)
            assertBitIdentical(
                [got.0, got.1, got.2, got.3], [want.0, want.1, want.2, want.3],
                "VLM q/k/v of \(input.dim(1)) tokens")
        }
    }

    func testVLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        let mlp = try XCTUnwrap(
            try vlmModel(prepared: false).modules().compactMap { $0 as? Qwen35Language.MLP }.first)
        let inputs = Self.projectionTokenCounts.map(rows)
        let separate = inputs.map { mlp($0) }

        XCTAssertTrue(try mlp.prepareFusedProjection())

        assertBitIdentical(inputs.map { mlp($0) }, separate, "VLM MLP")
    }

    // MARK: - Lifecycle

    func testLLMPrepareFusesEveryAttentionAndDenseMLP() throws {
        let model = try llmModel(prepared: true)
        let attention = model.modules().compactMap { $0 as? Qwen35Attention }
        let mlps = model.modules().compactMap { $0 as? Qwen3NextMLP }

        XCTAssertEqual(
            (attention.map(\.hasFusedProjection) + mlps.map(\.hasFusedProjection)),
            Array(repeating: true, count: attention.count + mlps.count))
    }

    func testVLMPrepareFusesEveryAttentionAndDenseMLP() throws {
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: Fixture.textConfigurationJSON)
                    .utf8))
        let model = try Fixture.seededQuantized(
            seed: Self.weightSeed, filter: { path, _ in path.hasPrefix("language_model") },
            prepare: { try $0.prepare() }, { Qwen35(configuration) })
        let attention = model.modules().compactMap { $0 as? Qwen35Language.Attention }
        let mlps = model.modules().compactMap { $0 as? Qwen35Language.MLP }

        XCTAssertEqual(
            (attention.map(\.hasFusedProjection) + mlps.map(\.hasFusedProjection)),
            Array(repeating: true, count: attention.count + mlps.count))
    }

    func testPreparationKeepsTheCheckpointKeys() throws {
        let unfused = Set(try llmModel(prepared: false).parameters().flattened().map(\.0))

        let fused = Set(try llmModel(prepared: true).parameters().flattened().map(\.0))

        XCTAssertEqual(fused, unfused)
    }

    func testProjectionsOfOtherBitsStaySeparate() throws {
        let attention = Qwen35Attention(try llmConfiguration())
        quantize(model: attention, groupSize: Fixture.groupSize, bits: Self.otherBits) { path, _ in
            path == "k_proj"
        }
        quantize(model: attention, groupSize: Fixture.groupSize, bits: Fixture.bits)

        XCTAssertFalse(try attention.prepareFusedProjection())
    }

    // MARK: - Adapters

    /// An adapter on the query and gate projections of every layer.
    private func adapter() -> LoRAContainer {
        let shape = { (rows: Int, columns: Int) in
            MLXArray.full([rows, columns], values: MLXArray(Self.adapterValue), dtype: .bfloat16)
        }
        let rank = Self.adapterRank
        let hidden = Fixture.hiddenSize
        let queryRows = 2 * Fixture.attentionHeads * Fixture.headDimension
        let gateRows = Self.intermediateSize
        return LoRAContainer(
            configuration: .init(
                numLayers: 2,
                loraParameters: .init(
                    rank: rank, scale: 1.0, keys: ["self_attn.q_proj", "mlp.gate_proj"])),
            parameters: ModuleParameters.unflattened([
                "model.layers.0.mlp.gate_proj.lora_a": shape(hidden, rank),
                "model.layers.0.mlp.gate_proj.lora_b": shape(rank, gateRows),
                "model.layers.1.mlp.gate_proj.lora_a": shape(hidden, rank),
                "model.layers.1.mlp.gate_proj.lora_b": shape(rank, gateRows),
                "model.layers.1.self_attn.q_proj.lora_a": shape(hidden, rank),
                "model.layers.1.self_attn.q_proj.lora_b": shape(rank, queryRows),
            ]))
    }

    func testAdapterAfterFusionGivesTheOutputOfTheAdaptedUnfusedModel() throws {
        let fused = try llmModel(prepared: true)
        let unfused = try llmModel(prepared: false)
        for model in [fused, unfused] {
            try adapter().load(into: model)
        }
        try fused.prepare()
        let prompt = tokens(Self.promptLength, seed: Self.promptSeed)

        let got = fused(prompt, cache: try fused.newCache(parameters: nil))
        let want = unfused(prompt, cache: try unfused.newCache(parameters: nil))

        assertBitIdentical([got], [want], "adapted logits")
    }

    func testAdapterOnAFusedProjectionTurnsTheFusionOff() throws {
        let model = try llmModel(prepared: true)

        try adapter().load(into: model)
        try model.prepare()

        let attention = try XCTUnwrap(model.modules().compactMap { $0 as? Qwen35Attention }.first)
        XCTAssertFalse(attention.hasFusedProjection)
    }

    // MARK: - Compiled decode and verify

    /// The hidden states of a decode after the prompt, step by step, then of
    /// a verify after them, each through `forward`.
    private func decodeThenVerify(
        cache: [KVCache?],
        _ forward: (MLXArray, [KVCache?], Bool) -> MLXArray
    ) -> [MLXArray] {
        var outputs = [forward(tokens(Self.promptLength, seed: Self.promptSeed), cache, false)]
        let following = tokens(Self.decodeSteps + Self.verifyTokens, seed: Self.followingSeed)
        for step in 0 ..< Self.decodeSteps {
            outputs.append(forward(following[0..., step ..< (step + 1)], cache, false))
            eval(outputs)
        }
        let verify = following[0..., Self.decodeSteps...]
        outputs.append(
            MultiRowQuantizedMatvec.run(asSpeculativeVerify: true) {
                forward(verify, cache, true)
            })
        eval(outputs)
        return outputs
    }

    /// The caches of the fixture layers: a gated delta net, then attention.
    private func caches() -> [KVCache?] {
        [MambaCache(), KVCacheSimple()]
    }

    private func vlmDecodeThenVerify(_ model: Qwen35Language.Model) -> [MLXArray] {
        decodeThenVerify(cache: caches()) { inputs, cache, verify in
            model(inputs, cache: cache, keepsSnapshots: verify)
        }
    }

    private func llmDecodeThenVerify(_ model: Qwen35TextModel) -> [MLXArray] {
        decodeThenVerify(cache: caches()) { inputs, cache, verify in
            model.model.forward(
                inputs, cache: cache, applyFinalNorm: true, keepsSnapshots: verify)
        }
    }

    func testVLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        let fused = try vlmModel(prepared: true)

        let got = vlmDecodeThenVerify(fused)

        XCTAssertGreaterThan(fused.compiledDecodeSegmentCount, 0)
        XCTAssertGreaterThan(fused.compiledVerifySegmentCount, 0)
        assertBitIdentical(got, vlmDecodeThenVerify(try vlmModel(prepared: false)), "VLM step")
    }

    func testLLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        let fused = try llmModel(prepared: true)

        let got = llmDecodeThenVerify(fused)

        XCTAssertGreaterThan(fused.model.compiledDecodeSegmentCount, 0)
        XCTAssertGreaterThan(fused.model.compiledVerifySegmentCount, 0)
        assertBitIdentical(got, llmDecodeThenVerify(try llmModel(prepared: false)), "LLM step")
    }

    /// Scales the `k_proj` scales of the attention layer of `model`.
    private func updateKeyScales(of model: Module) throws {
        let (key, scales) = try XCTUnwrap(
            model.parameters().flattened().first { key, _ in key.hasSuffix("k_proj.scales") })
        model.update(
            parameters: ModuleParameters.unflattened([key: scales * Self.scaleUpdateFactor]))
    }

    func testVLMParameterUpdateOfAFusedProjectionDropsTheCompiledDecode() throws {
        let model = try vlmModel(prepared: true)
        _ = vlmDecodeThenVerify(model)

        try updateKeyScales(of: model)

        XCTAssertEqual(
            [model.compiledDecodeSegmentCount, model.compiledVerifySegmentCount], [0, 0])
    }

    func testLLMParameterUpdateOfAFusedProjectionDropsTheCompiledDecode() throws {
        let model = try llmModel(prepared: true)
        _ = llmDecodeThenVerify(model)

        try updateKeyScales(of: model)

        XCTAssertEqual(
            [model.model.compiledDecodeSegmentCount, model.model.compiledVerifySegmentCount],
            [0, 0])
    }

    func testParameterUpdateOfAFusedProjectionTurnsTheFusionOff() throws {
        let model = try llmModel(prepared: true)

        try updateKeyScales(of: model)

        let attention = try XCTUnwrap(model.modules().compactMap { $0 as? Qwen35Attention }.first)
        XCTAssertFalse(attention.hasFusedProjection)
    }
}
