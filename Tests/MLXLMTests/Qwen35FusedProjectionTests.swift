// Copyright © 2026 Apple Inc.
//
// The loader fuses the q/k/v projections of each full-attention layer and the
// gate/up projections of each dense MLP of a quantized Qwen3.5 model into one
// matmul each, when it prepares the model. The fused model gives the bits of
// the unfused one on the general path, in the compiled decode and in the
// compiled verify, for 4-bit weights and for the 8-bit weights of
// `mlx-community/Qwen3.8-27B-8bit`. It keeps the checkpoint keys, and it falls
// back to the separate projections when an adapter or another quantization
// replaces one.

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

    /// The token counts of a projection check of the gated delta net input
    /// projection: a decode row and a verify. Its group fuses a prefill too
    /// (`FusedRows.everyCall`), which may sum in another order than the
    /// separate projections, at any bit width.
    private static let vectorCallTokenCounts = [1, 7]

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

    /// The modules of the fixture whose projections the loader fuses: one
    /// attention, the dense MLP of each of the two layers, and one gated
    /// delta net.
    private static let fusedModuleCount = 4

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

    /// The text model on seeded weights, quantized to `bits`, prepared as the
    /// loader prepares it when `prepared` is true.
    private func llmModel(prepared: Bool, bits: Int = Fixture.bits) throws -> Qwen35TextModel {
        let configuration = try llmConfiguration()
        return try Fixture.seededQuantized(
            seed: Self.weightSeed, bits: bits,
            prepare: { model in
                if prepared {
                    try model.prepare()
                }
            },
            { Qwen35TextModel(configuration) })
    }

    /// The language model of the vision-language model on seeded weights,
    /// quantized to `bits`, with its projections fused when `prepared` is true.
    private func vlmModel(prepared: Bool, bits: Int = Fixture.bits) throws
        -> Qwen35Language.Model
    {
        let configuration = try vlmConfiguration()
        return try Fixture.seededQuantized(
            seed: Self.weightSeed, bits: bits,
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

    /// Asserts that `prepare` fuses the projections, and that the outputs of
    /// `project` keep their bits for every token count of `tokenCounts`.
    ///
    /// - Parameters:
    ///   - label: The projection in a failure.
    ///   - prepare: Fuses the projections, and tells if it did.
    ///   - tokenCounts: The token counts of the inputs.
    ///   - project: The outputs of the projections of one input.
    private func assertFusionKeepsTheBits(
        _ label: String, prepare: () throws -> Bool,
        tokenCounts: [Int] = Qwen35FusedProjectionTests.projectionTokenCounts,
        project: (MLXArray) -> [MLXArray],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let inputs = tokenCounts.map(rows)
        let separate = inputs.map(project)

        XCTAssertTrue(try prepare(), "\(label): not fused", file: file, line: line)

        for (input, want) in zip(inputs, separate) {
            assertBitIdentical(
                project(input), want, "\(label) of \(input.dim(1)) tokens", file: file,
                line: line)
        }
    }

    private func assertLLMFusedAttentionKeepsTheBits(bits: Int) throws {
        let attention = try XCTUnwrap(
            try llmModel(prepared: false, bits: bits).modules()
                .compactMap { $0 as? Qwen35Attention }.first)
        try assertFusionKeepsTheBits(
            "LLM \(bits)-bit q/k/v", prepare: attention.prepareFusedProjection
        ) { input in
            let projected = attention.projectPreRope(input)
            return [projected.0, projected.1, projected.2, projected.3]
        }
    }

    private func assertLLMFusedMLPKeepsTheBits(bits: Int) throws {
        let mlp = try XCTUnwrap(
            try llmModel(prepared: false, bits: bits).modules()
                .compactMap { $0 as? Qwen3NextMLP }.first)
        try assertFusionKeepsTheBits(
            "LLM \(bits)-bit MLP", prepare: mlp.prepareFusedProjection
        ) { [mlp($0)] }
    }

    private func assertVLMFusedAttentionKeepsTheBits(bits: Int) throws {
        let attention = try XCTUnwrap(
            try vlmModel(prepared: false, bits: bits).modules()
                .compactMap { $0 as? Qwen35Language.Attention }.first)
        try assertFusionKeepsTheBits(
            "VLM \(bits)-bit q/k/v", prepare: attention.prepareFusedProjection
        ) { input in
            let projected = attention.projectPreNorm(input)
            return [projected.0, projected.1, projected.2, projected.3]
        }
    }

    private func assertVLMFusedMLPKeepsTheBits(bits: Int) throws {
        let mlp = try XCTUnwrap(
            try vlmModel(prepared: false, bits: bits).modules()
                .compactMap { $0 as? Qwen35Language.MLP }.first)
        try assertFusionKeepsTheBits(
            "VLM \(bits)-bit MLP", prepare: mlp.prepareFusedProjection
        ) { [mlp($0)] }
    }

    func testLLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        try assertLLMFusedAttentionKeepsTheBits(bits: Fixture.bits)
    }

    func testLLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        try assertLLMFusedMLPKeepsTheBits(bits: Fixture.bits)
    }

    func testVLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        try assertVLMFusedAttentionKeepsTheBits(bits: Fixture.bits)
    }

    func testVLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        try assertVLMFusedMLPKeepsTheBits(bits: Fixture.bits)
    }

    func testEightBitLLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        try assertLLMFusedAttentionKeepsTheBits(bits: Fixture.eightBits)
    }

    func testEightBitLLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        try assertLLMFusedMLPKeepsTheBits(bits: Fixture.eightBits)
    }

    func testEightBitVLMFusedAttentionProjectionHasTheBitsOfTheSeparateProjections() throws {
        try assertVLMFusedAttentionKeepsTheBits(bits: Fixture.eightBits)
    }

    func testEightBitVLMFusedMLPHasTheBitsOfTheSeparateProjections() throws {
        try assertVLMFusedMLPKeepsTheBits(bits: Fixture.eightBits)
    }

    func testEightBitVLMFusedGatedDeltaInputProjectionHasTheBitsOfTheSeparateProjections()
        throws
    {
        let gdn = try XCTUnwrap(
            try vlmModel(prepared: false, bits: Fixture.eightBits).modules()
                .compactMap { $0 as? Qwen35Language.GatedDeltaNet }.first)
        try assertFusionKeepsTheBits(
            "VLM 8-bit GDN input projection", prepare: gdn.prepareFusedInputProjection,
            tokenCounts: Self.vectorCallTokenCounts
        ) { input in
            let projected = gdn.projectInputs(input, batch: 1, sequence: input.dim(1))
            return [projected.qkv, projected.z, projected.b, projected.a]
        }
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

    /// The fusion state of each attention, dense MLP and gated delta net input
    /// projection of a vision-language model quantized to `bits`, after the
    /// loader prepares it.
    private func vlmFusionStates(bits: Int) throws -> [Bool] {
        let model = try Fixture.visionLanguageTarget(
            seed: Self.weightSeed, bits: bits, prepared: true)
        let modules = model.modules()
        return modules.compactMap { ($0 as? Qwen35Language.Attention)?.hasFusedProjection }
            + modules.compactMap { ($0 as? Qwen35Language.MLP)?.hasFusedProjection }
            + modules.compactMap { ($0 as? Qwen35Language.GatedDeltaNet)?.hasFusedInputProjection }
    }

    func testVLMPrepareFusesEveryAttentionAndDenseMLP() throws {
        let states = try vlmFusionStates(bits: Fixture.bits)

        XCTAssertEqual(states, Array(repeating: true, count: Self.fusedModuleCount))
    }

    func testEightBitVLMPrepareFusesEveryAttentionDenseMLPAndGatedDeltaInput() throws {
        let states = try vlmFusionStates(bits: Fixture.eightBits)

        XCTAssertEqual(states, Array(repeating: true, count: Self.fusedModuleCount))
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

    /// The hidden states of a decode after a prompt of `promptLength` tokens,
    /// step by step, then of a verify after them, each through `forward`.
    private func decodeThenVerify(
        cache: [KVCache?], promptLength: Int,
        _ forward: (MLXArray, [KVCache?], Bool) -> MLXArray
    ) -> [MLXArray] {
        var outputs = [forward(tokens(promptLength, seed: Self.promptSeed), cache, false)]
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

    /// The caches of the fixture layers of a run with `kvBits` 8: a gated
    /// delta net, then attention in the one format the fused quantized-KV
    /// attention kernels read.
    private func eightBitKVCaches() -> [KVCache?] {
        [
            MambaCache(),
            QuantizedKVCache(
                groupSize: QuantizedKVAttentionKernel.groupSize,
                bits: QuantizedKVAttentionKernel.bits),
        ]
    }

    private func vlmDecodeThenVerify(
        _ model: Qwen35Language.Model, cache: [KVCache?]? = nil,
        promptLength: Int = Qwen35FusedProjectionTests.promptLength
    ) -> [MLXArray] {
        decodeThenVerify(cache: cache ?? caches(), promptLength: promptLength) {
            inputs, cache, verify in
            model(inputs, cache: cache, keepsSnapshots: verify)
        }
    }

    private func llmDecodeThenVerify(_ model: Qwen35TextModel) -> [MLXArray] {
        decodeThenVerify(cache: caches(), promptLength: Self.promptLength) {
            inputs, cache, verify in
            model.model.forward(
                inputs, cache: cache, applyFinalNorm: true, keepsSnapshots: verify)
        }
    }

    /// Asserts that the compiled decode and verify of the fused model ran,
    /// and gave the hidden states of the unfused model.
    ///
    /// - Parameters:
    ///   - label: The model in a failure.
    ///   - compiledSegmentCounts: The compiled decode and verify segments of
    ///     the fused model after its run.
    ///   - run: The hidden states of a run of the fused model when its
    ///     argument is true, of the unfused model when it is false.
    private func assertCompiledStepsKeepTheBits(
        _ label: String, compiledSegmentCounts: () -> [Int], run: (Bool) throws -> [MLXArray],
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let got = try run(true)

        for count in compiledSegmentCounts() {
            XCTAssertGreaterThan(count, 0, "\(label): compiled segments", file: file, line: line)
        }
        assertBitIdentical(got, try run(false), label, file: file, line: line)
    }

    private func assertVLMCompiledStepsKeepTheBits(
        bits: Int, cache: @escaping () -> [KVCache?], promptLength: Int
    ) throws {
        let fused = try vlmModel(prepared: true, bits: bits)
        try assertCompiledStepsKeepTheBits(
            "VLM \(bits)-bit step",
            compiledSegmentCounts: {
                [fused.compiledDecodeSegmentCount, fused.compiledVerifySegmentCount]
            }
        ) { isFused in
            try vlmDecodeThenVerify(
                isFused ? fused : vlmModel(prepared: false, bits: bits), cache: cache(),
                promptLength: promptLength)
        }
    }

    private func assertLLMCompiledStepsKeepTheBits(bits: Int) throws {
        let fused = try llmModel(prepared: true, bits: bits)
        try assertCompiledStepsKeepTheBits(
            "LLM \(bits)-bit step",
            compiledSegmentCounts: {
                [fused.model.compiledDecodeSegmentCount, fused.model.compiledVerifySegmentCount]
            }
        ) { isFused in
            try llmDecodeThenVerify(isFused ? fused : llmModel(prepared: false, bits: bits))
        }
    }

    func testVLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        try assertVLMCompiledStepsKeepTheBits(
            bits: Fixture.bits, cache: caches, promptLength: Self.promptLength)
    }

    func testLLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        try assertLLMCompiledStepsKeepTheBits(bits: Fixture.bits)
    }

    func testEightBitVLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        try assertVLMCompiledStepsKeepTheBits(
            bits: Fixture.eightBits, cache: caches, promptLength: Self.promptLength)
    }

    func testEightBitLLMCompiledDecodeAndVerifyHaveTheBitsOfTheUnfusedModel() throws {
        try assertLLMCompiledStepsKeepTheBits(bits: Fixture.eightBits)
    }

    /// The prompt is long enough for the fused quantized-KV decode kernel to
    /// read the cache, thus the steps run the kernels of a `kvBits` 8 run. A
    /// quantized cache keeps the segments of the compiled decode schedule.
    func testEightBitVLMCompiledStepsOverAnEightBitKVCacheHaveTheBitsOfTheUnfusedModel()
        throws
    {
        try assertVLMCompiledStepsKeepTheBits(
            bits: Fixture.eightBits, cache: eightBitKVCaches,
            promptLength: DecodeAttentionKernel.fewestKeys)
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
