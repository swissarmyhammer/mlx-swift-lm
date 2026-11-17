// Copyright © 2026 Apple Inc.
//
// The quantized projections of a speculative verify of the Qwen3.5
// vision-language model run through ``MultiRowQuantizedMatvec``, as those of
// the text model do, for 4-bit weights and for the 8-bit weights of
// `mlx-community/Qwen3.8-27B-8bit`. Every other call of 3 to 8 rows keeps the
// matvec of MLX.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import XCTest

@testable import MLXLMCommon
@testable import MLXVLM

final class Qwen35VLMMultiRowProjectionTests: XCTestCase {

    private typealias Fixture = MultiRowProjectionFixture

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 20_261_032

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_033

    /// The seed of the tokens after the prompt: a verify or a prompt tail.
    private static let followingSeed: UInt64 = 20_261_034

    /// The prompt tokens the cache holds before the verify or the tail.
    private static let promptLength = 8

    /// The `intermediate_size` of the fixture configuration: the outputs of
    /// the gate and up projections of a dense MLP.
    private static let intermediateSize = 1024

    /// The text configuration of every module of these tests.
    private func textConfiguration() throws -> Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self,
            from: Data(Fixture.textConfigurationJSON.utf8))
    }

    /// A vision-language target on seeded weights in bfloat16, its language
    /// model quantized.
    ///
    /// - Parameters:
    ///   - prepared: True to fuse the input projections of each gated delta
    ///     net, as the model loader does.
    ///   - matvecOfMLX: True to give every projection the matvec of MLX.
    ///   - bits: The bits of each quantized weight.
    private func makeTarget(
        prepared: Bool = true, matvecOfMLX: Bool = false, bits: Int = Fixture.bits
    ) throws -> Qwen35 {
        try Fixture.visionLanguageTarget(
            seed: Self.weightSeed, matvecOfMLX: matvecOfMLX, bits: bits, prepared: prepared)
    }

    /// The seeded tokens of a call, shape `[1, count]`.
    private func tokens(_ count: Int, seed: UInt64, of target: Qwen35) -> MLXArray {
        seededTokens(count, vocabularySize: target.vocabularySize, seed: seed)[.newAxis]
    }

    /// The output of `count` tokens after the prompt, with `flags` set in the
    /// state the prompt left.
    private func callAfterPrompt(
        _ target: Qwen35, count: Int, flags: [LMOutput.Key<Bool>]
    ) throws -> LMOutput {
        let cache: [KVCache?] = try target.newCache(parameters: nil)
        let prompt = target.languageModel(
            tokens(Self.promptLength, seed: Self.promptSeed, of: target), cache: cache,
            state: nil)
        eval(prompt.logits)
        var state = prompt.state ?? LMOutput.State()
        for flag in flags {
            state[flag] = true
        }
        let output = target.languageModel(
            tokens(count, seed: Self.followingSeed, of: target), cache: cache, state: state)
        eval(output.logits)
        return output
    }

    /// The dense MLP of the fixture on seeded weights quantized to `bits`.
    private func makeMLP(bits: Int) throws -> Qwen35Language.MLP {
        try Fixture.seededQuantized(seed: Self.weightSeed, bits: bits) {
            Qwen35Language.MLP(
                dimensions: Fixture.hiddenSize, hiddenDimensions: Self.intermediateSize)
        }
    }

    private func assertMLPRowsMatchSingleRows(bits: Int) throws {
        let mlp = try makeMLP(bits: bits)
        Fixture.assertVerifyRowsMatchSingleRows(width: Fixture.hiddenSize, "\(bits)-bit mlp") {
            [mlp($0)]
        }
    }

    private func assertGatedDeltaInputRowsMatchSingleRows(bits: Int) throws {
        let configuration = try textConfiguration()
        let gdn = try Fixture.seededQuantized(seed: Self.weightSeed, bits: bits) {
            Qwen35Language.GatedDeltaNet(configuration)
        }
        XCTAssertTrue(try gdn.prepareFusedInputProjection())
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, "\(bits)-bit gdn input projection"
        ) { x in
            let projected = gdn.projectInputs(x, batch: 1, sequence: x.dim(1))
            return [projected.qkv, projected.z, projected.b, projected.a]
        }
    }

    private func assertGatedDeltaOutputRowsMatchSingleRows(bits: Int) throws {
        let configuration = try textConfiguration()
        let gdn = try Fixture.seededQuantized(seed: Self.weightSeed, bits: bits) {
            Qwen35Language.GatedDeltaNet(configuration)
        }
        Fixture.assertVerifyRowsMatchSingleRows(width: gdn.valueDim, "\(bits)-bit gdn out_proj") {
            [gdn.projectOutput($0)]
        }
    }

    private func assertAttentionRowsMatchSingleRows(bits: Int) throws {
        let configuration = try textConfiguration()
        let attention = try Fixture.seededQuantized(seed: Self.weightSeed, bits: bits) {
            Qwen35Language.Attention(configuration)
        }
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, axis: Fixture.headTokenAxis, "\(bits)-bit attention q/k/v"
        ) { x in
            let projection = attention.projectPreNorm(x)
            let normed = attention.normedHeads(queries: projection.queries, keys: projection.keys)
            return [normed.queries, normed.keys, projection.values]
        }
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, "\(bits)-bit attention gate"
        ) {
            [attention.projectPreNorm($0).gate]
        }
        let width = Fixture.attentionHeads * Fixture.headDimension
        Fixture.assertVerifyRowsMatchSingleRows(width: width, "\(bits)-bit attention o_proj") {
            gate in
            let heads = gate.reshaped(
                1, gate.dim(1), Fixture.attentionHeads, Fixture.headDimension
            ).transposed(0, 2, 1, 3)
            return [attention.mergeHeadsAndProject(attention: heads, gate: gate)]
        }
    }

    func testMLPRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertMLPRowsMatchSingleRows(bits: Fixture.bits)
    }

    func testGatedDeltaInputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertGatedDeltaInputRowsMatchSingleRows(bits: Fixture.bits)
    }

    func testGatedDeltaOutputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertGatedDeltaOutputRowsMatchSingleRows(bits: Fixture.bits)
    }

    func testAttentionProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertAttentionRowsMatchSingleRows(bits: Fixture.bits)
    }

    func testEightBitMLPProjectionsRunTheMultiRowKernelInAVerify() throws {
        let mlp = try makeMLP(bits: Fixture.eightBits)
        let rows = MLXArray.zeros([1, Fixture.rows, Fixture.hiddenSize], dtype: .bfloat16)
        let hidden = MLXArray.zeros([1, Fixture.rows, Self.intermediateSize], dtype: .bfloat16)

        let runs = [
            MultiRowQuantizedMatvec.canRun(rows, mlp.gateProj),
            MultiRowQuantizedMatvec.canRun(rows, mlp.upProj),
            MultiRowQuantizedMatvec.canRun(hidden, mlp.downProj),
        ]

        XCTAssertEqual(runs, [true, true, true])
    }

    func testEightBitMLPRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertMLPRowsMatchSingleRows(bits: Fixture.eightBits)
    }

    func testEightBitGatedDeltaInputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertGatedDeltaInputRowsMatchSingleRows(bits: Fixture.eightBits)
    }

    func testEightBitGatedDeltaOutputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertGatedDeltaOutputRowsMatchSingleRows(bits: Fixture.eightBits)
    }

    func testEightBitAttentionProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        try assertAttentionRowsMatchSingleRows(bits: Fixture.eightBits)
    }

    private func assertPrefillTailKeepsTheMatvecOfMLX(bits: Int) throws {
        let target = try makeTarget(prepared: false, bits: bits)
        let reference = try makeTarget(prepared: false, matvecOfMLX: true, bits: bits)

        for (label, flags) in [("plain", []), ("MTP prefill", [mtpEmitFlagKey])] {
            let got = try callAfterPrompt(target, count: Fixture.rows, flags: flags).logits
            let want = try callAfterPrompt(reference, count: Fixture.rows, flags: flags).logits
            XCTAssertEqual(got.shape, want.shape, label)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0,
                "\(bits)-bit \(label): a prompt tail of \(Fixture.rows) tokens left the matvec of MLX"
            )
        }
    }

    func testPrefillTailOutsideAVerifyKeepsTheMatvecOfMLX() throws {
        try assertPrefillTailKeepsTheMatvecOfMLX(bits: Fixture.bits)
    }

    func testEightBitPrefillTailOutsideAVerifyKeepsTheMatvecOfMLX() throws {
        try assertPrefillTailKeepsTheMatvecOfMLX(bits: Fixture.eightBits)
    }

    private func assertVerifyLogitsMatchSingleRowDecode(bits: Int) throws {
        let target = try makeTarget(bits: bits)
        let lmHead = try XCTUnwrap(target.languageModel.lmHead)

        for tokenCount in Fixture.kernelVerifyTokenCounts {
            let verify = try callAfterPrompt(
                target, count: tokenCount, flags: [mtpEmitFlagKey, mtpVerifyKeepsSnapshotsKey])
            let hidden = try XCTUnwrap(verify.state?[mtpLastHiddenStatesKey])
            Fixture.assertLogitsMatchSingleRowProjections(
                verify.logits, hidden: hidden, lmHead: lmHead,
                "\(bits)-bit, \(tokenCount) tokens")
        }
    }

    func testVerifyRowsGetTheLogitsOfSingleRowDecode() throws {
        try assertVerifyLogitsMatchSingleRowDecode(bits: Fixture.bits)
    }

    func testEightBitVerifyRowsGetTheLogitsOfSingleRowDecode() throws {
        try assertVerifyLogitsMatchSingleRowDecode(bits: Fixture.eightBits)
    }
}
