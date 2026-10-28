// Copyright © 2026 Apple Inc.
//
// The quantized projections of a speculative verify of the Qwen3.5
// vision-language model run through ``MultiRowQuantizedMatvec``, as those of
// the text model do. Every other call of 3 to 8 rows keeps the matvec of MLX.

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

    /// The text configuration of every module of these tests.
    private func textConfiguration() throws -> Qwen35Configuration.TextConfiguration {
        try JSONDecoder().decode(
            Qwen35Configuration.TextConfiguration.self,
            from: Data(Fixture.textConfigurationJSON.utf8))
    }

    /// `module` on seeded weights in bfloat16, every projection quantized.
    private func quantized<M: Module>(_ make: () throws -> M) rethrows -> M {
        let module = try withRandomState(MLXRandom.RandomState(seed: Self.weightSeed), body: make)
        module.update(parameters: module.parameters().mapValues { $0.asType(.bfloat16) })
        Fixture.quantize(module, matvecOfMLX: false)
        module.train(false)
        eval(module)
        return module
    }

    /// A vision-language target on seeded weights in bfloat16, its language
    /// model quantized.
    ///
    /// - Parameters:
    ///   - prepared: True to fuse the input projections of each gated delta
    ///     net, as the model loader does.
    ///   - matvecOfMLX: True to give every projection the matvec of MLX.
    private func makeTarget(prepared: Bool = true, matvecOfMLX: Bool = false) throws -> Qwen35 {
        let configuration = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: Fixture.textConfigurationJSON)
                    .utf8))
        let target = withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            Qwen35(configuration)
        }
        target.update(parameters: target.parameters().mapValues { $0.asType(.bfloat16) })
        Fixture.quantize(target, matvecOfMLX: matvecOfMLX) { path, _ in
            path.hasPrefix("language_model")
        }
        if prepared {
            try target.prepare()
        }
        target.train(false)
        eval(target)
        return target
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

    func testMLPRowsOfAVerifyMatchSingleRowDecode() throws {
        let mlp = quantized {
            Qwen35Language.MLP(dimensions: Fixture.hiddenSize, hiddenDimensions: 1024)
        }
        Fixture.assertVerifyRowsMatchSingleRows(width: Fixture.hiddenSize, "mlp") { [mlp($0)] }
    }

    func testGatedDeltaInputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let configuration = try textConfiguration()
        let gdn = quantized { Qwen35Language.GatedDeltaNet(configuration) }
        XCTAssertTrue(try gdn.prepareFusedInputProjection())
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, "gdn input projection"
        ) { x in
            let projected = gdn.projectInputs(x, batch: 1, sequence: x.dim(1))
            return [projected.qkv, projected.z, projected.b, projected.a]
        }
    }

    func testGatedDeltaOutputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let configuration = try textConfiguration()
        let gdn = quantized { Qwen35Language.GatedDeltaNet(configuration) }
        Fixture.assertVerifyRowsMatchSingleRows(width: gdn.valueDim, "gdn out_proj") {
            [gdn.projectOutput($0)]
        }
    }

    func testAttentionProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let configuration = try textConfiguration()
        let attention = quantized { Qwen35Language.Attention(configuration) }
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, axis: Fixture.headTokenAxis, "attention q/k/v"
        ) { x in
            let (queries, _, keys, values) = attention.projectPreRope(x)
            return [queries, keys, values]
        }
        Fixture.assertVerifyRowsMatchSingleRows(width: Fixture.hiddenSize, "attention gate") {
            [attention.projectPreRope($0).1]
        }
        let width = Fixture.attentionHeads * Fixture.headDimension
        Fixture.assertVerifyRowsMatchSingleRows(width: width, "attention o_proj") { gate in
            let heads = gate.reshaped(
                1, gate.dim(1), Fixture.attentionHeads, Fixture.headDimension
            ).transposed(0, 2, 1, 3)
            return [attention.mergeHeadsAndProject(attention: heads, gate: gate)]
        }
    }

    func testPrefillTailOutsideAVerifyKeepsTheMatvecOfMLX() throws {
        let target = try makeTarget(prepared: false)
        let reference = try makeTarget(prepared: false, matvecOfMLX: true)

        for (label, flags) in [("plain", []), ("MTP prefill", [mtpEmitFlagKey])] {
            let got = try callAfterPrompt(target, count: Fixture.rows, flags: flags).logits
            let want = try callAfterPrompt(reference, count: Fixture.rows, flags: flags).logits
            XCTAssertEqual(got.shape, want.shape, label)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0,
                "\(label): a prompt tail of \(Fixture.rows) tokens left the matvec of MLX")
        }
    }

    func testVerifyRowsGetTheLogitsOfSingleRowDecode() throws {
        let target = try makeTarget()
        let lmHead = try XCTUnwrap(target.languageModel.lmHead)

        for tokenCount in Fixture.kernelVerifyTokenCounts {
            let verify = try callAfterPrompt(
                target, count: tokenCount, flags: [mtpEmitFlagKey, mtpVerifyKeepsSnapshotsKey])
            let hidden = try XCTUnwrap(verify.state?[mtpLastHiddenStatesKey])
            Fixture.assertLogitsMatchSingleRowProjections(
                verify.logits, hidden: hidden, lmHead: lmHead, "\(tokenCount) tokens")
        }
    }
}
