// Copyright © 2026 Apple Inc.
//
// The quantized projections of a Qwen3.5 speculative verify run through
// ``MultiRowQuantizedMatvec``: each token row of a verify gets the same bits
// as the plain decode of that token alone. Every other call of 3 to 8 rows,
// such as the prompt tail a chat prefills after a checkpoint, keeps the
// matvec of MLX.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen35MultiRowProjectionTests: XCTestCase {

    private typealias Fixture = MultiRowProjectionFixture

    /// The seed of the weights.
    private static let weightSeed: UInt64 = 20_261_028

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_029

    /// The seed of the tokens after the prompt: a verify or a prompt tail.
    private static let followingSeed: UInt64 = 20_261_030

    /// The prompt tokens the cache holds before the verify or the tail.
    private static let promptLength = 8

    /// A model on seeded weights in bfloat16, quantized.
    ///
    /// - Parameters:
    ///   - prepared: True to fuse the input projections of each gated delta
    ///     net, as the model loader does.
    ///   - matvecOfMLX: True to give every projection the matvec of MLX.
    private func makeModel(
        prepared: Bool = true, matvecOfMLX: Bool = false
    ) throws -> Qwen35TextModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(Fixture.textConfigurationJSON.utf8))
        return try Fixture.seededQuantized(
            seed: Self.weightSeed, matvecOfMLX: matvecOfMLX,
            prepare: { model in
                if prepared {
                    try model.prepare()
                }
            },
            { Qwen35TextModel(configuration) })
    }

    /// The seeded tokens of a call, shape `[1, count]`.
    private func tokens(
        _ count: Int, seed: UInt64, of model: Qwen35TextModel
    ) -> LMInput.Text {
        let tokens = seededTokens(count, vocabularySize: model.vocabularySize, seed: seed)
        return LMInput.Text(tokens: tokens[.newAxis])
    }

    /// A cache that holds the prompt.
    private func promptCache(_ model: Qwen35TextModel) throws -> [KVCache] {
        let cache = try model.newCache(parameters: nil)
        let prompt = tokens(Self.promptLength, seed: Self.promptSeed, of: model)
        eval(model(prompt, cache: cache, state: nil).logits)
        return cache
    }

    /// The logits of a prompt tail of ``MultiRowProjectionFixture/rows``
    /// tokens after the prompt: a call of 3 to 8 rows that is not a verify.
    private func prefillTail(
        _ model: Qwen35TextModel, state: LMOutput.State?
    ) throws -> MLXArray {
        let cache = try promptCache(model)
        let tail = tokens(Fixture.rows, seed: Self.followingSeed, of: model)
        let logits = model(tail, cache: cache, state: state).logits
        eval(logits)
        return logits
    }

    /// The logits and the hidden rows of one verify of `tokenCount` tokens
    /// after the prompt, on the compiled or the general path.
    private func verify(
        _ model: Qwen35TextModel, tokenCount: Int, compiled: Bool
    ) throws -> (logits: MLXArray, hidden: MLXArray) {
        let cache = try promptCache(model)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        let output = model.forward(
            tokens(tokenCount, seed: Self.followingSeed, of: model), cache: cache, state: state,
            lastTokenOnly: false, useCompiledDecode: compiled)
        let hidden = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
        eval(output.logits, hidden)
        return (output.logits, hidden)
    }

    func testMLPRowsOfAVerifyMatchSingleRowDecode() throws {
        let model = try makeModel()
        let layer = model.model.layers[0]
        Fixture.assertVerifyRowsMatchSingleRows(width: Fixture.hiddenSize, "mlp") {
            [layer.mlp($0)]
        }
    }

    func testGatedDeltaInputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let model = try makeModel()
        let gdn = try XCTUnwrap(model.model.layers[0].linearAttn)
        XCTAssertTrue(gdn.hasFusedInputProjection)
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, "gdn input projection"
        ) { x in
            let projected = gdn.projectInputs(x, batch: 1, sequence: x.dim(1))
            return [projected.qkv, projected.z, projected.b, projected.a]
        }
    }

    func testGatedDeltaOutputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let model = try makeModel()
        let gdn = try XCTUnwrap(model.model.layers[0].linearAttn)
        Fixture.assertVerifyRowsMatchSingleRows(width: gdn.valueDim, "gdn out_proj") {
            [gdn.projectOutput($0)]
        }
    }

    func testAttentionProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let model = try makeModel()
        let attention = try XCTUnwrap(model.model.layers[1].selfAttn)
        Fixture.assertVerifyRowsMatchSingleRows(
            width: Fixture.hiddenSize, axis: Fixture.headTokenAxis, "attention q/k/v"
        ) { x in
            let (queries, _, keys, values) = attention.projectPreRope(x)
            return [queries, keys, values]
        }
        Fixture.assertVerifyRowsMatchSingleRows(width: Fixture.hiddenSize, "attention gate") {
            [attention.projectPreRope($0).1]
        }
    }

    func testAttentionOutputProjectionRowsOfAVerifyMatchSingleRowDecode() throws {
        let model = try makeModel()
        let attention = try XCTUnwrap(model.model.layers[1].selfAttn)
        let width = Fixture.attentionHeads * Fixture.headDimension
        Fixture.assertVerifyRowsMatchSingleRows(width: width, "attention o_proj") { gate in
            let heads = gate.reshaped(
                1, gate.dim(1), Fixture.attentionHeads, Fixture.headDimension
            ).transposed(0, 2, 1, 3)
            return [attention.mergeHeadsAndProject(attention: heads, gate: gate)]
        }
    }

    func testPrefillTailOutsideAVerifyKeepsTheMatvecOfMLX() throws {
        let model = try makeModel(prepared: false)
        let reference = try makeModel(prepared: false, matvecOfMLX: true)
        var drafterPrefill = LMOutput.State()
        drafterPrefill[mtpEmitFlagKey] = true

        for (label, state) in [("plain", nil), ("MTP prefill", drafterPrefill)] {
            let got = try prefillTail(model, state: state)
            let want = try prefillTail(reference, state: state)
            XCTAssertEqual(got.shape, want.shape, label)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0,
                "\(label): a prompt tail of \(Fixture.rows) tokens left the matvec of MLX")
        }
    }

    func testVerifyRowsGetTheLogitsOfSingleRowDecode() throws {
        let model = try makeModel()
        let lmHead = try XCTUnwrap(model.lmHead)

        for tokenCount in Fixture.kernelVerifyTokenCounts {
            for compiled in [false, true] {
                let verify = try verify(model, tokenCount: tokenCount, compiled: compiled)
                Fixture.assertLogitsMatchSingleRowProjections(
                    verify.logits, hidden: verify.hidden, lmHead: lmHead,
                    "\(tokenCount) tokens, compiled \(compiled)")
            }
        }
    }

    func testCompiledVerifyThroughTheKernelMatchesTheGeneralPath() throws {
        let model = try makeModel()
        let tolerance = AttentionKernelTolerance.bfloat16

        for tokenCount in Fixture.kernelVerifyTokenCounts {
            let general = try verify(model, tokenCount: tokenCount, compiled: false)
            let compiled = try verify(model, tokenCount: tokenCount, compiled: true)
            for (what, got, want) in [
                ("logits", compiled.logits, general.logits),
                ("hidden rows", compiled.hidden, general.hidden),
            ] {
                XCTAssertEqual(got.shape, want.shape, "\(tokenCount) tokens \(what)")
                let gap = AttentionKernelTolerance.worstGap(got, want)
                XCTAssertLessThanOrEqual(
                    gap, tolerance, "\(tokenCount) tokens \(what): worst gap \(gap)")
            }
        }
        XCTAssertGreaterThan(model.model.compiledVerifySegmentCount, 0)
    }
}
