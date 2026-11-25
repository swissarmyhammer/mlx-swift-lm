// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// A cluster model over Inkling 2 computes the logits a local Inkling 2 computes, bit for bit.
///
/// Inkling 2's own `prepare` runs every prompt token but the last as one chunk, and the last
/// token alone. A cluster model must feed the worker the same forwards, or the bfloat16 sums
/// of the two runs differ, and the expert routing of the mixture of experts turns that
/// difference into logits that differ by more than the KL gate allows.
@Suite("A cluster model over Inkling 2 computes the local logits", .serialized)
struct ClusterInklingParityTests {

    /// The unpadded vocabulary of the tiny model: the last dimension of its logits.
    private static let vocabularySize = 48

    /// The logits of the prompt and of each fed token, from `model` with a fresh cache.
    private static func logits(
        of model: any LanguageModel, prompt: [Int32], fed: [Int32]
    ) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        var rows: [MLXArray] = []
        switch try model.prepare(
            LMInput(tokens: MLXArray(prompt)), cache: cache, state: nil,
            prefill: PrefillParameters())
        {
        case .logits(let output):
            rows.append(output.logits[0..., -1, 0...])
        case .tokens(let rest):
            var tokens = rest.tokens
            if tokens.ndim == 1 { tokens = tokens[.newAxis] }
            let output = model(LMInput.Text(tokens: tokens), cache: cache, state: nil)
            rows.append(output.logits[0..., -1, 0...])
        }
        for token in fed {
            let output = model(
                LMInput.Text(tokens: MLXArray([token])[.newAxis]), cache: cache, state: nil)
            rows.append(output.logits[0..., -1, 0...])
        }
        return rows.map { $0.asType(.float32) }
    }

    @Test("the prompt and every fed token give the local logits", arguments: [5, 40])
    func everyStepGivesTheLocalLogits(promptLength: Int) throws {
        let model = try Inkling2ModelTests.makeModel(bfloat16: true)
        let prompt = (0 ..< promptLength).map { Int32(($0 * 7 + 3) % Self.vocabularySize) }
        let fed = (0 ..< 12).map { Int32(($0 * 5 + 1) % Self.vocabularySize) }

        let local = try Self.logits(of: model, prompt: prompt, fed: fed)
        let worker = InProcessRemoteModelWorker(model: model, vocabularySize: Self.vocabularySize)
        let clustered = try Self.logits(
            of: try ClusterLanguageModel.connect(over: worker), prompt: prompt, fed: fed)

        #expect(clustered.count == local.count)
        for (step, (a, b)) in zip(local, clustered).enumerated() {
            let difference = abs(a - b).max().item(Float.self)
            #expect(difference == 0, "step \(step): max |Δlogit| \(difference)")
        }
    }
}
