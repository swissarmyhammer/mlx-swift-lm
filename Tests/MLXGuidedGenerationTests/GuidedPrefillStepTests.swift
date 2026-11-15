// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Records the prefill step its `prepare` resolves, the way a model resolves it: with its own
/// default step when it has one, and with the generic default otherwise. Its logits prefer EOS,
/// so the grammar decides the output.
private final class PrefillStepRecordingModel: Module, LanguageModel {

    /// The logit of the EOS token, high enough to win every decision the grammar leaves open.
    private static let eosLogit: Float = 20

    /// The step the model prefills with when the caller sets none, or `nil` for a model without
    /// a default of its own.
    private let ownDefaultStepSize: Int?

    /// The step the last `prepare` resolved, or `nil` before the first `prepare`.
    private(set) var resolvedStepSize: Int?

    /// Makes a model whose own default step is `ownDefaultStepSize`.
    ///
    /// - Parameter ownDefaultStepSize: The model's own default step, or `nil` for none.
    init(ownDefaultStepSize: Int?) {
        self.ownDefaultStepSize = ownDefaultStepSize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        resolvedStepSize =
            ownDefaultStepSize.map { prefill.resolvedStepSize(defaultStepSize: $0) }
            ?? prefill.resolvedStepSize()
        return .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.size
        var row = [Float](repeating: 0, count: GuidedByteTokenizer.vocabularySize)
        row[GuidedByteTokenizer.eosTokenID] = Self.eosLogit
        let rows = Array(repeating: row, count: positions).flatMap { $0 }
        return LMOutput(
            logits: MLXArray(rows).reshaped(1, positions, GuidedByteTokenizer.vocabularySize))
    }
}

/// A guided generation leaves the prefill step to the model unless its caller sets one.
@Suite(.serialized)
struct GuidedPrefillStepTests {

    /// A model default that differs from ``PrefillParameters/defaultStepSize``.
    private static let ownDefaultStepSize = PrefillParameters.defaultStepSize * 4

    /// The prompt of each generation.
    private static let prompt: [Int32] = [1, 2, 3]

    /// The token budget of each generation.
    private static let maxTokens = 16

    /// The grammar of each generation: one fixed byte.
    private static let grammar = #"root ::= "A""#

    /// Runs one guided generation on `model` with the loop's default prefill parameters.
    ///
    /// - Parameter model: The model of the generation.
    /// - Throws: The error of the grammar or of the generation.
    private static func generate(on model: PrefillStepRecordingModel) throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let tokenizer = GuidedByteTokenizer()
            let vocabulary = (0 ..< GuidedByteTokenizer.vocabularySize).map {
                String(format: "<0x%02X>", $0)
            }
            let grammarTokenizer = try GrammarTokenizer(
                vocab: vocabulary, vocabType: .byteFallback,
                eosTokenId: Int32(GuidedByteTokenizer.eosTokenID))
            let constraint = try GrammarConstraint(
                tokenizer: grammarTokenizer, grammar: grammar, fastForward: false,
                hostTokenizer: tokenizer)
            let context = ModelContext(
                configuration: ModelConfiguration(id: "prefill-step"), model: model,
                processor: UnusedProcessor(), tokenizer: tokenizer)
            try GuidedGenerationLoop.run(
                input: LMInput(tokens: MLXArray(prompt)), context: context,
                constraint: constraint, maxTokens: maxTokens,
                vocabSize: GuidedByteTokenizer.vocabularySize
            ) { _ in true }
        }
    }

    @Test
    func aModelWithItsOwnDefaultStepPrefillsWithIt() throws {
        let model = PrefillStepRecordingModel(ownDefaultStepSize: Self.ownDefaultStepSize)

        try Self.generate(on: model)

        #expect(model.resolvedStepSize == Self.ownDefaultStepSize)
    }

    @Test
    func aModelWithoutADefaultStepPrefillsWithTheGenericDefault() throws {
        let model = PrefillStepRecordingModel(ownDefaultStepSize: nil)

        try Self.generate(on: model)

        #expect(model.resolvedStepSize == PrefillParameters.defaultStepSize)
    }
}
