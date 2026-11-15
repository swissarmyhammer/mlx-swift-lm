// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXGuidedGeneration

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
