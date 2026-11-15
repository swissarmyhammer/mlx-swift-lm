// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXNN
import Testing

@testable import MLXFoundationModels
import MLXLMCommon

@Suite("Prefill step size")
struct PrefillStepSizeTests {
    @Test func makeParametersSetsPrefillStepSize() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: nil,
            prefillStepSize: 8)

        #expect(parameters.prefill.stepSize == 8)
    }

    @Test func nilPrefillStepSizeKeepsModelDefault() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let parameters = MLXLanguageModel.Executor.makeParameters(
            maxTokens: 32,
            requestedTemperature: nil,
            samplingConfiguration: nil,
            prefillStepSize: nil)

        #expect(parameters.prefill.stepSize == nil)
    }

    @Test func initStoresPrefillStepSize() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "org/repo"),
            prefillStepSize: 8,
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: stubLoad())

        #expect(model.prefillStepSize == 8)
        #expect(makeStubModel("org/repo").prefillStepSize == nil)
    }
}

/// A guided pass, as schema output and required tool calls run it, leaves the prefill step to
/// the model unless the ``MLXLanguageModel/prefillStepSize`` of the model it serves sets one.
@Suite("Guided prefill step size", .serialized)
struct GuidedPrefillStepSizeTests {

    /// A model default that differs from ``PrefillParameters/defaultStepSize``.
    private static let ownDefaultStepSize = PrefillParameters.defaultStepSize * 4

    /// A step the caller sets, unlike either default.
    private static let callerStepSize = 8

    /// The token budget of each pass.
    private static let maxTokens = 16

    /// The schema of each pass.
    private static let schema = #"{"type":"boolean"}"#

    /// The model identifier the grammar caches of each pass key on.
    private static let modelID = "guided-prefill-step-size"

    /// Runs one guided pass on a model whose own default step is ``ownDefaultStepSize``, for a
    /// served model whose prefill step is `prefillStepSize`.
    ///
    /// - Parameter prefillStepSize: The prefill step of the served model, or `nil` for none.
    /// - Returns: The step the model's `prepare` resolved.
    /// - Throws: The error of the grammar or of the pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func resolvedStepSize(prefillStepSize: Int?) async throws -> Int? {
        let model = PrefillStepRecordingModel(ownDefaultStepSize: ownDefaultStepSize)
        let context = ModelContext(
            configuration: ModelConfiguration(id: modelID), model: model,
            processor: FixedPromptInputProcessor(), tokenizer: ToolCallAlphabetTokenizer())
        let grammar = try await MLXLanguageModel.Executor.guidedGrammar(
            kind: .json, source: schema, reserveSchemaJSON: schema, maxTokens: maxTokens,
            modelID: modelID, context: context)
        let input = try await context.processor.prepare(input: UserInput(prompt: ""))
        try MLXLanguageModel.Executor.$prefillStepSize.withValue(prefillStepSize) {
            _ = try MLXLanguageModel.Executor.runGuidedLoop(
                input: input, grammar: grammar, maxTokens: maxTokens, speculation: nil,
                context: context
            ) { _ in }
        }
        return model.resolvedStepSize
    }

    @Test func aGuidedPassWithoutAStepPrefillsWithTheModelDefault() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        #expect(try await Self.resolvedStepSize(prefillStepSize: nil) == Self.ownDefaultStepSize)
    }

    @Test func aGuidedPassPrefillsWithTheStepOfTheServedModel() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        #expect(
            try await Self.resolvedStepSize(prefillStepSize: Self.callerStepSize)
                == Self.callerStepSize)
    }
}

/// Records the prefill step its `prepare` resolves with its own default step, as a model with a
/// default of its own resolves it. Its logits prefer EOS, so the grammar decides the output.
private final class PrefillStepRecordingModel: Module, MLXLMCommon.LanguageModel {

    /// The logit of the EOS token, high enough to win every decision the grammar leaves open.
    private static let eosLogit: Float = 20

    /// The tokens of the vocabulary.
    private static let vocabularySize = ToolCallAlphabetTokenizer.alphabet.count

    /// The step the model prefills with when the caller sets none.
    private let ownDefaultStepSize: Int

    /// The step the last `prepare` resolved, or `nil` before the first `prepare`.
    private(set) var resolvedStepSize: Int?

    /// Makes a model whose own default step is `ownDefaultStepSize`.
    ///
    /// - Parameter ownDefaultStepSize: The model's own default step.
    init(ownDefaultStepSize: Int) {
        self.ownDefaultStepSize = ownDefaultStepSize
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        resolvedStepSize = prefill.resolvedStepSize(defaultStepSize: ownDefaultStepSize)
        return .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.size
        var row = [Float](repeating: 0, count: Self.vocabularySize)
        row[Self.vocabularySize - 1] = Self.eosLogit
        let rows = Array(repeating: row, count: positions).flatMap { $0 }
        return LMOutput(logits: MLXArray(rows).reshaped(1, positions, Self.vocabularySize))
    }
}

#endif
