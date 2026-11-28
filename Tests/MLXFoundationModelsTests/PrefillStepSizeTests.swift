// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels
import MLXLMCommon
import MLXLMTestSupport

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
            processor: FixedPromptInputProcessor(), tokenizer: GuidedByteTokenizer())
        let grammar = try await MLXLanguageModel.Executor.guidedGrammar(
            kind: .json, source: schema, reserveSchemaJSON: schema, maxTokens: maxTokens,
            modelID: modelID, context: context)
        let input = try await context.processor.prepare(input: UserInput(prompt: ""))
        try MLXLanguageModel.Executor.$prefillStepSize.withValue(prefillStepSize) {
            _ = try MLXLanguageModel.Executor.runGuidedLoop(
                input: input, checkpointBoundaries: ExecutorCheckpointBoundaries(),
                grammar: grammar,
                maxTokens: maxTokens,
                promptCache: ExecutorPromptCacheSlot(nil), speculation: nil, context: context
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

#endif
