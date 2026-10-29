// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXGuidedGeneration

/// Guided generation feeds the target every token it emits: a decision whose commit forces
/// tokens feeds the decided token, then the forced tokens. The target's cache thus holds the
/// prompt and every emitted token, and each decision reads logits conditioned on all of them.
///
/// The fixture's target decodes through ``FeedRecordingModel``, which records each token a
/// forward feeds, and its grammar writes JSON over ``ToolCallAlphabetTokenizer``, whose keys the
/// grammar forces after the decided quote that opens them.
@Suite(.serialized)
struct GuidedGenerationModelFeedTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [7, 21, 30, 12, 4]

    /// The emitted tokens no forward feeds: the token of the decision that terminates the
    /// grammar, at most.
    private static let unfedFinalTokens = 1

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases)
    func theTargetIsFedEveryEmittedToken(schema: GuidedFixtureSchema) throws {
        let fixture = try Qwen35MTPFixture()
        let recorder = FeedRecordingModel(target: fixture.target)

        let run = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil,
            target: recorder)

        let emitted = ToolCallAlphabetTokenizer().encode(text: run.text, addSpecialTokens: false)
        let generated = Array(recorder.fedTokens.dropFirst(Self.prompt.count))
        #expect(!run.sink.fastForwardTokenIDs.isEmpty)
        #expect(Array(recorder.fedTokens.prefix(Self.prompt.count)) == Self.prompt)
        #expect(emitted.starts(with: generated))
        // A decision whose commit terminates the grammar emits its token and stops, so no
        // forward feeds that last token. A generation that stops any other way fed them all.
        #expect(emitted.count - generated.count <= Self.unfedFinalTokens)
    }
}

/// A target that records each token its forwards feed, in order, and decodes through the model
/// it wraps.
///
/// It prepares no prompt itself, so the prompt feeds through ``callAsFunction(_:cache:state:)``
/// like every other forward. A class, because the loop holds the model by reference, and the
/// test reads the record after the generation.
private final class FeedRecordingModel: Module, LanguageModel {

    /// The model each forward runs.
    private let target: Qwen35TextModel

    /// The tokens the forwards fed, in order.
    private(set) var fedTokens: [Int] = []

    /// Wraps `target`.
    ///
    /// - Parameter target: The model each forward runs.
    init(target: Qwen35TextModel) {
        self.target = target
    }

    func prepare(
        _ input: LMInput, cache _: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try target.newCache(parameters: parameters)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        record(inputs)
        return target(inputs, cache: cache)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        record(input.tokens)
        return target(input, cache: cache, state: state)
    }

    /// Appends the tokens of `tokens` to ``fedTokens``.
    private func record(_ tokens: MLXArray) {
        fedTokens += tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
    }
}

#endif
