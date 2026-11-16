// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXScriptedLM
import XCTest

@testable import MLXLLM
@testable import MLXVLM

/// The prefill step of Qwen 3.5 when the caller sets none: the 2048-token
/// chunk of mlx-vlm and mlx-lm, on the vision-language model and on both text
/// models, on tiny random-weight models.
final class Qwen35PrefillStepTests: XCTestCase {

    /// The prefill chunk of mlx-vlm and mlx-lm, which a Qwen 3.5 prefill
    /// reads in one forward when the caller sets no step size.
    private let mlxVLMPrefillChunk = 2048

    /// The number of decoder layers of the test models: one linear (GDN)
    /// layer and one full-attention layer.
    private let layerCount = 2

    /// One layer of each interval of this many layers is a full-attention
    /// layer.
    private let fullAttentionInterval = 2

    /// The model type of the text-only checkpoint that wraps a text
    /// configuration.
    private let wrappedModelType = "qwen3_5"

    /// The JSON of the text configuration of the test models.
    private var textJSON: String {
        Qwen35FixtureJSON.text(
            mtpLayers: 0, hiddenLayers: layerCount, fullAttentionInterval: fullAttentionInterval)
    }

    /// Decodes a configuration from its JSON text.
    private func decode<Configuration: Decodable>(
        _ type: Configuration.Type, _ json: String
    ) throws -> Configuration {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    /// A prompt of `count` tokens in the vocabulary of the test models,
    /// without a batch axis, as the token iterator hands it to a prepare.
    private func prompt(count: Int) -> LMInput {
        let vocabularySize = Qwen35FixtureJSON.defaultVocabularySize
        return LMInput(tokens: MLXArray((0 ..< count).map { Int32($0 % vocabularySize) }))
    }

    /// The vision-language model.
    private func visionLanguageModel() throws -> MLXVLM.Qwen35 {
        MLXVLM.Qwen35(
            try decode(
                MLXVLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: textJSON)))
    }

    /// The text model.
    private func textModel() throws -> Qwen35TextModel {
        Qwen35TextModel(try decode(Qwen35TextConfiguration.self, textJSON))
    }

    /// The text-only model that wraps the text model.
    private func wrappedTextModel() throws -> Qwen35Model {
        Qwen35Model(
            try decode(
                MLXLLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.wrappedText(
                    modelType: wrappedModelType, textConfigJSON: textJSON)))
    }

    /// The `processed` count of each progress report of a prepare of a prompt
    /// of one mlx-vlm chunk plus the last token, with no step size from the
    /// caller.
    private func progressOfOneChunkPrompt(_ model: any LanguageModel) throws -> [Int] {
        let log = PrefillProgressLog()
        let prefill = log.recordingPrefill()
        _ = try model.prepare(
            prompt(count: mlxVLMPrefillChunk + 1),
            cache: try model.newCache(parameters: nil), state: nil, prefill: prefill)
        return log.processed
    }

    /// The vision-language model prefills one mlx-vlm chunk in one forward,
    /// and its final forward reads the last token.
    func testVisionLanguageModelReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try progressOfOneChunkPrompt(try visionLanguageModel()),
            [mlxVLMPrefillChunk, mlxVLMPrefillChunk + 1])
    }

    /// The text model prefills one mlx-vlm chunk in one forward, and leaves
    /// the last token for the iterator.
    func testTextModelReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(try progressOfOneChunkPrompt(try textModel()), [mlxVLMPrefillChunk])
    }

    /// The text-only model that wraps the text model prefills one mlx-vlm
    /// chunk in one forward, and leaves the last token for the iterator.
    func testWrappedTextModelReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try progressOfOneChunkPrompt(try wrappedTextModel()), [mlxVLMPrefillChunk])
    }

    // MARK: - Checkpoint feeds

    /// The generic prefill step, which a caller sets to keep the chunks of a
    /// checkpoint feed at that step.
    private let callerStepSize = PrefillParameters.defaultStepSize

    /// The `processed` count of each progress report of a checkpoint feed of
    /// one mlx-vlm chunk of prompt tokens, which `ChatSession` and the
    /// `MLXFoundationModels` executor run up to each message boundary.
    private func progressOfOneChunkCheckpointFeed(
        _ model: any LanguageModel, stepSize: Int? = nil
    ) throws -> [Int] {
        let log = PrefillProgressLog()
        let prefill = log.recordingPrefill(stepSize: stepSize)
        let vocabularySize = Qwen35FixtureJSON.defaultVocabularySize
        _ = try PromptCacheCheckpoint.prefill(
            (0 ..< mlxVLMPrefillChunk).map { $0 % vocabularySize },
            into: try model.newCache(parameters: nil), state: nil, model: model,
            prefill: prefill)
        return log.processed
    }

    /// A checkpoint feed of the vision-language model with no step from the
    /// caller reads one mlx-vlm chunk in one forward.
    func testVisionLanguageModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try progressOfOneChunkCheckpointFeed(try visionLanguageModel()), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed of the text model with no step from the caller reads
    /// one mlx-vlm chunk in one forward.
    func testTextModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try progressOfOneChunkCheckpointFeed(try textModel()), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed of the text-only model that wraps the text model with
    /// no step from the caller reads one mlx-vlm chunk in one forward.
    func testWrappedTextModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try progressOfOneChunkCheckpointFeed(try wrappedTextModel()), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed keeps the step a caller sets.
    func testCheckpointFeedKeepsTheStepOfTheCaller() throws {
        let progress = try progressOfOneChunkCheckpointFeed(
            try textModel(), stepSize: callerStepSize)
        XCTAssertEqual(
            progress,
            Array(stride(from: callerStepSize, through: mlxVLMPrefillChunk, by: callerStepSize)))
    }

    /// A model that forwards to the text model takes the prefill step of the
    /// text model.
    func testForwardingModelTakesThePrefillStepOfTheWrappedModel() throws {
        let model = ForwardingLanguageModel(wrapping: try textModel())
        XCTAssertEqual(model.defaultPrefillStepSize, mlxVLMPrefillChunk)
    }
}
