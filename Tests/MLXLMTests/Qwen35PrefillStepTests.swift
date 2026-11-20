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

    /// A checkpoint feed of the vision-language model with no step from the
    /// caller reads one mlx-vlm chunk in one forward.
    func testVisionLanguageModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                try visionLanguageModel(), tokenCount: mlxVLMPrefillChunk), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed of the text model with no step from the caller reads
    /// one mlx-vlm chunk in one forward.
    func testTextModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                try textModel(), tokenCount: mlxVLMPrefillChunk), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed of the text-only model that wraps the text model with
    /// no step from the caller reads one mlx-vlm chunk in one forward.
    func testWrappedTextModelCheckpointFeedReadsOneMLXVLMChunkInOneForward() throws {
        XCTAssertEqual(
            try PrefillProgressLog.progressOfCheckpointFeed(
                try wrappedTextModel(), tokenCount: mlxVLMPrefillChunk), [mlxVLMPrefillChunk])
    }

    /// A checkpoint feed keeps the step a caller sets.
    func testCheckpointFeedKeepsTheStepOfTheCaller() throws {
        let progress = try PrefillProgressLog.progressOfCheckpointFeed(
            try textModel(), tokenCount: mlxVLMPrefillChunk, stepSize: callerStepSize)
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

    // MARK: - The limit below the head-dimension split of MLX

    /// The head dimension of Qwen3.5-9B and Qwen3.8-27B, at which MLX can
    /// select its `dsplit` attention kernel.
    private let splitHeadDimension = 256

    /// The JSON of a text configuration with a head dimension of 256. The
    /// fixture has 2 query heads and 1 KV head, thus its heads do not make
    /// whole head groups of ``PrefillAttentionKernel``.
    private var splitHeadTextJSON: String {
        Qwen35FixtureJSON.text(
            mtpLayers: 0, hiddenLayers: layerCount, fullAttentionInterval: fullAttentionInterval,
            headDimension: splitHeadDimension)
    }

    /// The largest forward of a prepare of a prompt of one mlx-vlm chunk plus
    /// the last token, from its progress reports.
    private func largestForward(_ model: any LanguageModel, stepSize: Int? = nil) throws -> Int {
        let log = PrefillProgressLog()
        let prefill = log.recordingPrefill(stepSize: stepSize)
        _ = try model.prepare(
            prompt(count: mlxVLMPrefillChunk + 1),
            cache: try model.newCache(parameters: nil), state: nil, prefill: prefill)
        let processed = log.processed
        XCTAssertFalse(processed.isEmpty)
        return zip(processed, [0] + processed).map { $0 - $1 }.max() ?? 0
    }

    /// The heads of Qwen3.5-9B (16 query heads, 4 KV heads) do not make
    /// whole head groups, thus its prefill goes to MLX and gets the limit.
    func testPolicyLimitsTheStepOfQwen35NineB() {
        XCTAssertEqual(
            Qwen35PrefillPolicy.maximumStepSize(queryHeads: 16, kvHeads: 4, headDimension: 256),
            Qwen35PrefillPolicy.largestStepBelowHeadDimensionSplit)
        XCTAssertLessThan(Qwen35PrefillPolicy.largestStepBelowHeadDimensionSplit, 1024)
    }

    /// The heads of Qwen3.8-27B (24 query heads, 4 KV heads) make whole head
    /// groups, thus its own kernel does the prefill and it keeps its step.
    func testPolicyKeepsTheStepOfQwen38TwentySevenB() {
        XCTAssertNil(
            Qwen35PrefillPolicy.maximumStepSize(queryHeads: 24, kvHeads: 4, headDimension: 256))
    }

    /// MLX selects the `dsplit` kernel only at a head dimension of 256, thus
    /// another head dimension gets no limit.
    func testPolicyKeepsTheStepAtAnotherHeadDimension() {
        XCTAssertNil(
            Qwen35PrefillPolicy.maximumStepSize(queryHeads: 16, kvHeads: 4, headDimension: 128))
    }

    /// The text model, the text-only model that wraps it, and the
    /// vision-language model with a head dimension of 256 forward at most
    /// the limit at a time, also when the caller sets a larger step.
    func testModelsWithSplitHeadDimensionForwardAtMostTheLimit() throws {
        let text = Qwen35TextModel(try decode(Qwen35TextConfiguration.self, splitHeadTextJSON))
        let wrapped = Qwen35Model(
            try decode(
                MLXLLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.wrappedText(
                    modelType: wrappedModelType, textConfigJSON: splitHeadTextJSON)))
        let visionLanguage = MLXVLM.Qwen35(
            try decode(
                MLXVLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: splitHeadTextJSON)))
        let limit = Qwen35PrefillPolicy.largestStepBelowHeadDimensionSplit
        let models: [any LanguageModel] = [text, wrapped, visionLanguage]
        for model in models {
            XCTAssertEqual(model.maximumPrefillStepSize, limit)
            XCTAssertLessThanOrEqual(try largestForward(model), limit)
            XCTAssertLessThanOrEqual(
                try largestForward(model, stepSize: mlxVLMPrefillChunk), limit)
        }
    }

    /// The test models of the other tests have a small head dimension, thus
    /// they get no limit.
    func testModelsWithSmallHeadDimensionHaveNoLimit() throws {
        XCTAssertNil(try textModel().maximumPrefillStepSize)
        XCTAssertNil(try wrappedTextModel().maximumPrefillStepSize)
        XCTAssertNil(try visionLanguageModel().maximumPrefillStepSize)
    }
}
