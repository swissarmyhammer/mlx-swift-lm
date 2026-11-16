// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
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

    /// The `processed` count of each progress report of a prepare of a prompt
    /// of one mlx-vlm chunk plus the last token, with no step size from the
    /// caller.
    private func progressOfOneChunkPrompt(_ model: any LanguageModel) throws -> [Int] {
        let log = PrefillProgressLog()
        var prefill = PrefillParameters()
        prefill.progress = { processed, _ in log.record(processed) }
        _ = try model.prepare(
            prompt(count: mlxVLMPrefillChunk + 1),
            cache: try model.newCache(parameters: nil), state: nil, prefill: prefill)
        return log.processed
    }

    /// The vision-language model prefills one mlx-vlm chunk in one forward,
    /// and its final forward reads the last token.
    func testVisionLanguageModelReadsOneMLXVLMChunkInOneForward() throws {
        let model = MLXVLM.Qwen35(
            try decode(
                MLXVLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.visionLanguage(textConfigJSON: textJSON)))
        XCTAssertEqual(
            try progressOfOneChunkPrompt(model), [mlxVLMPrefillChunk, mlxVLMPrefillChunk + 1])
    }

    /// The text model prefills one mlx-vlm chunk in one forward, and leaves
    /// the last token for the iterator.
    func testTextModelReadsOneMLXVLMChunkInOneForward() throws {
        let model = Qwen35TextModel(try decode(Qwen35TextConfiguration.self, textJSON))
        XCTAssertEqual(try progressOfOneChunkPrompt(model), [mlxVLMPrefillChunk])
    }

    /// The text-only model that wraps the text model prefills one mlx-vlm
    /// chunk in one forward, and leaves the last token for the iterator.
    func testWrappedTextModelReadsOneMLXVLMChunkInOneForward() throws {
        let model = Qwen35Model(
            try decode(
                MLXLLM.Qwen35Configuration.self,
                Qwen35FixtureJSON.wrappedText(
                    modelType: wrappedModelType, textConfigJSON: textJSON)))
        XCTAssertEqual(try progressOfOneChunkPrompt(model), [mlxVLMPrefillChunk])
    }
}
