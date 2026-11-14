// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// The token IDs of the mock vocabulary.
private enum MockToken {
    static let start = 2
    static let end = 3
    static let endOfTurn = 7
}

/// A model whose logits always peak at the end delimiter, with the
/// end-of-turn token second: the shape of the Flint fork after a status line.
private final class EndDelimiterPeakModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }

    let vocabularySize = 16

    func prepare(
        _ input: LMInput, cache _: [KVCache], state _: LMOutput.State?,
        prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let positions = inputs.dim(-1)
        // Write each position to the cache, as a real model does, so a
        // speculative iterator can trim the rejected drafts.
        let rows = zeros([1, 1, positions, 1])
        _ = cache?.first?.update(keys: rows, values: rows)
        var data = [Float](repeating: 0, count: positions * vocabularySize)
        for position in 0 ..< positions {
            data[position * vocabularySize + MockToken.end] = 10
            data[position * vocabularySize + MockToken.endOfTurn] = 5
        }
        return MLXArray(data, [1, positions, vocabularySize])
    }
}

/// A tokenizer whose `<think>` and `</think>` are one token each, and whose
/// `</thought>` is two tokens.
private struct ThinkTagTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        switch text {
        case "<think>": [MockToken.start]
        case "</think>": [MockToken.end]
        default: Array(text.utf8.map { Int($0) % 16 })
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class ClosedReasoningGuardTests: XCTestCase {

    private let logits = MLXArray([0.5, 1.0, 2.0, 9.0, 4.0] as [Float])[.newAxis, .ellipsis]

    // MARK: - The processor

    func testDoesNotMaskBeforeTheBlockCloses() {
        var processor = ClosedReasoningEndProcessor(endTokenId: MockToken.end)
        processor.prompt(MLXArray([Int32(1), Int32(MockToken.start)]))

        XCTAssertEqual(processor.process(logits: logits)[0, 3].item(Float.self), 9.0)
    }

    func testMasksTheEndDelimiterAfterItIsSampled() {
        var processor = ClosedReasoningEndProcessor(endTokenId: MockToken.end)
        processor.prompt(MLXArray([Int32(1)]))
        processor.didSample(token: MLXArray([Int32(4)]))
        XCTAssertEqual(processor.process(logits: logits)[0, 3].item(Float.self), 9.0)

        processor.didSample(token: MLXArray([Int32(MockToken.end)]))
        processor.didSample(token: MLXArray([Int32(4)]))
        let processed = processor.process(logits: logits)

        XCTAssertEqual(processed[0, 3].item(Float.self), -Float.infinity)
        XCTAssertEqual(processed[0, 0].item(Float.self), 0.5)
        XCTAssertEqual(processed[0, 4].item(Float.self), 4.0)
    }

    /// A closed, empty block at the end of the prompt (thinking off) has
    /// closed the block of the pass.
    func testPromptThatEndsWithTheEndDelimiterMasksAtOnce() {
        var processor = ClosedReasoningEndProcessor(endTokenId: MockToken.end)
        processor.prompt(MLXArray([Int32(MockToken.start), Int32(MockToken.end)]))

        XCTAssertEqual(processor.process(logits: logits)[0, 3].item(Float.self), -Float.infinity)
    }

    func testANewPromptStartsOpen() {
        var processor = ClosedReasoningEndProcessor(endTokenId: MockToken.end)
        processor.prompt(MLXArray([Int32(1)]))
        processor.didSample(token: MLXArray([Int32(MockToken.end)]))
        processor.prompt(MLXArray([Int32(1)]))

        XCTAssertEqual(processor.process(logits: logits)[0, 3].item(Float.self), 9.0)
    }

    // MARK: - Which protocols mask

    func testOnlyAnOptedInProtocolWithAOneTokenDelimiterMasks() {
        let tokenizer = ThinkTagTokenizer()
        var flintLike = ReasoningConfig.alwaysOnThinking
        flintLike.masksEndDelimiterAfterClose = true
        XCTAssertEqual(
            closedReasoningEndTokenId(reasoning: flintLike, tokenizer: tokenizer), MockToken.end)

        var multiToken = flintLike
        multiToken.endDelimiter = "</thought>"
        XCTAssertNil(closedReasoningEndTokenId(reasoning: multiToken, tokenizer: tokenizer))

        for reasoning in [
            ReasoningConfig.alwaysOnThinking, QwenReasoningProtocol.qwen3,
            QwenReasoningProtocol.qwen35, QwenReasoningProtocol.tagged,
        ] {
            XCTAssertNil(closedReasoningEndTokenId(reasoning: reasoning, tokenizer: tokenizer))
        }
        XCTAssertNil(closedReasoningEndTokenId(reasoning: nil, tokenizer: tokenizer))
    }

    func testFlintMasksItsEndDelimiter() {
        XCTAssertTrue(FlintModel.reasoning.masksEndDelimiterAfterClose)
    }

    // MARK: - Generation

    private func generate(model: any LanguageModel, speculative: Bool = false) throws -> [Int] {
        let input = LMInput(tokens: MLXArray([Int32(1), Int32(MockToken.start)]))
        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var tokens = [Int]()
        if speculative {
            var iterator = try SpeculativeTokenIterator(
                input: input, mainModel: model, draftModel: model, parameters: parameters,
                numDraftTokens: 2)
            while let token = iterator.next() { tokens.append(token) }
        } else {
            var iterator = try TokenIterator(input: input, model: model, parameters: parameters)
            while let token = iterator.next() { tokens.append(token) }
        }
        return tokens
    }

    func testTokenIteratorClosesOnceThenTakesTheNextChoice() throws {
        let model = EndDelimiterPeakModel()
        var reasoning = ReasoningConfig.alwaysOnThinking
        reasoning.masksEndDelimiterAfterClose = true
        registerClosedReasoningEndGuard(
            model: model, reasoning: reasoning, tokenizer: ThinkTagTokenizer())

        XCTAssertEqual(try generate(model: model), [3, 7, 7, 7, 7])
    }

    func testTokenIteratorOfAModelThatDoesNotOptInIsUnchanged() throws {
        let model = EndDelimiterPeakModel()
        registerClosedReasoningEndGuard(
            model: model, reasoning: .alwaysOnThinking, tokenizer: ThinkTagTokenizer())

        XCTAssertEqual(try generate(model: model), [3, 3, 3, 3, 3])
    }

    /// The verifier feeds the stateful processor each accepted token, thus a
    /// draft of a second end delimiter is rejected.
    func testSpeculativeVerificationClosesOnceThenTakesTheNextChoice() throws {
        let model = EndDelimiterPeakModel()
        var reasoning = ReasoningConfig.alwaysOnThinking
        reasoning.masksEndDelimiterAfterClose = true
        registerClosedReasoningEndGuard(
            model: model, reasoning: reasoning, tokenizer: ThinkTagTokenizer())

        XCTAssertEqual(try generate(model: model, speculative: true), [3, 7, 7, 7, 7])
    }
}
