// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// The closed, empty think block that the Qwen 3.5 chat template writes after
/// its generation prompt when thinking is off.
private let qwenClosedThinkBlock = "<think>\n\n</think>\n\n"

/// The chat template flag that turns Qwen thinking on and off.
private let thinkingFlag = "enable_thinking"

/// The ChatML header that opens the assistant turn.
private let assistantHeader = "<|im_start|>assistant\n"

/// A conversation that comes before the generation prompt.
private let conversation =
    "<|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"

/// A tokenizer with the two Qwen think tags as special tokens and one token for
/// two newlines, like the byte-pair merge of the real Qwen vocabulary.
///
/// A special tag stops the merge of the text around it, thus `<think>\n` and
/// `\n</think>` each give their newline as one token, while text between two tags
/// that holds two newlines gives one token for both.
private struct ThinkTagTokenizer: MLXLMCommon.Tokenizer {

    /// The special tags and their token IDs. Each ID is above every byte value.
    static let specialTokens: [String: Int] = ["<think>": 1_000, "</think>": 1_001]

    /// The token ID of two newlines.
    static let doubleNewline = 1_002

    /// The text of ``doubleNewline``.
    static let doubleNewlineText = "\n\n"

    /// Encodes each special tag and each pair of newlines as one token, and
    /// every other character as its UTF-8 bytes.
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let (tag, id) = Self.specialTokens.first(where: { rest.hasPrefix($0.key) }) {
                tokens.append(id)
                rest = rest.dropFirst(tag.count)
            } else if rest.hasPrefix(Self.doubleNewlineText) {
                tokens.append(Self.doubleNewline)
                rest = rest.dropFirst(Self.doubleNewlineText.count)
            } else {
                tokens.append(contentsOf: rest.prefix(1).utf8.map { Int($0) })
                rest = rest.dropFirst()
            }
        }
        return tokens
    }

    /// Joins the text of each token, special tags included.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { convertIdToToken($0) ?? "" }.joined()
    }

    /// The ID of a special tag, or nil for any other text.
    func convertTokenToId(_ token: String) -> Int? {
        Self.specialTokens[token]
    }

    /// The text of a special tag, of the pair of newlines, or of one byte.
    func convertIdToToken(_ id: Int) -> String? {
        if let tag = Self.specialTokens.first(where: { $0.value == id })?.key { return tag }
        if id == Self.doubleNewline { return Self.doubleNewlineText }
        guard let byte = UInt8(exactly: id) else { return nil }
        return String(decoding: [byte], as: UTF8.self)
    }

    /// The tokenizer has no beginning-of-sequence token.
    var bosToken: String? { nil }
    /// The tokenizer has no end-of-sequence token.
    var eosToken: String? { nil }
    /// The tokenizer has no unknown token.
    var unknownToken: String? { nil }

    /// The tests encode their text directly, thus no template renders.
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite
struct ClosedReasoningBlockTests {

    private let tokenizer = ThinkTagTokenizer()

    // MARK: - The protocols

    @Test func `the Qwen 3.5 protocol closes an empty think block when thinking is off`() {
        #expect(
            QwenReasoningProtocol.qwen35.promptStrategy
                == .templateFlag(
                    key: thinkingFlag, defaultOn: true,
                    thinkingOff: .closedBlock(qwenClosedThinkBlock)))
    }

    @Test func `the other Qwen protocols pass the template flag as false`() {
        // Their templates read the flag only in the generation prompt, thus the
        // flag changes no earlier token.
        #expect(
            QwenReasoningProtocol.tagged.promptStrategy
                == .templateFlag(key: thinkingFlag, defaultOn: true, thinkingOff: .templateFlagOff))
        #expect(
            QwenReasoningProtocol.qwen3.promptStrategy
                == .templateFlag(key: thinkingFlag, defaultOn: true, thinkingOff: .templateFlagOff))
    }

    // MARK: - The strategy

    @Test func `a closed block strategy keeps the template flag on when thinking is off`() throws {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: thinkingFlag, defaultOn: true, thinkingOff: .closedBlock(qwenClosedThinkBlock))
        let context = try strategy.additionalContext(forThinkingEnabled: false)
        #expect(context?[thinkingFlag] as? Bool == true)
        #expect(strategy.closedBlock(forThinkingEnabled: false) == qwenClosedThinkBlock)
    }

    @Test func `a closed block strategy writes no block when thinking is on`() throws {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: thinkingFlag, defaultOn: true, thinkingOff: .closedBlock(qwenClosedThinkBlock))
        #expect(
            try strategy.additionalContext(forThinkingEnabled: true)?[thinkingFlag] as? Bool == true
        )
        #expect(strategy.closedBlock(forThinkingEnabled: true) == nil)
        #expect(strategy.closedBlock(forThinkingEnabled: nil) == nil)
    }

    @Test
    func `a closed block strategy whose default is off closes the block when no level is given`() {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: thinkingFlag, defaultOn: false, thinkingOff: .closedBlock(qwenClosedThinkBlock))
        #expect(strategy.closedBlock(forThinkingEnabled: nil) == qwenClosedThinkBlock)
    }

    @Test func `a flag strategy writes no block when thinking is off`() {
        let strategy = ReasoningPromptStrategy.templateFlag(key: thinkingFlag, defaultOn: true)
        #expect(strategy.closedBlock(forThinkingEnabled: false) == nil)
        #expect(ReasoningPromptStrategy.alwaysOn.closedBlock(forThinkingEnabled: false) == nil)
        #expect(ReasoningPromptStrategy.none.closedBlock(forThinkingEnabled: false) == nil)
    }

    // MARK: - Closing the open block of a render

    @Test
    func `closing a render that primes an open block gives the tokens of the thinking off render`()
    {
        let thinkingOn = tokenizer.encode(
            text: conversation + assistantHeader + "<think>\n", addSpecialTokens: false)
        let thinkingOff = tokenizer.encode(
            text: conversation + assistantHeader + qwenClosedThinkBlock, addSpecialTokens: false)

        let closed = QwenReasoningProtocol.qwen35.closingReasoning(
            in: thinkingOn, with: qwenClosedThinkBlock, tokenizer: tokenizer)

        #expect(closed == thinkingOff)
    }

    @Test func `closing a render with no open block appends the closed block`() {
        let generationPrompt = tokenizer.encode(
            text: conversation + assistantHeader, addSpecialTokens: false)
        let expected = tokenizer.encode(
            text: conversation + assistantHeader + qwenClosedThinkBlock, addSpecialTokens: false)

        let closed = QwenReasoningProtocol.qwen35.closingReasoning(
            in: generationPrompt, with: qwenClosedThinkBlock, tokenizer: tokenizer)

        #expect(closed == expected)
    }

    @Test func `closing a render keeps an earlier closed block of the history`() {
        let history =
            conversation + assistantHeader + qwenClosedThinkBlock + "Hello<|im_end|>\n"
        let rendered = tokenizer.encode(text: history + assistantHeader, addSpecialTokens: false)
        let expected = tokenizer.encode(
            text: history + assistantHeader + qwenClosedThinkBlock, addSpecialTokens: false)

        let closed = QwenReasoningProtocol.qwen35.closingReasoning(
            in: rendered, with: qwenClosedThinkBlock, tokenizer: tokenizer)

        #expect(closed == expected)
    }
}
