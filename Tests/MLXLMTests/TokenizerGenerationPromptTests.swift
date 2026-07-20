// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// Tests for the optional
/// `applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:)`
/// capability on `Tokenizer`.
///
/// The capability lets callers render a message array as past turns — without
/// the template's generation-priming region — so the executor can compute
/// which prefix of a prompt re-renders identically on later rounds.
@Suite struct TokenizerGenerationPromptTests {

    /// A minimal conformer that does NOT implement the optional
    /// generation-prompt render. The protocol-extension default must return
    /// `nil` ("renderer cannot control the generation prompt") so existing
    /// conformers opt out with zero behavior change.
    private struct MinimalTokenizer: Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            []
        }
    }

    @Test func defaultImplementationReturnsNil() throws {
        let tokenizer: any Tokenizer = MinimalTokenizer()

        let rendered = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "hello"]],
            tools: nil,
            additionalContext: nil,
            addGenerationPrompt: false)

        #expect(rendered == nil)
    }

    /// A deterministic ChatML-style conformer that implements the optional
    /// method the way the macro-generated `TokenizerBridge` does: the default
    /// render primes generation (trailing `<|im_start|>assistant\n` header),
    /// and `addGenerationPrompt: false` omits exactly that region.
    private struct ChatMLTokenizer: Tokenizer {
        static let assistantHeader = "<|im_start|>assistant\n"

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.unicodeScalars.map { Int($0.value) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(String.UnicodeScalarView(tokenIds.compactMap { Unicode.Scalar($0) }))
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        private func render(
            messages: [[String: any Sendable]], addGenerationPrompt: Bool
        ) -> String {
            var text = messages.map { message in
                let role = message["role"] as? String ?? ""
                let content = message["content"] as? String ?? ""
                return "<|im_start|>\(role)\n\(content)<|im_end|>\n"
            }.joined()
            if addGenerationPrompt {
                text += Self.assistantHeader
            }
            return text
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            encode(text: render(messages: messages, addGenerationPrompt: true))
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?,
            addGenerationPrompt: Bool
        ) throws -> [Int]? {
            encode(text: render(messages: messages, addGenerationPrompt: addGenerationPrompt))
        }
    }

    /// Contract the dependent stable-boundary logic relies on: for a
    /// ChatML-style template the no-genprompt render is a strict prefix of the
    /// with-genprompt render and lacks the trailing assistant header.
    @Test func noGenerationPromptRenderIsStrictPrefixWithoutAssistantHeader() throws {
        let tokenizer: any Tokenizer = ChatMLTokenizer()
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You are helpful."],
            ["role": "user", "content": "hello"],
        ]

        let primed = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let past = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: nil,
                addGenerationPrompt: false))

        #expect(past.count < primed.count)
        #expect(Array(primed.prefix(past.count)) == past)

        let pastText = tokenizer.decode(tokenIds: past)
        let primedText = tokenizer.decode(tokenIds: primed)
        #expect(!pastText.hasSuffix(ChatMLTokenizer.assistantHeader))
        #expect(primedText.hasSuffix(ChatMLTokenizer.assistantHeader))
    }

    /// `addGenerationPrompt: true` must reproduce the default (priming) render.
    @Test func generationPromptTrueMatchesDefaultRender() throws {
        let tokenizer: any Tokenizer = ChatMLTokenizer()
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]

        let primed = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)
        let explicit = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: nil,
                addGenerationPrompt: true))

        #expect(explicit == primed)
    }

    /// The `(messages:addGenerationPrompt:)` convenience must match the full
    /// 4-parameter render with `tools: nil, additionalContext: nil`.
    @Test func messagesAndGenerationPromptConvenienceMatchesFullRender() throws {
        let tokenizer: any Tokenizer = ChatMLTokenizer()
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]

        let full = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: nil,
                addGenerationPrompt: false))
        let convenience = try #require(
            try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false))

        #expect(convenience == full)
    }

    /// The `(messages:tools:addGenerationPrompt:)` convenience must match the
    /// full 4-parameter render with `additionalContext: nil`.
    @Test func messagesToolsAndGenerationPromptConvenienceMatchesFullRender() throws {
        let tokenizer: any Tokenizer = ChatMLTokenizer()
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]

        let full = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: nil,
                addGenerationPrompt: true))
        let convenience = try #require(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, addGenerationPrompt: true))

        #expect(convenience == full)
    }

    /// The convenience overloads must preserve the default-`nil` opt-out for
    /// conformers that do not implement the generation-prompt render.
    @Test func convenienceOverloadsReturnNilForOptedOutTokenizer() throws {
        let tokenizer: any Tokenizer = MinimalTokenizer()
        let messages: [[String: any Sendable]] = [["role": "user", "content": "hi"]]

        #expect(
            try tokenizer.applyChatTemplate(messages: messages, addGenerationPrompt: false) == nil)
        #expect(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, addGenerationPrompt: false) == nil)
    }
}
