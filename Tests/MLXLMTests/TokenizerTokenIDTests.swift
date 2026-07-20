// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

/// Tests for the token-ID convenience properties (`bosTokenID`, `eosTokenID`,
/// `unknownTokenID`) provided by the `Tokenizer` protocol extension: each must
/// resolve its token string through `convertTokenToId` and return `nil` when
/// the tokenizer does not define the token.
@Suite struct TokenizerTokenIDTests {

    /// A conformer with a fixed vocabulary so the convenience properties can
    /// resolve token strings to ids.
    private struct VocabTokenizer: Tokenizer {
        static let vocabulary: [String: Int] = ["<s>": 1, "</s>": 2, "<unk>": 0]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { Self.vocabulary[token] }
        func convertIdToToken(_ id: Int) -> String? {
            Self.vocabulary.first(where: { $0.value == id })?.key
        }

        var bosToken: String? { "<s>" }
        var eosToken: String? { "</s>" }
        var unknownToken: String? { "<unk>" }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// A conformer that defines none of the special tokens, so every token-id
    /// convenience must return `nil`.
    private struct TokenlessTokenizer: Tokenizer {
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
        ) throws -> [Int] { [] }
    }

    @Test func tokenIDsResolveThroughVocabulary() {
        let tokenizer: any Tokenizer = VocabTokenizer()

        #expect(tokenizer.bosTokenID == 1)
        #expect(tokenizer.eosTokenID == 2)
        #expect(tokenizer.unknownTokenID == 0)
    }

    @Test func tokenIDsAreNilWhenTokensUndefined() {
        let tokenizer: any Tokenizer = TokenlessTokenizer()

        #expect(tokenizer.bosTokenID == nil)
        #expect(tokenizer.eosTokenID == nil)
        #expect(tokenizer.unknownTokenID == nil)
    }
}
