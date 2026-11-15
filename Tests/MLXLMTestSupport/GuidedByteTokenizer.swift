// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Byte-level tokenizer: the id of a token is its byte value, and the last byte value is EOS.
public struct GuidedByteTokenizer: MLXLMCommon.Tokenizer {

    /// One token for each byte value.
    public static let vocabularySize = 256

    /// The id of the end-of-sequence token: the last byte value.
    public static let eosTokenID = vocabularySize - 1

    /// The text of the end-of-sequence token.
    private static let eosTokenText = "<eos>"

    /// Makes the tokenizer.
    public init() {}

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != Self.eosTokenID }.map(UInt8.init), encoding: .utf8)
            ?? ""
    }

    public func convertTokenToId(_ token: String) -> Int? {
        if token == Self.eosTokenText { return Self.eosTokenID }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }

    public func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eosTokenID { return Self.eosTokenText }
        return (0 ..< Self.eosTokenID).contains(id) ? String(UnicodeScalar(UInt8(id))) : nil
    }

    /// No beginning-of-sequence token.
    public var bosToken: String? { nil }

    /// The text of the end-of-sequence token.
    public var eosToken: String? { Self.eosTokenText }

    /// No unknown token: every byte value is a token.
    public var unknownToken: String? { nil }

    /// No chat template: the tests supply the prepared input.
    public func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
