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

    /// The tokens of `text`, one for each UTF-8 byte.
    ///
    /// - Parameters:
    ///   - text: The text to encode.
    ///   - addSpecialTokens: Unused: the tokenizer adds no special tokens.
    /// - Returns: The byte value of each UTF-8 byte of `text`.
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    /// The text of `tokenIds`, one UTF-8 byte for each token, with every EOS token left out.
    ///
    /// - Parameters:
    ///   - tokenIds: The tokens to decode.
    ///   - skipSpecialTokens: Unused: the EOS token is always left out.
    /// - Returns: The UTF-8 text of the bytes, or an empty string when the bytes are not UTF-8.
    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != Self.eosTokenID }.map(UInt8.init), encoding: .utf8)
            ?? ""
    }

    /// The id of the token `token` names.
    ///
    /// - Parameter token: The EOS text, or a text of one UTF-8 byte.
    /// - Returns: ``eosTokenID`` for the EOS text, the byte value for a text of one byte, and
    ///   `nil` otherwise.
    public func convertTokenToId(_ token: String) -> Int? {
        if token == Self.eosTokenText { return Self.eosTokenID }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }

    /// The text of token `id`.
    ///
    /// - Parameter id: A token of the vocabulary.
    /// - Returns: The EOS text for ``eosTokenID``, the scalar of the byte value for every other
    ///   token, and `nil` for an id outside the vocabulary.
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
    ///
    /// - Parameters:
    ///   - messages: Unused: the tests supply the prepared input.
    ///   - tools: Unused: the template renders no tools.
    ///   - additionalContext: Unused: the template has no variables.
    /// - Returns: No tokens.
    public func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
