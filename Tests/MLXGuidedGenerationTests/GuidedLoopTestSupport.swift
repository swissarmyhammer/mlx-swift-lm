// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Byte-level tokenizer: the id of a token is its byte value, and the last byte value is EOS.
struct GuidedByteTokenizer: MLXLMCommon.Tokenizer {

    /// One token for each byte value.
    static let vocabularySize = 256

    /// The id of the end-of-sequence token: the last byte value.
    static let eosTokenID = vocabularySize - 1

    /// The text of the end-of-sequence token.
    private static let eosTokenText = "<eos>"

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != Self.eosTokenID }.map(UInt8.init), encoding: .utf8)
            ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == Self.eosTokenText { return Self.eosTokenID }
        return token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eosTokenID { return Self.eosTokenText }
        return (0 ..< Self.eosTokenID).contains(id) ? String(UnicodeScalar(UInt8(id))) : nil
    }

    var bosToken: String? { nil }
    var eosToken: String? { Self.eosTokenText }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A processor the guided-loop tests never call: each test supplies the prepared input.
struct UnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("the test supplies the prepared input")
    }
}
