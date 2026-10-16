// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon

/// A tokenizer whose vocabulary is a list of pieces: the markers of a chat
/// protocol, then each character of a small alphabet.
///
/// The encoder tries the pieces in order at each position, thus a marker is
/// one token and every other character is its own token. A character that is
/// not in the vocabulary is the unknown token. The tokenizer renders no chat
/// template: a test wraps it, or gives it a template of its own.
///
/// Shared by the `ChatSession` suites of the Qwen 3.8 history render and of
/// the DeepSeek-V4 history render.
struct PieceTokenizer: Tokenizer {

    /// Each character that the vocabulary holds after the markers.
    static let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,".map(String.init)

    /// The pieces of the vocabulary, in the order the encoder tries them.
    let pieces: [String]

    /// The marker that ends a turn, which is also the token that stops a
    /// generation.
    let endOfTurn: String

    /// - Parameters:
    ///   - markers: the markers of the chat protocol, in the order the encoder
    ///     tries them. A marker that is a prefix of another comes after it.
    ///   - endOfTurn: the marker that ends a turn.
    init(markers: [String], endOfTurn: String) {
        self.pieces = markers + Self.alphabet
        self.endOfTurn = endOfTurn
    }

    /// The size of the vocabulary: each piece, and the unknown token.
    var vocabularySize: Int { pieces.count + 1 }

    var bosToken: String? { nil }
    var eosToken: String? { endOfTurn }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { pieces.count }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let index = pieces.firstIndex(where: { rest.hasPrefix($0) }) {
                tokens.append(index)
                rest = rest.dropFirst(pieces[index].count)
            } else {
                tokens.append(pieces.count)
                rest = rest.dropFirst()
            }
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.compactMap(convertIdToToken).joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        pieces.firstIndex(of: token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        pieces.indices.contains(id) ? pieces[id] : nil
    }
}
