// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// A ``ScalarTokenizer`` whose chat template closes each message with an end-of-turn token,
/// writes the reasoning of a past assistant turn inside a think block before its content, and
/// opens a think block after the last message, as the Qwen 3.8 template does with `<|im_end|>`,
/// `reasoning_content` and `<think>`. It names the three markers as the Qwen protocol does, thus
/// the cache-reuse rule of that protocol finds them.
///
/// The ``Qwen35MTPFixture`` tests of a session and of the FoundationModels executor share it.
public struct QwenThinkBlockScalarTokenizer: Tokenizer {

    /// The text of the marker that closes a message, as the Qwen protocol names it.
    public static let endOfTurnMarker = "<|im_end|>"

    /// The text of the marker that opens a think block, as the Qwen protocol names it.
    public static let openThinkMarker = "<think>"

    /// The text of the marker that closes a think block, as the Qwen protocol names it.
    public static let closeThinkMarker = "</think>"

    /// The token that opens a think block.
    public let open: Int

    /// The token that closes a think block.
    public let close: Int

    /// The token that closes each message.
    public let endOfTurn: Int

    /// The tokenizer that encodes, decodes and spells the tokens.
    private let base = ScalarTokenizer(vocabularySize: Qwen35MTPFixture.vocabularySize)

    /// Creates a tokenizer whose markers are three tokens of the fixture's vocabulary.
    ///
    /// - Parameters:
    ///   - open: The token that opens a think block.
    ///   - close: The token that closes a think block.
    ///   - endOfTurn: The token that closes each message.
    public init(open: Int, close: Int, endOfTurn: Int) {
        self.open = open
        self.close = close
        self.endOfTurn = endOfTurn
    }

    public var bosToken: String? { base.bosToken }
    public var eosToken: String? { base.eosToken }
    public var unknownToken: String? { base.unknownToken }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    /// The marker token `token` names in the Qwen protocol, or the token of `base`.
    public func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case Self.endOfTurnMarker: endOfTurn
        case Self.openThinkMarker: open
        case Self.closeThinkMarker: close
        default: base.convertTokenToId(token)
        }
    }

    public func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    /// The tokens of each message, then the open think block of the generation prompt.
    public func applyChatTemplate(
        messages: [[String: any Sendable]], tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap(render) + [open]
    }

    /// The tokens of `message`, then the end-of-turn token: its content, after its reasoning
    /// inside a think block when it is an assistant message.
    ///
    /// - Parameter message: The message dictionary the message generator made.
    /// - Returns: The tokens.
    private func render(_ message: [String: any Sendable]) -> [Int] {
        let content = tokens(of: message, key: "content") + [endOfTurn]
        guard message["role"] as? String == Chat.Message.Role.assistant.rawValue else {
            return content
        }
        return [open] + tokens(of: message, key: "reasoning_content") + [close] + content
    }

    /// The tokens of the text `message` holds under `key`, or none when it holds no text there.
    ///
    /// - Parameters:
    ///   - message: The message dictionary.
    ///   - key: The key of the text.
    /// - Returns: The tokens.
    private func tokens(of message: [String: any Sendable], key: String) -> [Int] {
        encode(text: message[key] as? String ?? "", addSpecialTokens: false)
    }
}

extension Qwen35MTPFixture {

    /// An end-of-turn token, and the greedy decode of a conversation that token closes the
    /// messages of.
    public struct MarkerFreeDecode: Sendable {

        /// The token that closes each message.
        public let endOfTurn: Int

        /// The plain greedy decode after the conversation and the open think block.
        public let tokens: [Int]
    }

    /// The error of a fixture whose every decode writes a marker token.
    public struct NoMarkerFreeDecode: Error {}

    /// The first token of the vocabulary outside `messages` and `reserved` and other than
    /// `open` whose plain greedy decode of `messages` writes neither it nor `open`, thus no turn
    /// writes a marker.
    ///
    /// Each message renders as ``QwenThinkBlockScalarTokenizer`` renders a user or system
    /// message: its tokens and the end-of-turn token. The open think block follows them.
    ///
    /// - Parameters:
    ///   - messages: The tokens of each message, in order.
    ///   - reserved: Tokens of later messages, which the end-of-turn token must not be either.
    ///   - open: The token that opens a think block.
    ///   - maxTokens: The tokens the decode generates.
    /// - Returns: The end-of-turn token and the decode.
    /// - Throws: The error of the decode, or ``NoMarkerFreeDecode`` when every decode writes a
    ///   marker.
    public func markerFreeDecode(
        of messages: [[Int]], reserving reserved: [Int] = [], open: Int, maxTokens: Int
    ) async throws -> MarkerFreeDecode {
        let messageTokens = Set(messages.joined()).union(reserved)
        for endOfTurn in 0 ..< Self.vocabularySize
        where endOfTurn != open && !messageTokens.contains(endOfTurn) {
            let decode = try await decodeGreedily(
                prompt: messages.flatMap { $0 + [endOfTurn] } + [open], maxTokens: maxTokens,
                speculative: false)
            if !decode.tokens.contains(endOfTurn), !decode.tokens.contains(open) {
                return MarkerFreeDecode(endOfTurn: endOfTurn, tokens: decode.tokens)
            }
        }
        throw NoMarkerFreeDecode()
    }
}
