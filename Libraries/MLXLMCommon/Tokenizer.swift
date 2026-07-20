// Copyright © 2024 Apple Inc.

import Foundation

/// A protocol for tokenizing text into token IDs and decoding token IDs into text.
public protocol Tokenizer: Sendable {

    /// Encodes text into token IDs.
    ///
    /// - Parameters:
    ///   - text: the text to tokenize
    ///   - addSpecialTokens: whether to add the tokenizer's special tokens
    ///     (e.g. a beginning-of-sequence marker) to the result
    /// - Returns: the token IDs representing the text
    func encode(text: String, addSpecialTokens: Bool) -> [Int]

    /// Decodes token IDs back into text.
    ///
    /// - Parameters:
    ///   - tokenIds: the token IDs to decode
    ///   - skipSpecialTokens: whether to omit special tokens (e.g. end-of-sequence
    ///     markers) from the decoded text
    /// - Returns: the decoded text
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String

    /// Converts a single token string to its token ID.
    ///
    /// - Parameter token: the token text, e.g. `"<|im_end|>"`
    /// - Returns: the token's ID, or `nil` when the token is not in the vocabulary
    func convertTokenToId(_ token: String) -> Int?

    /// Converts a single token ID to its token string.
    ///
    /// - Parameter id: the token ID
    /// - Returns: the token text, or `nil` when the ID is not in the vocabulary
    func convertIdToToken(_ id: Int) -> String?

    /// The beginning-of-sequence token, or `nil` when the tokenizer does not
    /// define one.
    var bosToken: String? { get }

    /// The end-of-sequence token, or `nil` when the tokenizer does not
    /// define one.
    var eosToken: String? { get }

    /// The token substituted for out-of-vocabulary input, or `nil` when the
    /// tokenizer does not define one.
    var unknownToken: String? { get }

    /// Renders the chat template over the given conversation and encodes the
    /// result, priming the model for generation (e.g. with ChatML's trailing
    /// `<|im_start|>assistant\n` header).
    ///
    /// - Parameters:
    ///   - messages: array of message dictionaries representing the conversation
    ///   - tools: optional array of tool specifications available to the model
    ///   - additionalContext: optional extra template variables
    /// - Returns: token IDs for the rendered conversation
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]

    /// Renders the chat template with explicit control over the template's
    /// generation-priming region (e.g. ChatML's trailing
    /// `<|im_start|>assistant\n` header).
    ///
    /// This is an optional capability. The default implementation returns
    /// `nil`, meaning the renderer cannot control the generation prompt;
    /// callers must treat `nil` as "no stable boundary computable" and skip
    /// any behavior that depends on it.
    ///
    /// Passing `addGenerationPrompt: false` renders the messages as past
    /// turns — the exact form they re-render as once later turns are appended
    /// — which lets callers compute the prefix of a prompt that stays stable
    /// across rounds.
    ///
    /// - Parameters:
    ///   - messages: array of message dictionaries representing the conversation
    ///   - tools: optional array of tool specifications available to the model
    ///   - additionalContext: optional extra template variables
    ///   - addGenerationPrompt: whether to append the generation-priming region
    /// - Returns: token ids for the rendered conversation, or `nil` when the
    ///   tokenizer does not support controlling the generation prompt
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int]?
}

extension Tokenizer {

    /// Encodes text into token IDs, adding the tokenizer's special tokens.
    ///
    /// Convenience for ``encode(text:addSpecialTokens:)`` with
    /// `addSpecialTokens: true`.
    ///
    /// - Parameter text: the text to tokenize
    /// - Returns: the token IDs representing the text
    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    /// Decodes token IDs back into text, keeping special tokens.
    ///
    /// Convenience for ``decode(tokenIds:skipSpecialTokens:)`` with
    /// `skipSpecialTokens: false`.
    ///
    /// - Parameter tokenIDs: the token IDs to decode
    /// - Returns: the decoded text
    public func decode(tokenIDs: [Int]) -> String {
        decode(tokenIds: tokenIDs, skipSpecialTokens: false)
    }

    /// Looks up the ID of an optional special token via ``convertTokenToId(_:)``.
    ///
    /// Shared implementation behind ``bosTokenID``, ``eosTokenID`` and
    /// ``unknownTokenID``.
    ///
    /// - Parameter token: the token text, or `nil` when the tokenizer does not
    ///   define the token
    /// - Returns: the token's ID, or `nil` when the token is undefined or not
    ///   in the vocabulary
    private func tokenID(of token: String?) -> Int? {
        guard let token else { return nil }
        return convertTokenToId(token)
    }

    /// The ID of ``bosToken``, or `nil` when the token is undefined or not in
    /// the vocabulary.
    public var bosTokenID: Int? {
        tokenID(of: bosToken)
    }

    /// The ID of ``eosToken``, or `nil` when the token is undefined or not in
    /// the vocabulary.
    public var eosTokenID: Int? {
        tokenID(of: eosToken)
    }

    /// The ID of ``unknownToken``, or `nil` when the token is undefined or not
    /// in the vocabulary.
    public var unknownTokenID: Int? {
        tokenID(of: unknownToken)
    }

    /// Renders the chat template over the conversation, priming the model for
    /// generation.
    ///
    /// Convenience for ``applyChatTemplate(messages:tools:additionalContext:)``
    /// with `tools: nil, additionalContext: nil`.
    ///
    /// - Parameter messages: array of message dictionaries representing the
    ///   conversation
    /// - Returns: token IDs for the rendered conversation
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    public func applyChatTemplate(
        messages: [[String: any Sendable]]
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
    }

    /// Renders the chat template over the conversation with tools, priming the
    /// model for generation.
    ///
    /// Convenience for ``applyChatTemplate(messages:tools:additionalContext:)``
    /// with `additionalContext: nil`.
    ///
    /// - Parameters:
    ///   - messages: array of message dictionaries representing the conversation
    ///   - tools: optional array of tool specifications available to the model
    /// - Returns: token IDs for the rendered conversation
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages, tools: tools, additionalContext: nil)
    }

    /// Renders the chat template with explicit control over the
    /// generation-priming region.
    ///
    /// Convenience for
    /// ``applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:)``
    /// with `tools: nil, additionalContext: nil`.
    ///
    /// - Parameters:
    ///   - messages: array of message dictionaries representing the conversation
    ///   - addGenerationPrompt: whether to append the generation-priming region
    /// - Returns: token IDs for the rendered conversation, or `nil` when the
    ///   tokenizer does not support controlling the generation prompt
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool
    ) throws -> [Int]? {
        try applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil,
            addGenerationPrompt: addGenerationPrompt)
    }

    /// Renders the chat template with tools and explicit control over the
    /// generation-priming region.
    ///
    /// Convenience for
    /// ``applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:)``
    /// with `additionalContext: nil`.
    ///
    /// - Parameters:
    ///   - messages: array of message dictionaries representing the conversation
    ///   - tools: optional array of tool specifications available to the model
    ///   - addGenerationPrompt: whether to append the generation-priming region
    /// - Returns: token IDs for the rendered conversation, or `nil` when the
    ///   tokenizer does not support controlling the generation prompt
    /// - Throws: `TokenizerError.missingChatTemplate` if no chat template is
    ///   configured
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        addGenerationPrompt: Bool
    ) throws -> [Int]? {
        try applyChatTemplate(
            messages: messages, tools: tools, additionalContext: nil,
            addGenerationPrompt: addGenerationPrompt)
    }

    /// Default implementation of the optional generation-prompt-controlled
    /// render: this tokenizer cannot control the generation prompt, so it
    /// opts out by returning `nil` and callers skip the dependent behavior.
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        addGenerationPrompt: Bool
    ) throws -> [Int]? {
        nil
    }
}

/// Errors thrown by ``Tokenizer`` chat-template rendering.
public enum TokenizerError: LocalizedError {
    /// The tokenizer has no chat template configured, so a conversation
    /// cannot be rendered.
    case missingChatTemplate

    /// A human-readable description of the error.
    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            "This tokenizer does not have a chat template."
        }
    }
}

/// An incremental detokenizer: tokens are appended one at a time and
/// ``IteratorProtocol/next()`` yields the newly decoded text — or `nil` when
/// the appended tokens do not yet form a complete unicode character.
public protocol StreamingDetokenizer: IteratorProtocol<String> {

    /// Appends a token to the stream to be decoded.
    ///
    /// - Parameter token: the token ID to append
    mutating func append(token: Int)
}

/// A ``StreamingDetokenizer`` that decodes by re-running the tokenizer's
/// `decode` over the current segment of tokens and emitting the suffix that
/// changed, starting a new segment at each newline.
public struct NaiveStreamingDetokenizer: StreamingDetokenizer {
    let tokenizer: any Tokenizer

    var segmentTokens = [Int]()
    var segment = ""

    /// Creates a streaming detokenizer.
    ///
    /// - Parameter tokenizer: the tokenizer used to decode the streamed tokens
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Appends a token to the stream to be decoded.
    ///
    /// - Parameter token: the token ID to append
    public mutating func append(token: Int) {
        segmentTokens.append(token)
    }

    mutating func startNewSegment() {
        let lastToken = segmentTokens.last
        segmentTokens.removeAll()
        if let lastToken {
            segmentTokens.append(lastToken)
            segment = tokenizer.decode(tokenIDs: segmentTokens)
        } else {
            segment = ""
        }
    }

    /// Returns the text newly decoded since the last call.
    ///
    /// - Returns: the new text, or `nil` when the appended tokens do not yet
    ///   decode to a complete unicode character
    public mutating func next() -> String? {
        let newSegment = tokenizer.decode(tokenIDs: segmentTokens)
        let new = newSegment.suffix(newSegment.count - segment.count)

        // if the new segment ends with REPLACEMENT CHARACTER this means
        // that the token didn't produce a complete unicode character
        if new.last == "\u{fffd}" {
            return nil
        }

        if new.hasSuffix("\n") {
            startNewSegment()
        } else {
            self.segment = newSegment
        }

        return String(new)
    }
}
