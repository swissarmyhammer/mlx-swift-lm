// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// A tokenizer that spells each token as one Unicode private-use scalar, for models on random
/// weights.
///
/// Token `n` decodes to the scalar `U+E000 + n`, and that scalar encodes back to `n`, thus
/// every generated reply encodes to the tokens that generated it. A later chat turn then
/// renders the earlier reply to the same tokens, and a prompt cache can reuse it. A scalar
/// outside that block encodes to its value modulo the vocabulary size, so any prompt text maps
/// to tokens the model knows. There is no end-of-sequence token: generation runs until its
/// token budget.
public struct ScalarTokenizer: Tokenizer {

    /// The scalar value that token 0 decodes to, the first scalar of the private-use block.
    public static let firstScalar = 0xE000

    /// The number of tokens of the model.
    public let vocabularySize: Int

    /// Makes a tokenizer for a model of `vocabularySize` tokens.
    ///
    /// - Parameter vocabularySize: The number of tokens of the model.
    public init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
    }

    /// No beginning-of-sequence token.
    public var bosToken: String? { nil }

    /// No end-of-sequence token, thus generation runs until its token budget.
    public var eosToken: String? { nil }

    /// No unknown token: every scalar encodes to a token of the vocabulary.
    public var unknownToken: String? { nil }

    /// The text of one token.
    ///
    /// - Parameter token: A token of the vocabulary.
    /// - Returns: The private-use scalar of `token`.
    public func text(for token: Int) -> String {
        String(Character(Unicode.Scalar(UInt32(Self.firstScalar + token)) ?? "\u{FFFD}"))
    }

    /// The tokens of `text`, one for each scalar.
    ///
    /// - Parameters:
    ///   - text: The text to encode.
    ///   - addSpecialTokens: Unused: the tokenizer has no special tokens.
    /// - Returns: The token of each scalar of `text`.
    public func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { scalar in
            let token = Int(scalar.value) - Self.firstScalar
            return (0 ..< vocabularySize).contains(token)
                ? token : Int(scalar.value) % vocabularySize
        }
    }

    /// The text of `tokenIds`, one private-use scalar for each token.
    ///
    /// - Parameters:
    ///   - tokenIds: The tokens to decode.
    ///   - skipSpecialTokens: Unused: the tokenizer has no special tokens.
    /// - Returns: The text.
    public func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map(text(for:)).joined()
    }

    /// No token has a name: the tokenizer has no special tokens.
    ///
    /// - Parameter token: The name of a token.
    /// - Returns: `nil`.
    public func convertTokenToId(_: String) -> Int? { nil }

    /// The text of token `id`.
    ///
    /// - Parameter id: A token of the vocabulary.
    /// - Returns: The private-use scalar of `id`.
    public func convertIdToToken(_ id: Int) -> String? { text(for: id) }

    /// The tokens of the contents of `messages`, one after the other, with no role markers.
    ///
    /// - Parameters:
    ///   - messages: The messages of the conversation.
    ///   - tools: Unused: the template renders no tools.
    ///   - additionalContext: Unused: the template has no variables.
    /// - Returns: The tokens of the conversation.
    public func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            encode(text: message["content"] as? String ?? "", addSpecialTokens: false)
        }
    }
}

/// An input processor that renders the messages of a `UserInput` with the chat template of a
/// test tokenizer, such as ``ScalarTokenizer``.
public struct ScalarTokenizerInputProcessor: UserInputProcessor {

    /// The tokenizer that renders the messages.
    public let tokenizer: any Tokenizer

    /// The configuration of the model of the processor.
    public let configuration: ModelConfiguration

    /// Makes a processor that renders with `tokenizer`.
    ///
    /// - Parameters:
    ///   - tokenizer: The tokenizer that renders the messages.
    ///   - configuration: The configuration of the model of the processor.
    public init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    /// The tokens of the messages of `input`.
    ///
    /// - Parameter input: The input of one turn.
    /// - Returns: The rendered tokens.
    /// - Throws: The error of the chat template.
    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        return LMInput(
            tokens: MLXArray(
                try tokenizer.applyChatTemplate(
                    messages: messages, tools: input.tools,
                    additionalContext: input.additionalContext)))
    }
}
