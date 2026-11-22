// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// The token texts of a tokenizer's vocabulary, which the bias builders of guided generation
/// (``ClosingTokenBias``, ``WhitespaceTokenBias``, ``JSONStringTokens``) all read.
enum TokenizerVocabulary {

    /// The vocabulary size at which a scan stops, against a tokenizer that names every id.
    static let maximumSize = 500_000

    /// The text of each token of `tokenizer`, by id: from id 0 up to the first id with no text,
    /// or up to ``maximumSize``.
    ///
    /// - Parameter tokenizer: The tokenizer whose vocabulary is read.
    /// - Returns: The token texts, indexed by token id.
    static func tokens(of tokenizer: any Tokenizer) -> [String] {
        var tokens: [String] = []
        while tokens.count < maximumSize, let token = tokenizer.convertIdToToken(tokens.count) {
            tokens.append(token)
        }
        return tokens
    }
}
