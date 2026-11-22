// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Whether the committed output of a guided generation stands inside an open JSON string, and
/// whether a backslash escapes the next character of that string.
public struct JSONStringState: Equatable, Sendable {

    /// Whether a string is open: an opening `"` came, and its closing `"` did not.
    public private(set) var isOpen = false

    /// Whether the last character of the open string is an escaping backslash, thus the next
    /// character is part of the escape and a `"` there does not close the string.
    public private(set) var isEscaped = false

    /// The state before any output: no string open.
    public init() {}

    /// Whether a `"` at this state closes a string.
    public var closesWithQuote: Bool { isOpen && !isEscaped }

    /// The state after one character of the output.
    ///
    /// - Parameter mark: What the character is to a JSON string.
    mutating func read(_ mark: JSONStringTokens.Mark) {
        switch mark {
        case .quote:
            if !isOpen {
                isOpen = true
            } else if isEscaped {
                isEscaped = false
            } else {
                isOpen = false
            }
        case .backslash:
            if isOpen { isEscaped.toggle() }
        case .other:
            isEscaped = false
        }
    }
}

/// The tokens of a vocabulary that move the string state of JSON output
/// (``JSONStringState``), and the tokens that close an open string.
///
/// The hard zone of a guided generation prefers ``stringClosingTokenIDs`` while a string is open:
/// inside a string the grammar allows `}`, `]` and the digits as content, and a model that keeps
/// writing them never closes the string. Outside a string a `"` opens one, and gets no
/// preference there.
public struct JSONStringTokens: Sendable {

    /// What one character of a token is to a JSON string.
    enum Mark: Sendable {
        /// `"`: opens or closes a string, or ends an escape.
        case quote
        /// `\`: starts or ends an escape inside a string.
        case backslash
        /// Any other character: ends an escape inside a string.
        case other
    }

    /// The vocabulary size past which the scan stops, as ``ClosingTokenBias`` stops.
    private static let maximumVocabularySize = 500_000

    /// The text of the token that closes a string.
    private static let quote = "\""

    /// The byte-fallback token texts of `"` and `\`, as SentencePiece vocabularies write them.
    private static let byteFallbackMarks: [String: Mark] = ["<0x22>": .quote, "<0x5C>": .backslash]

    /// No string tokens: the string state never opens, and the hard zone prefers no token.
    public static let empty = JSONStringTokens(stringClosingTokenIDs: [], marks: [:])

    /// The tokens whose text is exactly `"`, which close an open string.
    public let stringClosingTokenIDs: Set<Int>

    /// The marks of each token whose text holds a `"` or a `\`, in order. Every other token
    /// reads as one ``Mark/other`` character.
    let marks: [Int: [Mark]]

    /// Creates the string tokens from their parts.
    ///
    /// - Parameters:
    ///   - stringClosingTokenIDs: The tokens that close an open string.
    ///   - marks: The marks of each token whose text holds a `"` or a `\`.
    init(stringClosingTokenIDs: Set<Int>, marks: [Int: [Mark]]) {
        self.stringClosingTokenIDs = stringClosingTokenIDs
        self.marks = marks
    }

    /// The string tokens of `tokenizer`'s vocabulary.
    ///
    /// - Parameter tokenizer: The tokenizer whose token texts are read.
    /// - Returns: The string tokens.
    public static func compute(tokenizer: any Tokenizer) -> JSONStringTokens {
        var closing: Set<Int> = []
        var marks: [Int: [Mark]] = [:]
        var id = 0
        while id < maximumVocabularySize, let token = tokenizer.convertIdToToken(id) {
            if token == quote || byteFallbackMarks[token] == .quote {
                closing.insert(id)
            }
            if let mark = byteFallbackMarks[token] {
                marks[id] = [mark]
            } else if token.contains(quote) || token.contains("\\") {
                marks[id] = token.map { character in
                    switch character {
                    case "\"": .quote
                    case "\\": .backslash
                    default: .other
                    }
                }
            }
            id += 1
        }
        return JSONStringTokens(stringClosingTokenIDs: closing, marks: marks)
    }

    /// The state after `tokens`, from `state`.
    ///
    /// - Parameters:
    ///   - tokens: The committed tokens, in order.
    ///   - state: The state before them.
    /// - Returns: The state after them.
    func state(after tokens: some Sequence<Int>, from state: JSONStringState) -> JSONStringState {
        var state = state
        for token in tokens {
            for mark in marks[token] ?? [.other] {
                state.read(mark)
            }
        }
        return state
    }
}
