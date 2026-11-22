// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Whether the committed output of a guided generation stands inside an open JSON string, and
/// whether a backslash escapes the next character of that string.
struct JSONStringState: Equatable, Sendable {

    /// Whether a string is open: an opening `"` came, and its closing `"` did not.
    private(set) var isOpen = false

    /// Whether the last character of the open string is an escaping backslash, thus the next
    /// character is part of the escape and a `"` there does not close the string.
    private(set) var isEscaped = false

    /// Whether a `"` at this state closes a string.
    var closesWithQuote: Bool { isOpen && !isEscaped }

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

/// The tokens of a vocabulary that move the string state of JSON output, and the tokens that
/// close an open string.
///
/// The hard zone of a guided generation prefers ``stringClosingTokenIDs`` while a string is open:
/// inside a string the grammar allows `}`, `]` and the digits as content, and a model that keeps
/// writing them never closes the string. Outside a string a `"` opens one, and gets no
/// preference there.
public struct JSONStringTokens: Sendable {

    /// What one byte of a token is to a JSON string.
    enum Mark: Sendable {
        /// `"`: opens or closes a string, or ends an escape.
        case quote
        /// `\`: starts or ends an escape inside a string.
        case backslash
        /// Any other byte: ends an escape inside a string.
        case other
    }

    /// The byte of `"`.
    private static let quoteByte = UInt8(ascii: "\"")

    /// The byte of `\`.
    private static let backslashByte = UInt8(ascii: "\\")

    /// The text of the token that closes a string.
    private static let quote = "\""

    /// The byte-fallback token texts of `"` and `\`, as SentencePiece vocabularies write them.
    private static let byteFallbackMarks: [String: Mark] = ["<0x22>": .quote, "<0x5C>": .backslash]

    /// No string tokens: the string state never opens, and the hard zone prefers no token.
    public static let empty = JSONStringTokens(stringClosingTokenIDs: [], marks: [:])

    /// The tokens that close an open string: the token whose text is exactly `"`, and the
    /// byte-fallback token of that byte (`<0x22>`).
    public let stringClosingTokenIDs: Set<Int>

    /// The marks of each token whose bytes hold a `"` or a `\`, in order, with each run of other
    /// bytes read as one ``Mark/other``. Every other token reads as one ``Mark/other``.
    let marks: [Int: [Mark]]

    /// Creates the string tokens from their parts.
    ///
    /// - Parameters:
    ///   - stringClosingTokenIDs: The tokens that close an open string.
    ///   - marks: The marks of each token whose bytes hold a `"` or a `\`.
    init(stringClosingTokenIDs: Set<Int>, marks: [Int: [Mark]]) {
        self.stringClosingTokenIDs = stringClosingTokenIDs
        self.marks = marks
    }

    /// The string tokens of `tokenizer`'s vocabulary.
    ///
    /// Each token text is read as UTF-8 bytes, not as characters: a `"` that a combining mark
    /// follows is one character, but its quote byte still moves the state.
    ///
    /// - Parameter tokenizer: The tokenizer whose token texts are read.
    /// - Returns: The string tokens.
    public static func compute(tokenizer: any Tokenizer) -> JSONStringTokens {
        var closing: Set<Int> = []
        var marks: [Int: [Mark]] = [:]
        for (id, token) in TokenizerVocabulary.tokens(of: tokenizer).enumerated() {
            if let mark = byteFallbackMarks[token] {
                marks[id] = [mark]
                if mark == .quote { closing.insert(id) }
                continue
            }
            if token == quote { closing.insert(id) }
            if let tokenMarks = Self.marks(of: token) { marks[id] = tokenMarks }
        }
        return JSONStringTokens(stringClosingTokenIDs: closing, marks: marks)
    }

    /// The marks of the bytes of `token`, or nil when it holds no `"` and no `\`.
    private static func marks(of token: String) -> [Mark]? {
        var marks: [Mark] = []
        var holdsMark = false
        for byte in token.utf8 {
            switch byte {
            case quoteByte:
                marks.append(.quote)
                holdsMark = true
            case backslashByte:
                marks.append(.backslash)
                holdsMark = true
            default:
                if marks.last != .other { marks.append(.other) }
            }
        }
        return holdsMark ? marks : nil
    }

    /// The state after a committed `token` and the tokens the grammar fast-forwarded after it.
    ///
    /// - Parameters:
    ///   - token: The committed token.
    ///   - forced: The tokens the grammar fast-forwarded after it.
    ///   - state: The state before `token`.
    /// - Returns: The state after the forced tokens.
    func state(after token: Int, forced: [Int32], from state: JSONStringState) -> JSONStringState {
        var state = state
        read(token, into: &state)
        for forcedToken in forced {
            read(Int(forcedToken), into: &state)
        }
        return state
    }

    /// Moves `state` over the marks of `token`.
    private func read(_ token: Int, into state: inout JSONStringState) {
        guard let tokenMarks = marks[token] else {
            state.read(.other)
            return
        }
        for mark in tokenMarks {
            state.read(mark)
        }
    }
}
