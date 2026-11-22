// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXGuidedGeneration

/// One token for each byte, whose text is the byte's character, with EOS at 255.
private struct CharacterTokenizer: MLXLMCommon.Tokenizer {
    static let eos = 255

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.filter { $0 != Self.eos }.map(UInt8.init), encoding: .utf8) ?? ""
    }
    func convertTokenToId(_ token: String) -> Int? {
        token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eos { return "<eos>" }
        return id >= 0 && id < Self.eos ? String(UnicodeScalar(UInt8(id))) : nil
    }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// A tokenizer with a fixed list of token texts. The token at index `i` has id `i`.
private struct TextListTokenizer: MLXLMCommon.Tokenizer {
    let tokens: [String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { tokens.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { tokens.indices.contains(id) ? tokens[id] : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// The token of an ASCII character in ``CharacterTokenizer``.
private func token(_ character: Unicode.Scalar) -> Int {
    Int(UInt8(ascii: character))
}

/// The decisions of the rules of a real JSON-schema grammar, all in the hard zone, over
/// ``CharacterTokenizer``: the test commits text, then decides tokens from fixed logits.
private struct HardZoneDriver {

    /// The vocabulary of the grammar and the width of the logits.
    static let vocabularySize = 256

    /// The token budget, which the hard zone covers whole.
    static let maxTokens = 64

    /// The decisions after which a generation counts as not terminating.
    static let decisionLimit = 32

    let rules: GuidedDecisionRules
    private(set) var position: GuidedPosition
    private(set) var text = ""
    private(set) var terminated = false

    /// Compiles `schema`.
    ///
    /// - Parameters:
    ///   - schema: A JSON schema.
    ///   - fastForward: Whether the grammar fast-forwards the tokens it forces.
    init(schema: String, fastForward: Bool = true) throws {
        let host = CharacterTokenizer()
        let vocabulary = (0 ..< Self.vocabularySize).map { String(format: "<0x%02X>", $0) }
        let grammarTokenizer = try GrammarTokenizer(
            vocab: vocabulary, vocabType: .byteFallback, eosTokenId: Int32(CharacterTokenizer.eos))
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, jsonSchema: schema, fastForward: fastForward,
            hostTokenizer: host)
        let (whitespaceBias, whitespaceTokenIDs) = WhitespaceTokenBias.compute(tokenizer: host)
        rules = GuidedDecisionRules(
            constraint: constraint, maxTokens: Self.maxTokens, vocabSize: Self.vocabularySize,
            logitDim: Self.vocabularySize, stopTokenIDs: [CharacterTokenizer.eos],
            unknownTokenID: nil, completionReserve: Self.maxTokens, hardReserve: Self.maxTokens,
            closingBias: ClosingTokenBias.compute(
                tokenizer: host, eosTokenId: CharacterTokenizer.eos),
            whitespaceBias: whitespaceBias, whitespaceTokenIDs: whitespaceTokenIDs,
            stringTokens: JSONStringTokens.compute(tokenizer: host))
        position = try rules.startPosition()
    }

    /// Commits the bytes of `target` after the text so far, one at a time: the grammar's
    /// fast-forwarded tokens write some of them, which the test then does not commit.
    ///
    /// - Parameter target: The text the output reads after the commits; it extends the text so
    ///   far.
    mutating func write(_ target: String) throws {
        let bytes = Array(target.utf8)
        while text.utf8.count < bytes.count {
            try #require(
                bytes.starts(with: text.utf8),
                "the grammar wrote <<<\(text)>>>, not <<<\(target)>>>")
            try commit(token: Int(bytes[text.utf8.count]))
        }
        try #require(text == target, "the grammar wrote <<<\(text)>>>, not <<<\(target)>>>")
    }

    /// Commits `token` and the tokens the grammar fast-forwards after it, and moves to the
    /// position after them.
    mutating func commit(token: Int) throws {
        guard token != CharacterTokenizer.eos else {
            terminated = true
            return
        }
        let result = try rules.constraint.commitToken(Int32(token))
        let committed = [token] + result.tokens.map { Int($0) }
        text += String(bytes: committed.map(UInt8.init), encoding: .utf8) ?? ""
        let after = rules.after(token: token, forced: result.tokens, from: position)
        position = try rules.position(
            tokenCount: position.tokenCount + committed.count, whitespace: after.whitespace,
            stringState: after.stringState)
        terminated = result.isTerminated || position.mask.isTerminated
    }

    /// The token the rules decide at the current position from `preferences`: the raw logit of
    /// each named token, 0 for every other token.
    func decide(_ preferences: [Int: Float]) -> Int {
        var logits = [Float](repeating: 0, count: Self.vocabularySize)
        for (token, logit) in preferences {
            logits[token] = logit
        }
        let constrained = rules.constrainedLogits(MLXArray(logits), at: position)
        return argMax(constrained, axis: -1).item(Int.self)
    }

    /// Decides and commits tokens from `preferences` until the grammar terminates or
    /// ``decisionLimit`` decisions pass.
    mutating func decideUntilTerminated(_ preferences: [Int: Float]) throws {
        var decisions = 0
        while !terminated && decisions < Self.decisionLimit {
            try commit(token: decide(preferences))
            decisions += 1
        }
    }
}

/// An object with one required string.
private let noteSchema = #"""
    {"type": "object", "properties": {"note": {"type": "string"}},
     "required": ["note"], "additionalProperties": false}
    """#

/// The hard zone prefers the token that closes a string only inside an open string, through
/// the rules of a real JSON-schema grammar.
@Suite
struct HardZoneStringClosingTests {

    /// An array of strings that may be empty.
    private static let tagsSchema = #"{"type": "array", "items": {"type": "string"}}"#

    /// A model that prefers string content that is also a closing token over the quote.
    private static let contentOverQuote: [Int: Float] = [
        token("6"): 5, token("}"): 5, token("]"): 5, token("\""): 0,
    ]

    @Test("inside an open string the quote wins over the digits and the braces")
    func insideAStringTheQuoteWins() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            var driver = try HardZoneDriver(schema: noteSchema)
            try driver.write(#"{"note":"a"#)

            #expect(driver.position.stringState.closesWithQuote, "text <<<\(driver.text)>>>")
            #expect(driver.decide(Self.contentOverQuote) == token("\""))
        }
    }

    @Test("after the opening bracket of an array that may be empty, the bracket closes it")
    func afterAnOpeningBracketTheBracketWins() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            var driver = try HardZoneDriver(schema: Self.tagsSchema)
            try driver.write("[")

            #expect(!driver.position.stringState.isOpen)
            // The model leans to `]` by a little; a quote preferred outside a string, where it
            // opens one, would outweigh that lean and write `[""]`.
            let decided = driver.decide([token("\""): 0, token("]"): 1])
            #expect(decided == token("]"), "decided \(decided)")
        }
    }

    @Test("after a backslash inside a string the output still terminates")
    func afterABackslashTheOutputTerminates() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            var driver = try HardZoneDriver(schema: noteSchema)
            try driver.write(#"{"note":"a\"#)
            #expect(driver.position.stringState.isEscaped)

            try driver.decideUntilTerminated(
                Self.contentOverQuote.merging([token("\\"): 9, token("n"): 7]) { $1 })

            #expect(driver.terminated, "text <<<\(driver.text)>>>")
            let object = try JSONSerialization.jsonObject(with: Data(driver.text.utf8))
            #expect((object as? [String: String])?["note"]?.hasPrefix("a") == true)
        }
    }
}

/// The drafts of an MTP round carry the string state after each draft, and a rollback restores
/// the state of the position it returns to.
@Suite
struct GuidedDraftChainStringStateTests {

    /// The drafted text: an object whose string value has an escape open.
    private static let draftedText = #"{"note":"a\"#

    /// Whether a string is open, and whether an escape is, after each byte of ``draftedText``.
    private static let statesAfter: [(isOpen: Bool, isEscaped: Bool)] = [
        (false, false),  // {
        (true, false),  // "
        (true, false), (true, false), (true, false), (true, false),  // note
        (false, false),  // "
        (false, false),  // :
        (true, false),  // "
        (true, false),  // a
        (true, true),  // \
    ]

    /// The index of the draft the rollback returns to: the `a` inside the open value.
    private static let rollbackIndex = 9

    @Test("each draft carries the string state after it, and a restore returns to it")
    func draftsCarryTheStringStateAndRestoreReturnsToIt() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let driver = try HardZoneDriver(schema: noteSchema, fastForward: false)
            var decoder = try GuidedDecoder(
                rules: driver.rules, tokenizer: CharacterTokenizer(), diagnosticLog: false,
                bufferPoolClearing: GenerationBufferPoolClearing()
            ) { _ in true }
            let chain = GuidedDraftChain(rules: driver.rules, start: decoder.position)

            for byte in Self.draftedText.utf8 {
                chain.record(MLXArray([Int32(byte)]))
            }

            #expect(chain.failure == nil)
            try #require(chain.drafts.count == Self.statesAfter.count)
            for (index, draft) in chain.drafts.enumerated() {
                let expected = Self.statesAfter[index]
                #expect(draft.stringStateAfter.isOpen == expected.isOpen, "draft \(index)")
                #expect(draft.stringStateAfter.isEscaped == expected.isEscaped, "draft \(index)")
            }
            #expect(chain.nextPosition?.stringState == chain.drafts.last?.stringStateAfter)

            let returnedTo = chain.drafts[Self.rollbackIndex].position
            try chain.rollBack(draftsFrom: Self.rollbackIndex)
            try decoder.restore(to: returnedTo)
            #expect(decoder.position.stringState == returnedTo.stringState)
            #expect(decoder.position.stringState.closesWithQuote)
            // The next position comes from the state the decoder restored, not from the drafts.
            try decoder.advance()
            #expect(decoder.position.stringState == returnedTo.stringState)
        }
    }
}

/// The string state of JSON output over the marks of the bytes of its tokens.
@Suite
struct JSONStringTokensTests {

    /// Tokens with several marks each, the byte-fallback tokens of `"` and `\`, the bare quote,
    /// and a quote that a combining mark follows, which is one character but holds the quote
    /// byte.
    private static let tokens = [
        #"":""#, #"",""#, #""\""#, #"\\""#, "<0x22>", "<0x5C>", "\"", "\"\u{0301}", "a",
    ]

    /// The id of `text` in ``tokens``.
    private static func id(_ text: String) throws -> Int {
        try #require(tokens.firstIndex(of: text))
    }

    /// The string state after `texts`, from a closed string.
    private static func state(after texts: [String], from start: JSONStringState = .init())
        throws -> JSONStringState
    {
        let stringTokens = JSONStringTokens.compute(tokenizer: TextListTokenizer(tokens: tokens))
        var state = start
        for text in texts {
            state = stringTokens.state(after: try id(text), forced: [], from: state)
        }
        return state
    }

    @Test("the string-closing tokens are the quote and its byte-fallback token")
    func theClosingTokensAreTheQuoteAndItsByteFallback() throws {
        let stringTokens = JSONStringTokens.compute(
            tokenizer: TextListTokenizer(tokens: Self.tokens))
        #expect(stringTokens.stringClosingTokenIDs == [try Self.id("\""), try Self.id("<0x22>")])
    }

    @Test("a separator between two quotes closes a string and opens the next")
    func aSeparatorTokenClosesAndReopens() throws {
        for separator in [#"":""#, #"",""#] {
            let state = try Self.state(after: ["\"", "a", separator])
            #expect(state.closesWithQuote, "\(separator)")
        }
    }

    @Test("an escaped quote leaves the string open, and an escaped backslash does not")
    func escapesInsideOneToken() throws {
        #expect(try Self.state(after: [#""\""#]).closesWithQuote)
        #expect(!(try Self.state(after: ["\"", #"\\""#]).isOpen))
    }

    @Test("the byte-fallback tokens move the state as their bytes do")
    func byteFallbackTokens() throws {
        #expect(try Self.state(after: ["\"", "<0x5C>"]).isEscaped)
        #expect(try Self.state(after: ["\"", "<0x5C>", "<0x22>"]).closesWithQuote)
        #expect(!(try Self.state(after: ["\"", "<0x22>"]).isOpen))
    }

    @Test("a quote that a combining mark follows still opens a string")
    func aCombiningMarkAfterAQuote() throws {
        #expect(try Self.state(after: ["\"\u{0301}"]).closesWithQuote)
        #expect(!(try Self.state(after: ["\"\u{0301}", "\""]).isOpen))
    }
}
