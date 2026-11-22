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

    init(schema: String) throws {
        let host = CharacterTokenizer()
        let vocabulary = (0 ..< Self.vocabularySize).map { String(format: "<0x%02X>", $0) }
        let grammarTokenizer = try GrammarTokenizer(
            vocab: vocabulary, vocabType: .byteFallback, eosTokenId: Int32(CharacterTokenizer.eos))
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer, jsonSchema: schema, fastForward: true,
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
            precondition(
                bytes.starts(with: text.utf8),
                "the grammar wrote <<<\(text)>>>, not <<<\(target)>>>")
            try commit(token: Int(bytes[text.utf8.count]))
        }
        precondition(text == target, "the grammar wrote <<<\(text)>>>, not <<<\(target)>>>")
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
        let stringState = rules.stringState(
            after: token, forced: result.tokens, from: position.stringState)
        position = try rules.position(
            tokenCount: position.tokenCount + committed.count,
            whitespace: rules.whitespace(after: token, from: position.whitespace),
            stringState: stringState)
        terminated = result.isTerminated || position.mask.isTerminated
    }

    /// The token the rules decide at the current position from `preferences`: the raw logit of
    /// each named character, 0 for every other token.
    func decide(_ preferences: [Character: Float]) -> Int {
        var logits = [Float](repeating: 0, count: Self.vocabularySize)
        for (character, logit) in preferences {
            logits[Int(character.asciiValue!)] = logit
        }
        let constrained = rules.constrainedLogits(MLXArray(logits), at: position)
        return argMax(constrained, axis: -1).item(Int.self)
    }

    /// Decides and commits tokens from `preferences` until the grammar terminates or
    /// ``decisionLimit`` decisions pass.
    mutating func decideUntilTerminated(_ preferences: [Character: Float]) throws {
        var decisions = 0
        while !terminated && decisions < Self.decisionLimit {
            try commit(token: decide(preferences))
            decisions += 1
        }
    }
}

/// The hard zone prefers the token that closes a string only inside an open string, through
/// the rules of a real JSON-schema grammar.
@Suite
struct HardZoneStringClosingTests {

    /// An object with one required string.
    private static let noteSchema = #"""
        {"type": "object", "properties": {"note": {"type": "string"}},
         "required": ["note"], "additionalProperties": false}
        """#

    /// An array of strings that may be empty.
    private static let tagsSchema = #"{"type": "array", "items": {"type": "string"}}"#

    /// A model that prefers string content that is also a closing token over the quote.
    private static let contentOverQuote: [Character: Float] = ["6": 5, "}": 5, "]": 5, "\"": 0]

    @Test("inside an open string the quote wins over the digits and the braces")
    func insideAStringTheQuoteWins() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            var driver = try HardZoneDriver(schema: Self.noteSchema)
            try driver.write(#"{"note":"a"#)

            #expect(driver.position.stringState.closesWithQuote, "text <<<\(driver.text)>>>")
            #expect(driver.decide(Self.contentOverQuote) == Int(Character("\"").asciiValue!))
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
            let decided = driver.decide(["\"": 0, "]": 1])
            #expect(decided == Int(Character("]").asciiValue!), "decided \(decided)")
        }
    }

    @Test("after a backslash inside a string the output still terminates")
    func afterABackslashTheOutputTerminates() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            var driver = try HardZoneDriver(schema: Self.noteSchema)
            try driver.write(#"{"note":"a\"#)
            #expect(driver.position.stringState.isEscaped)

            try driver.decideUntilTerminated(
                Self.contentOverQuote.merging(["\\": 9, "n": 7]) { $1 })

            #expect(driver.terminated, "text <<<\(driver.text)>>>")
            let object = try JSONSerialization.jsonObject(with: Data(driver.text.utf8))
            #expect((object as? [String: String])?["note"]?.hasPrefix("a") == true)
        }
    }
}

/// The string state of JSON output over the marks of its characters.
@Suite
struct JSONStringStateTests {

    private static func state(after text: String) -> JSONStringState {
        let tokens = JSONStringTokens.compute(tokenizer: CharacterTokenizer())
        return tokens.state(after: text.utf8.map(Int.init), from: JSONStringState())
    }

    @Test("a quote opens a string and the next quote closes it")
    func quotesOpenAndClose() {
        #expect(Self.state(after: #"{"a"#).closesWithQuote)
        #expect(!Self.state(after: #"{"a""#).isOpen)
    }

    @Test("an escaped quote leaves the string open")
    func anEscapedQuoteLeavesTheStringOpen() {
        #expect(Self.state(after: #""a\"#).isEscaped)
        #expect(Self.state(after: #""a\""#).closesWithQuote)
        #expect(!Self.state(after: #""a\\""#).isOpen)
    }

    @Test("the string-closing tokens are the quote and its byte-fallback token")
    func theClosingTokensAreTheQuote() {
        let tokens = JSONStringTokens.compute(tokenizer: CharacterTokenizer())
        #expect(tokens.stringClosingTokenIDs == [Int(Character("\"").asciiValue!)])
    }
}
