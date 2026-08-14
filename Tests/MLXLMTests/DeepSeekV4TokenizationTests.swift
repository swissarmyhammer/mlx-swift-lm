// Copyright © 2026 Apple Inc.
//
// The DeepSeek-V4 tokenization path of card ^t56pqr4.
//
// `swift-transformers` 1.3.3 sends the `Split` pre-tokenizer pattern through
// `String.range(of:options:.regularExpression)`, and that Foundation search
// cannot match `\r` or `\n` inside a character class. Each newline therefore
// became its own piece, where the published tokenizer joins a run of newlines
// into one token. These tests pin the correction: the same published patterns,
// read by `NSRegularExpression`.
//
// Every test here is weight-free. The pre-tokenizer needs no vocabulary at
// all, and the merge step reads a synthetic vocabulary of a few entries.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite
struct DeepSeekV4TokenizationTests {

    // MARK: - Fixtures

    /// A tokenizer that answers from a small table of token texts.
    ///
    /// `encode` maps each UTF-8 byte of the text to one identifier, thus a
    /// test can tell the fallback answer from the byte-level answer.
    private struct TableTokenizer: MLXLMCommon.Tokenizer {
        /// The identifier of each known token text.
        let identifierOfToken: [String: Int]

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map { Int($0) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map { UInt8(clamping: $0) }, as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? { identifierOfToken[token] }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    /// The identifier that the synthetic vocabulary gives the newline pair.
    private static let newlinePairIdentifier = 900
    /// The identifier that the synthetic vocabulary gives the user marker.
    private static let userMarkerIdentifier = 999
    /// The identifier that the synthetic vocabulary gives the letter `a`.
    private static let letterAIdentifier = 65
    /// The first identifier that ``vocabularyWithoutTheMarker`` gives a byte
    /// character.
    ///
    /// It stands above every byte value, thus the byte-level answer and the
    /// fallback answer of ``TableTokenizer/encode(text:addSpecialTokens:)``
    /// cannot look the same.
    private static let firstByteCharacterIdentifier = 700

    /// A byte-level vocabulary that holds one marker, the letter `a`, the
    /// newline byte token and the pair of newline byte tokens.
    private static let byteLevelVocabulary = TableTokenizer(identifierOfToken: [
        DeepSeekV4ChatEncoder.SpecialToken.user: userMarkerIdentifier,
        "a": letterAIdentifier,
        "\u{010A}": 10,
        "\u{010A}\u{010A}": newlinePairIdentifier,
    ])

    /// A byte-level vocabulary that holds every byte character of the user
    /// marker and the letter `a`, and that holds no marker.
    ///
    /// The byte characters are all there, thus a path that reads the marker as
    /// ordinary text answers with identifiers. Only a path that requires the
    /// marker itself reports the failure.
    private static let vocabularyWithoutTheMarker: TableTokenizer = {
        var identifierOfToken: [String: Int] = ["a": letterAIdentifier]
        var identifier = firstByteCharacterIdentifier
        let markerByteText = DeepSeekV4ByteLevel.text(
            of: DeepSeekV4ChatEncoder.SpecialToken.user)
        for scalar in markerByteText.unicodeScalars {
            identifierOfToken[String(scalar)] = identifier
            identifier += 1
        }
        return TableTokenizer(identifierOfToken: identifierOfToken)
    }()

    // MARK: - The pre-tokenizer

    @Test("a run of newlines stays with the punctuation in front of it")
    func aNewlineRunStaysWithItsPunctuation() {
        #expect(
            DeepSeekV4PreTokenizer.pieces(of: "given.\n\n## Tools\n\nYou have")
                == ["given", ".\n\n", "##", " Tools", "\n\n", "You", " have"])
    }

    @Test("a run of carriage returns and newlines stays in one piece")
    func aCarriageReturnRunStaysInOnePiece() {
        #expect(DeepSeekV4PreTokenizer.pieces(of: "a\r\n\r\nb") == ["a", "\r\n\r\n", "b"])
    }

    @Test("whitespace that ends with newlines joins the newlines")
    func whitespaceJoinsTheNewlinesAfterIt() {
        #expect(DeepSeekV4PreTokenizer.pieces(of: "  \n   \n\n end") == ["  \n   \n\n", " end"])
    }

    @Test("a run of digits breaks into groups of three")
    func digitsBreakIntoGroupsOfThree() {
        #expect(DeepSeekV4PreTokenizer.pieces(of: "12345 items") == ["123", "45", " items"])
    }

    // MARK: - The byte-level step

    @Test("each byte reads as the character the published vocabulary spells it with")
    func eachByteReadsAsItsPublishedCharacter() {
        #expect(DeepSeekV4ByteLevel.text(of: "\n") == "\u{010A}")
        #expect(DeepSeekV4ByteLevel.text(of: " ") == "\u{0120}")
        #expect(DeepSeekV4ByteLevel.text(of: "a") == "a")
    }

    // MARK: - The merge step

    @Test("the merge takes the pair with the lowest vocabulary identifier first")
    func theMergeTakesTheLowestIdentifierFirst() {
        // "bc" holds 9 and "ab" holds 10, and the vocabulary holds no "abc".
        // A merge that takes "bc" first answers `a` and `bc`, thus [3, 9]. A
        // merge that takes "ab" first answers `ab` and `c`, thus [10, 5].
        let vocabulary = ["a": 3, "b": 4, "c": 5, "ab": 10, "bc": 9]
        #expect(
            DeepSeekV4BytePairMerge.identifiers(of: "abc", vocabulary: { vocabulary[$0] })
                == [3, 9])
    }

    @Test("the merge stops when the vocabulary holds no pair")
    func theMergeStopsWithNoPair() {
        let vocabulary = ["a": 3, "b": 4, "c": 5]
        #expect(
            DeepSeekV4BytePairMerge.identifiers(of: "abc", vocabulary: { vocabulary[$0] })
                == [3, 4, 5])
    }

    @Test("the merge reports nothing when the vocabulary holds no byte token")
    func theMergeReportsNothingWithoutByteTokens() {
        #expect(DeepSeekV4BytePairMerge.identifiers(of: "abc", vocabulary: { _ in nil }) == nil)
    }

    // MARK: - The whole path

    @Test("a marker is one identifier and a run of newlines is another")
    func aMarkerAndANewlineRunEachMakeOneIdentifier() {
        let tokenization = DeepSeekV4Tokenization(vocabulary: Self.byteLevelVocabulary)
        #expect(
            tokenization.identifiers(of: DeepSeekV4ChatEncoder.SpecialToken.user + "a\n\n")
                == [Self.userMarkerIdentifier, Self.letterAIdentifier, Self.newlinePairIdentifier])
    }

    @Test("a vocabulary with no byte tokens falls back to the wrapped tokenizer")
    func aVocabularyWithNoByteTokensFallsBack() {
        let tokenization = DeepSeekV4Tokenization(
            vocabulary: TableTokenizer(identifierOfToken: [:]))
        #expect(tokenization.identifiers(of: "ab") == Array("ab".utf8).map { Int($0) })
    }

    @Test("a marker that the vocabulary does not hold stops the byte-level path")
    func aMarkerOutsideTheVocabularyStopsTheByteLevelPath() {
        // Each marker must become exactly one identifier. This vocabulary
        // holds every byte character of the marker, thus a path that reads the
        // marker as ordinary text answers with the identifiers 700 and above.
        // The marker itself is absent, thus the byte-level path must report
        // the failure and the whole text must go to the wrapped tokenizer.
        let text = DeepSeekV4ChatEncoder.SpecialToken.user + "a"
        let tokenization = DeepSeekV4Tokenization(vocabulary: Self.vocabularyWithoutTheMarker)
        #expect(tokenization.identifiers(of: text) == Array(text.utf8).map { Int($0) })
    }
}
