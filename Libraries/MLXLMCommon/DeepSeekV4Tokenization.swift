// Copyright © 2026 Apple Inc.
//
// The DeepSeek-V4 tokenization path, from card ^t56pqr4.
//
// `swift-transformers` 1.3.3 runs the `Split` pre-tokenizer of a
// `tokenizer.json` through `String.split(by:options:includeSeparators:)`, which
// loops on `String.range(of:options:.regularExpression)`. That Foundation
// search cannot match a carriage return or a newline inside a character class,
// thus `[\r\n]*` and `[\r\n]+` matched nothing and each newline became its own
// piece. The published DeepSeek-V4 tokenizer joins a run of newlines into one
// token, and it joins that run onto the punctuation in front of it. A prompt
// with a blank line therefore reached the model in a token shape the model
// never saw in training.
//
// This file runs the SAME published patterns through `NSRegularExpression`,
// which reads them correctly, and it finishes the three steps that the
// published `tokenizer.json` states after the split: the byte-level spelling,
// the byte-pair merge, and the markers that stand outside both. It reads the
// loaded tokenizer only through `convertTokenToId(_:)`.

import Foundation

/// Compiles one pattern that this file holds as a constant.
///
/// Each pattern is a transcription of the published `tokenizer.json`, thus a
/// pattern that does not compile is a programming error and not a condition
/// that a caller can meet at run time.
///
/// - Parameter pattern: the regular expression to compile.
/// - Returns: the compiled expression.
private func compiledExpression(of pattern: String) -> NSRegularExpression {
    guard let expression = try? NSRegularExpression(pattern: pattern) else {
        preconditionFailure("the DeepSeek-V4 pattern does not compile: \(pattern)")
    }
    return expression
}

/// Reads a text as the matches of an expression and the gaps between them.
///
/// This is the `Isolated` behaviour of a `Split` pre-tokenizer: the reader
/// gets every part of the text, in order, and nothing is lost.
///
/// - Parameters:
///   - text: the text to read.
///   - expression: the expression that finds the matches.
///   - reader: gets each part and `true` when that part is a match.
private func readParts(
    of text: String, with expression: NSRegularExpression,
    reader: (_ part: String, _ isMatch: Bool) -> Void
) {
    let textAsNSString = text as NSString
    var nextPartStart = 0
    expression.enumerateMatches(
        in: text, range: NSRange(location: 0, length: textAsNSString.length)
    ) { match, _, _ in
        guard let match, match.range.length > 0 else { return }
        if match.range.location > nextPartStart {
            let gap = NSRange(
                location: nextPartStart, length: match.range.location - nextPartStart)
            reader(textAsNSString.substring(with: gap), false)
        }
        reader(textAsNSString.substring(with: match.range), true)
        nextPartStart = match.range.location + match.range.length
    }
    if nextPartStart < textAsNSString.length {
        reader(textAsNSString.substring(from: nextPartStart), false)
    }
}

/// The byte-level spelling that the published DeepSeek-V4 vocabulary uses.
///
/// The vocabulary spells each byte with one character, so that a token text
/// holds no byte that a `String` cannot carry. A newline reads `Ċ` (`U+010A`)
/// and a space reads `Ġ` (`U+0120`). This is the `bytes_to_unicode` map of
/// HuggingFace, in the forward direction.
enum DeepSeekV4ByteLevel {

    /// The first byte of the printable ASCII range that keeps its own code
    /// point.
    private static let firstPrintableASCIIByte = 0x21
    /// The last byte of the printable ASCII range that keeps its own code
    /// point.
    private static let lastPrintableASCIIByte = 0x7E
    /// The first byte of the lower Latin-1 range that keeps its own code
    /// point.
    private static let firstLowerLatin1Byte = 0xA1
    /// The last byte of the lower Latin-1 range that keeps its own code point.
    private static let lastLowerLatin1Byte = 0xAC
    /// The first byte of the upper Latin-1 range that keeps its own code
    /// point.
    private static let firstUpperLatin1Byte = 0xAE
    /// The last byte of the upper Latin-1 range that keeps its own code point.
    private static let lastUpperLatin1Byte = 0xFF
    /// The first code point that spells a byte which keeps no code point of
    /// its own.
    private static let firstMovedCodePoint: UInt32 = 0x100
    /// The number of byte values.
    private static let byteCount = 256

    /// The character that spells each byte, in byte order.
    private static let characterOfByte: [Character] = {
        var characters: [Character] = []
        characters.reserveCapacity(byteCount)
        var nextMovedCodePoint = firstMovedCodePoint
        for byte in 0 ..< byteCount {
            let keepsItsOwnCodePoint =
                (byte >= firstPrintableASCIIByte && byte <= lastPrintableASCIIByte)
                || (byte >= firstLowerLatin1Byte && byte <= lastLowerLatin1Byte)
                || (byte >= firstUpperLatin1Byte && byte <= lastUpperLatin1Byte)
            if keepsItsOwnCodePoint {
                characters.append(Character(Unicode.Scalar(UInt8(byte))))
            } else {
                characters.append(Character(scalar(of: nextMovedCodePoint)))
                nextMovedCodePoint += 1
            }
        }
        return characters
    }()

    /// The scalar of one moved code point.
    ///
    /// The moved code points run from `U+0100`, thus every one of them is a
    /// scalar and a failure here is a programming error.
    ///
    /// - Parameter codePoint: the code point to read.
    /// - Returns: the scalar.
    private static func scalar(of codePoint: UInt32) -> Unicode.Scalar {
        guard let scalar = Unicode.Scalar(codePoint) else {
            preconditionFailure("the byte-level map holds a code point that is not a scalar")
        }
        return scalar
    }

    /// Spells a text the way the vocabulary spells it.
    ///
    /// - Parameter text: the text to spell.
    /// - Returns: one character for each UTF-8 byte of the text.
    static func text(of text: String) -> String {
        var spelledText = ""
        spelledText.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            spelledText.append(characterOfByte[Int(byte)])
        }
        return spelledText
    }
}

/// The pre-tokenizer that the published DeepSeek-V4 `tokenizer.json` states.
///
/// The published `pre_tokenizer` is a sequence of three `Split` steps, each
/// with the `Isolated` behaviour, and one `ByteLevel` step whose `use_regex`
/// is false. The `ByteLevel` step therefore splits nothing, and
/// ``DeepSeekV4ByteLevel`` does its work. What is left is the three splits.
enum DeepSeekV4PreTokenizer {

    /// The three `Split` patterns of the published `pre_tokenizer`, in the
    /// published order.
    ///
    /// The published file writes a carriage return and a newline as the
    /// characters themselves. `\r` and `\n` here mean the same two characters
    /// to the regular-expression engine.
    static let publishedPatterns: [String] = [
        ##"\p{N}{1,3}"##,
        ##"[一-龥぀-ゟ゠-ヿ]+"##,
        ##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##,
    ]

    /// The compiled form of ``publishedPatterns``.
    private static let expressions = publishedPatterns.map(compiledExpression(of:))

    /// Breaks a text into the pieces that the published pre-tokenizer makes.
    ///
    /// - Parameter text: the text to break.
    /// - Returns: the pieces, in order. Joined together they give the text
    ///   back.
    static func pieces(of text: String) -> [String] {
        var pieces = [text]
        for expression in expressions {
            var nextPieces: [String] = []
            for piece in pieces {
                readParts(of: piece, with: expression) { part, _ in nextPieces.append(part) }
            }
            pieces = nextPieces
        }
        return pieces
    }
}

/// The byte-pair merge of the published DeepSeek-V4 vocabulary.
///
/// The published `tokenizer.json` holds a merge list, and a loaded
/// `MLXLMCommon.Tokenizer` does not publish it. It does not have to: the
/// checkpoint gives each merge result the next free identifier, thus the
/// identifier of a merge result grows with the merge order. Measured over all
/// 127741 merges of the published file, the identifier order and the merge
/// order agree, thus "take the neighbouring pair whose joined text holds the
/// LOWEST identifier" is the published merge rule read another way.
enum DeepSeekV4BytePairMerge {

    /// Merges a byte-level text into the tokens of the vocabulary.
    ///
    /// - Parameters:
    ///   - byteText: the text in the spelling of ``DeepSeekV4ByteLevel``.
    ///   - vocabulary: gives the identifier of a token text, or `nil` when the
    ///     vocabulary does not hold it.
    /// - Returns: one identifier for each token, or `nil` when the vocabulary
    ///   holds no identifier for a token that no merge can grow.
    static func identifiers(of byteText: String, vocabulary: (String) -> Int?) -> [Int]? {
        var parts = byteText.unicodeScalars.map { String($0) }
        while parts.count > 1 {
            guard let index = lowestPair(in: parts, vocabulary: vocabulary) else { break }
            parts.replaceSubrange(index ... index + 1, with: [parts[index] + parts[index + 1]])
        }
        var identifiers: [Int] = []
        identifiers.reserveCapacity(parts.count)
        for part in parts {
            guard let identifier = vocabulary(part) else { return nil }
            identifiers.append(identifier)
        }
        return identifiers
    }

    /// The index of the neighbouring pair whose joined text holds the lowest
    /// identifier.
    ///
    /// - Parameters:
    ///   - parts: the tokens so far.
    ///   - vocabulary: gives the identifier of a token text.
    /// - Returns: the index of the first token of the pair, or `nil` when the
    ///   vocabulary holds no pair.
    private static func lowestPair(in parts: [String], vocabulary: (String) -> Int?) -> Int? {
        var lowestIdentifier: Int?
        var lowestIndex: Int?
        for index in parts.indices.dropLast() {
            guard let identifier = vocabulary(parts[index] + parts[index + 1]) else { continue }
            if let lowestIdentifierSoFar = lowestIdentifier,
                identifier >= lowestIdentifierSoFar
            {
                continue
            }
            lowestIdentifier = identifier
            lowestIndex = index
        }
        return lowestIndex
    }
}

/// The tokenization that the published DeepSeek-V4 `tokenizer.json` states.
///
/// It reads a rendered prompt in three steps, in the order that
/// `PreTrainedTokenizer.tokenize` of `swift-transformers` uses:
///
/// 1. Each marker of ``DeepSeekV4ChatEncoder/SpecialToken`` becomes one
///    identifier. A marker stands outside the pre-tokenizer, thus a newline
///    beside a marker never joins the marker's last character.
/// 2. Each text between two markers breaks into the pieces of
///    ``DeepSeekV4PreTokenizer``, and each piece takes the byte-level spelling
///    of ``DeepSeekV4ByteLevel``.
/// 3. Each spelled piece merges through ``DeepSeekV4BytePairMerge``.
///
/// A vocabulary that holds no identifier for a marker, and a vocabulary that
/// is not byte-level, are both signs that the wrapped tokenizer is not the
/// published DeepSeek-V4 one. The three steps then stop, and the whole text
/// goes to the wrapped tokenizer in one call. Step 1 therefore never reads a
/// marker as ordinary text, and this type stays safe to wrap around any
/// tokenizer.
public struct DeepSeekV4Tokenization: Sendable {

    /// The markers of ``DeepSeekV4ChatEncoder``, longest first, as one
    /// alternation.
    ///
    /// The longest marker comes first so that a marker which starts with
    /// another marker still matches whole.
    private static let markerExpression: NSRegularExpression = {
        let escapedMarkers = DeepSeekV4ChatEncoder.SpecialToken.allMarkers
            .sorted { $0.count > $1.count }
            .map { NSRegularExpression.escapedPattern(for: $0) }
        return compiledExpression(of: escapedMarkers.joined(separator: "|"))
    }()

    /// The tokenizer that holds the vocabulary of the checkpoint.
    private let vocabulary: any Tokenizer

    /// Creates the tokenization.
    /// - Parameter vocabulary: the tokenizer loaded from the checkpoint.
    public init(vocabulary: any Tokenizer) {
        self.vocabulary = vocabulary
    }

    /// Reads a text as the identifiers the published tokenizer gives it.
    ///
    /// - Parameter text: the text to read, markers included.
    /// - Returns: the identifiers, in order.
    public func identifiers(of text: String) -> [Int] {
        guard let identifiers = byteLevelIdentifiers(of: text) else {
            // The wrapped tokenizer holds no byte-level vocabulary, or it
            // holds no identifier for a marker, thus the published path
            // cannot run. Give it the whole text, exactly as a caller that
            // knows nothing of this type would.
            return vocabulary.encode(text: text, addSpecialTokens: false)
        }
        return identifiers
    }

    /// Reads a text through the marker step, the pre-tokenizer and the merge.
    ///
    /// - Parameter text: the text to read, markers included.
    /// - Returns: the identifiers, or `nil` when the vocabulary is not
    ///   byte-level, or when it holds no identifier for a marker.
    private func byteLevelIdentifiers(of text: String) -> [Int]? {
        var allIdentifiers: [Int] = []
        var isByteLevel = true
        readParts(of: text, with: Self.markerExpression) { part, isMatch in
            guard isByteLevel else { return }
            if isMatch {
                // Each marker becomes exactly one identifier. A vocabulary
                // that holds no identifier for a marker is not the published
                // one, and reading the marker as ordinary text would break the
                // structure the model reads. Report the failure instead.
                guard let markerIdentifier = vocabulary.convertTokenToId(part) else {
                    isByteLevel = false
                    return
                }
                allIdentifiers.append(markerIdentifier)
            } else {
                guard let partIdentifiers = segmentIdentifiers(of: part) else {
                    isByteLevel = false
                    return
                }
                allIdentifiers.append(contentsOf: partIdentifiers)
            }
        }
        return isByteLevel ? allIdentifiers : nil
    }

    /// Reads one text between two markers.
    ///
    /// - Parameter segment: the text to read.
    /// - Returns: the identifiers, or `nil` when the vocabulary holds no
    ///   identifier for a piece of the text.
    private func segmentIdentifiers(of segment: String) -> [Int]? {
        var allIdentifiers: [Int] = []
        for piece in DeepSeekV4PreTokenizer.pieces(of: segment) {
            guard
                let pieceIdentifiers = DeepSeekV4BytePairMerge.identifiers(
                    of: DeepSeekV4ByteLevel.text(of: piece),
                    vocabulary: vocabulary.convertTokenToId)
            else { return nil }
            allIdentifiers.append(contentsOf: pieceIdentifiers)
        }
        return allIdentifiers
    }
}
