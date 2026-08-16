// Copyright © 2026 Apple Inc.
//
// Integration tests for the published DeepSeek-V4 tokenizer, from card
// ^zg3d8wv. The tests download the tokenizer of
// `deepseek-ai/DeepSeek-V4-Flash` and prove three facts a unit test cannot:
//
// 1. Each prompt marker is one token, and `NaiveStreamingDetokenizer` never
//    splits it. The turn markers are not `special`, thus a skip-specials
//    decode keeps them.
// 2. `Tokenizer.eosTokenId` is 1, and id 1 decodes to the end-of-sentence
//    marker.
// 3. A rendered tool prompt tokenizes to the identifiers the published
//    tokenizer gives the same text, one for one. Only the identifiers reach
//    the model, thus correct text is not enough.
//
// The marker strings come from `DeepSeekV4ChatEncoder.SpecialToken`. The
// downloaded tokenizer is the independent oracle, thus these tests also prove
// the encoder constants against the published `tokenizer.json`.
//
// Note on the eos source: `tokenizer_config.json` holds no `eos_token_id`
// key. It holds `eos_token`, a string. `Tokenizer.eosTokenId` looks that
// string up in the vocabulary, where `tokenizer.json` gives it the id 1.
// Thus the test asserts through `eosTokenId` and does not read a
// `eos_token_id` key from `tokenizer_config.json`.

import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// One row of the `added_tokens` array of the published `tokenizer.json`.
private struct MarkerFixture {
    /// The token id of the marker.
    let id: Int
    /// The full marker text.
    let text: String
    /// The value of the `special` flag in `tokenizer.json`.
    let isSpecial: Bool
}

/// The marker that opens a conversation.
private let beginOfSentenceMarker = MarkerFixture(
    id: 0, text: DeepSeekV4ChatEncoder.SpecialToken.beginOfSentence, isSpecial: true)
/// The marker that closes an assistant turn.
private let endOfSentenceMarker = MarkerFixture(
    id: 1, text: DeepSeekV4ChatEncoder.SpecialToken.endOfSentence, isSpecial: true)
/// The marker that opens a user turn.
private let userMarker = MarkerFixture(
    id: 128803, text: DeepSeekV4ChatEncoder.SpecialToken.user, isSpecial: false)
/// The marker that opens an assistant turn.
private let assistantMarker = MarkerFixture(
    id: 128804, text: DeepSeekV4ChatEncoder.SpecialToken.assistant, isSpecial: false)
/// The marker that opens a reasoning block.
private let thinkStartMarker = MarkerFixture(
    id: 128821, text: DeepSeekV4ChatEncoder.SpecialToken.thinkStart, isSpecial: false)
/// The marker that closes a reasoning block.
private let thinkEndMarker = MarkerFixture(
    id: 128822, text: DeepSeekV4ChatEncoder.SpecialToken.thinkEnd, isSpecial: false)
/// The marker that opens and closes every DSML tag.
private let dsmlMarker = MarkerFixture(
    id: 128825, text: DeepSeekV4ChatEncoder.SpecialToken.dsml, isSpecial: false)
/// The marker that opens a reminder turn.
private let latestReminderMarker = MarkerFixture(
    id: 128828, text: DeepSeekV4ChatEncoder.SpecialToken.latestReminder, isSpecial: false)

/// All the marker rows under test, in id order.
private let deepSeekV4Markers = [
    beginOfSentenceMarker, endOfSentenceMarker, userMarker, assistantMarker,
    thinkStartMarker, thinkEndMarker, dsmlMarker, latestReminderMarker,
]

// MARK: - The published token identifiers of one tool prompt

/// The token identifiers the published tokenizer gives one rendered tool
/// prompt.
///
/// The fixture comes from the published `tokenizer.json` of
/// `deepseek-ai/DeepSeek-V4-Flash` @ 60d8d70770c6776ff598c94bb586a859a38244f1,
/// over the prompt that `encoding/encoding_dsv4.py` renders for the same
/// conversation:
///
/// ```python
/// import json
/// from tokenizers import Tokenizer
/// from encoding_dsv4 import encode_messages
/// tokenizer = Tokenizer.from_file("tokenizer.json")
/// prompt = encode_messages(messages, thinking_mode="chat")
/// json.dump({"prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False).ids},
///           open("deepseek-v4-flash-tool-prompt-tokens.json", "w"))
/// ```
private struct ToolPromptTokenFixture: Decodable {
    /// The token identifiers the published tokenizer gives the prompt.
    let promptTokenIDs: [Int]

    private enum CodingKeys: String, CodingKey {
        case promptTokenIDs = "prompt_token_ids"
    }
}

/// The on-disk location of the tool-prompt token fixture, next to this source
/// file.
private var toolPromptTokenFixtureURL: URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures", isDirectory: true)
        .appendingPathComponent("deepseek-v4-flash-tool-prompt-tokens.json")
}

/// The conversation of the fixture, in the raw dictionary form
/// `applyChatTemplate` reads. The two turns come from
/// `DeepseekV4IntegrationTests`, thus the fixture and the real-weights tool
/// test render the same conversation.
private let toolPromptMessages: [[String: any Sendable]] = [
    ["role": "system", "content": stockAgentInstructions],
    ["role": "user", "content": stockToolUserPrompt],
]

/// The number of identifiers a failure message prints on each side of a
/// difference.
private let tokenReportWindow = 6

/// The index of the first identifier the two sequences disagree on.
///
/// A sequence that is a prefix of the other disagrees at the end of the
/// shorter one.
///
/// - Parameters:
///   - left: the identifiers under test.
///   - right: the identifiers of the reference.
/// - Returns: the index, or `nil` when the two sequences are equal.
private func firstDifferingIndex(_ left: [Int], _ right: [Int]) -> Int? {
    let shared = min(left.count, right.count)
    for index in 0 ..< shared where left[index] != right[index] {
        return index
    }
    return left.count == right.count ? nil : shared
}

/// The identifiers around one index, each beside its token text, for a
/// failure message.
///
/// The token text is the byte-level spelling of the vocabulary, thus a space
/// reads `Ġ` and a newline reads `Ċ`. That spelling is what tells one
/// whitespace token from another.
///
/// - Parameters:
///   - identifiers: the sequence to read.
///   - index: the index to centre the window on.
///   - tokenizer: the tokenizer that names each identifier.
/// - Returns: the window, or `[]` when the index is past the end.
private func identifierWindow(
    _ identifiers: [Int], around index: Int, through tokenizer: any MLXLMCommon.Tokenizer
) -> String {
    let lower = max(0, index - tokenReportWindow)
    let upper = min(identifiers.count, index + tokenReportWindow)
    guard lower < upper else { return "[]" }
    let pieces = identifiers[lower ..< upper].map { identifier in
        "\(identifier) \(tokenizer.convertIdToToken(identifier).map { "\"\($0)\"" } ?? "unknown")"
    }
    return "[" + pieces.joined(separator: ", ") + "]"
}

/// One shared download and load of the DeepSeek-V4 tokenizer.
enum DeepSeekV4TokenizerLoad {
    /// The repository that publishes the DeepSeek-V4 tokenizer.
    static let repositoryID = "deepseek-ai/DeepSeek-V4-Flash"
    /// The pinned revision. It is the revision the encoder port names in
    /// `Libraries/MLXLMCommon/DeepSeekV4ChatEncoder.swift`.
    static let revision = "60d8d70770c6776ff598c94bb586a859a38244f1"
    /// The file patterns that get the tokenizer files and not the weights.
    /// This is the same set as `tokenizerDownloadPatterns` in
    /// `Libraries/MLXLMCommon/ModelFactory.swift`, which is `package` and
    /// thus not visible here.
    static let filePatterns = ["*.json", "*.jinja"]

    /// The one shared load task. Each test awaits the same download.
    static let shared: Task<any MLXLMCommon.Tokenizer, any Error> = Task {
        let directory = try await #hubDownloader().download(
            id: repositoryID,
            revision: revision,
            matching: filePatterns,
            useLatest: false,
            progressHandler: { _ in }
        )
        return try await #huggingFaceTokenizerLoader().load(from: directory)
    }
}

/// Real-tokenizer tests for the DeepSeek-V4 markers and the eos token id.
/// The first run downloads the tokenizer files from Hugging Face Hub.
///
/// Run explicitly via:
/// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepSeekV4TokenizerIntegrationTests`
@Suite(.serialized, .timeLimit(.minutes(60)))
struct DeepSeekV4TokenizerIntegrationTests {

    /// A DSML tool call streams back whole, newlines included.
    ///
    /// The real-weights run of card ^2dvj1g6 read the closing tag of the
    /// invoke element as `</｜DSML｜inv>`, where the syntax states
    /// `</｜DSML｜invoke>`. The published vocabulary holds no `invoke` token:
    /// it writes the word as `inv` (40148) and `oke` (5406). Exactly one token
    /// was thus absent from the text the tool-call parser read.
    ///
    /// This test streams the answer that a tool round needs, one token at a
    /// time, through the production detokenizer, and asks for the text back
    /// unchanged. `markersSurviveStreamingDetokenizationWhole` cannot see this
    /// defect: its stream holds no newline, and
    /// `NaiveStreamingDetokenizer` starts a new segment at each newline. The
    /// DSML answer holds four newlines, and the damaged tag follows one.
    @Test func aToolCallStreamsBackWholeAcrossItsNewlines() async throws {
        let tokenizer = try await DeepSeekV4TokenizerLoad.shared.value
        let marker = DeepSeekV4ChatEncoder.SpecialToken.dsml
        let answer = """
            <\(marker)tool_calls>
            <\(marker)invoke name="get_stock_level">
            <\(marker)parameter name="bay" string="true">bay 7</\(marker)parameter>
            </\(marker)invoke>
            </\(marker)tool_calls>
            """

        let identifiers = DeepSeekV4Tokenization(vocabulary: tokenizer).identifiers(of: answer)

        // The encode side first: the identifiers must carry the whole answer.
        // A failure here is a tokenization defect, not a streaming defect.
        #expect(
            tokenizer.decode(tokenIds: identifiers) == answer,
            "the identifiers of the answer must decode back to the answer")

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var pieces: [String] = []
        for identifier in identifiers {
            detokenizer.append(token: identifier)
            guard let piece = detokenizer.next() else { continue }
            pieces.append(piece)
        }

        #expect(
            pieces.joined() == answer,
            """
            the streamed text must equal the answer. The tool-call parser reads \
            this text, thus a token the stream drops is a tool round that never \
            completes.
            """)
    }

    /// Each marker is one token that the streaming detokenizer emits whole,
    /// a mixed stream keeps the markers complete and the text in order, and
    /// a skip-specials decode keeps the turn markers.
    @Test func markersSurviveStreamingDetokenizationWhole() async throws {
        let tokenizer = try await DeepSeekV4TokenizerLoad.shared.value

        // Each marker id decodes to its complete marker, with no split.
        for marker in deepSeekV4Markers {
            var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            detokenizer.append(token: marker.id)
            #expect(
                detokenizer.next() == marker.text,
                "id \(marker.id) must stream as one whole marker")
        }

        // A stream that mixes markers and ordinary text. The text ids come
        // from the tokenizer itself, thus the test makes no assumption about
        // the text vocabulary.
        let userTextIDs = tokenizer.encode(
            text: "Hello, how are you?", addSpecialTokens: false)
        let replyTextIDs = tokenizer.encode(
            text: "I am fine.", addSpecialTokens: false)
        let stream =
            [beginOfSentenceMarker.id, userMarker.id] + userTextIDs
            + [assistantMarker.id, thinkStartMarker.id] + replyTextIDs
            + [thinkEndMarker.id, endOfSentenceMarker.id]

        var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var pieces: [String] = []
        for token in stream {
            detokenizer.append(token: token)
            guard let piece = detokenizer.next() else { continue }
            pieces.append(piece)
            if let marker = deepSeekV4Markers.first(where: { $0.id == token }) {
                // The piece for a marker id is the marker, whole and alone.
                #expect(
                    piece == marker.text,
                    "id \(marker.id) must stream as one whole marker in a mixed stream")
            }
        }

        // The streamed text holds the markers complete and the text in order.
        let streamed = pieces.joined()
        let expected =
            beginOfSentenceMarker.text + userMarker.text
            + tokenizer.decode(tokenIds: userTextIDs)
            + assistantMarker.text + thinkStartMarker.text
            + tokenizer.decode(tokenIds: replyTextIDs)
            + thinkEndMarker.text + endOfSentenceMarker.text
        #expect(streamed == expected)

        // A skip-specials decode removes only the two sentence markers,
        // because only they carry `special: true`. The turn markers stay.
        let skipSpecials = tokenizer.decode(tokenIds: stream, skipSpecialTokens: true)
        for marker in deepSeekV4Markers where stream.contains(marker.id) {
            #expect(
                skipSpecials.contains(marker.text) == !marker.isSpecial,
                "skip-specials must keep id \(marker.id) exactly when it is not special")
        }
    }

    /// A rendered DeepSeek-V4 prompt tokenizes with each marker as its own
    /// token.
    ///
    /// The render is correct text only when the tokenizer maps each marker
    /// back to the one token the model saw in training. A marker that splits
    /// into ordinary word pieces gives a prompt that reads correctly and
    /// tokenizes wrongly, which no text comparison can find.
    @Test func aRenderedPromptTokenizesEachMarkerAsOneToken() async throws {
        let tokenizer = try await DeepSeekV4TokenizerLoad.shared.value
        let prompt = DeepSeekV4ChatEncoder().encode(
            messages: [
                .system(content: "S"),
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_1", name: "alpha", argumentsJSON: #"{"n": "1"}"#)
                    ]),
                .toolResult(content: "done", toolCallID: "call_1"),
            ],
            thinkingMode: .thinking)
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)

        for marker in deepSeekV4Markers where prompt.contains(marker.text) {
            #expect(
                tokens.contains(marker.id),
                "the render writes \(marker.text), thus the prompt must carry id \(marker.id)")
        }
    }

    /// The tool prompt tokenizes to the identifiers the published tokenizer
    /// gives it, one for one.
    ///
    /// ``aRenderedPromptTokenizesEachMarkerAsOneToken()`` asks only that each
    /// marker id is PRESENT, and a byte comparison of the rendered text asks
    /// only that the text is right. Neither finds a prompt whose text is
    /// correct and whose identifiers are not, and only the identifiers reach
    /// the model. The `## Tools` section is where that matters most: it is
    /// the one part of the prompt that states the tool schema and the DSML
    /// rule, thus a tool round rides on it.
    @Test func theToolPromptTokenizesToThePublishedIdentifiers() async throws {
        let fixture = try JSONDecoder().decode(
            ToolPromptTokenFixture.self, from: Data(contentsOf: toolPromptTokenFixtureURL))
        let base = try await DeepSeekV4TokenizerLoad.shared.value
        let tokens = try DeepSeekV4EncodingTokenizer(wrapping: base).applyChatTemplate(
            messages: toolPromptMessages,
            tools: [stockToolSpec],
            additionalContext: ["thinking": false])

        let expected = fixture.promptTokenIDs
        let difference = firstDifferingIndex(tokens, expected)
        let report =
            difference.map { index in
                "index \(index): Swift "
                    + identifierWindow(tokens, around: index, through: base)
                    + " against reference "
                    + identifierWindow(expected, around: index, through: base)
            } ?? "none"
        #expect(
            difference == nil,
            """
            the rendered tool prompt must tokenize to the identifiers the published \
            tokenizer gives it. Swift wrote \(tokens.count) identifiers and the \
            reference holds \(expected.count). The first difference is at \(report).
            """)
    }

    /// `Tokenizer.eosTokenId` is 1, and id 1 decodes to the end-of-sentence
    /// marker.
    @Test func endOfSentenceTokenIdIsOne() async throws {
        let tokenizer = try await DeepSeekV4TokenizerLoad.shared.value
        #expect(tokenizer.eosTokenId == endOfSentenceMarker.id)
        #expect(
            tokenizer.decode(tokenIds: [endOfSentenceMarker.id])
                == endOfSentenceMarker.text)
    }
}
