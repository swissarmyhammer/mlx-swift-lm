// Copyright © 2026 Apple Inc.
//
// Integration tests for the published DeepSeek-V4 tokenizer, from card
// ^zg3d8wv. The tests download the tokenizer of
// `deepseek-ai/DeepSeek-V4-Flash` and prove two facts a unit test cannot:
//
// 1. Each prompt marker is one token, and `NaiveStreamingDetokenizer` never
//    splits it. The turn markers are not `special`, thus a skip-specials
//    decode keeps them.
// 2. `Tokenizer.eosTokenId` is 1, and id 1 decodes to the end-of-sentence
//    marker.
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

/// One shared download and load of the DeepSeek-V4 tokenizer.
private enum DeepSeekV4TokenizerLoad {
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
