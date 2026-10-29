// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A turn that stopped on its token budget inside its reasoning leaves an empty response entry
/// and a reasoning entry in the transcript. A model whose protocol replays reasoning gets that
/// turn back as an assistant message with an empty content and the reasoning, as
/// `ChatSession` records it, thus the history render holds the text the model wrote.
@Suite
struct TranscriptConverterReasoningOnlyTurnTests {

    /// The reasoning the turn wrote before its token budget ran out.
    private static let reasoning = "Let me add: 2 plus"

    /// The entries of a reasoning-only turn and the prompt after it, in the order the framework
    /// appends them: the response entry opens with the turn, and the reasoning entry opens
    /// when the first reasoning fragment arrives.
    ///
    /// - Parameter responseSegments: The segments of the response entry of the turn.
    /// - Returns: The entries.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func reasoningOnlyTurn(
        responseSegments: [Transcript.Segment] = []
    ) -> [Transcript.Entry] {
        [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "What is 2+2?"))],
                    responseFormat: nil)),
            .response(Transcript.Response(assetIDs: [], segments: responseSegments)),
            .reasoning(
                Transcript.Reasoning(segments: [.text(Transcript.TextSegment(content: reasoning))])),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "And 3+3?"))],
                    responseFormat: nil)),
        ]
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [[], [Transcript.Segment.text(Transcript.TextSegment(content: ""))]])
    func aReplayedReasoningOnlyTurnBecomesAnAssistantMessageWithTheReasoning(
        responseSegments: [Transcript.Segment]
    ) throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: Self.reasoningOnlyTurn(responseSegments: responseSegments), reasoning: .everyTurn)

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].content.isEmpty)
        #expect(messages[1].reasoning == Self.reasoning)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aReasoningOnlyTurnThatIsNotReplayedStaysOutOfTheHistory() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: Self.reasoningOnlyTurn(), reasoning: .dropped)

        #expect(messages.map(\.role) == [.user, .user])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anEmptyResponseWithNoReasoningStaysOutOfTheHistory() throws {
        let entries = Self.reasoningOnlyTurn().filter {
            if case .reasoning = $0 { return false }
            return true
        }

        let messages = try TranscriptConverter.mlxMessages(for: entries, reasoning: .everyTurn)

        #expect(messages.map(\.role) == [.user, .user])
    }
}

#endif
