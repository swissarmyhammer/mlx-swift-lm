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

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: [[], [ReasoningOnlyTurnFixture.text("")]])
    func aReplayedReasoningOnlyTurnBecomesAnAssistantMessageWithTheReasoning(
        responseSegments: [Transcript.Segment]
    ) throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.reasoningOnlyTurn(responseSegments: responseSegments),
            reasoning: .everyTurn)

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].content.isEmpty)
        #expect(messages[1].reasoning == ReasoningOnlyTurnFixture.reasoning)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aReasoningOnlyTurnThatIsNotReplayedStaysOutOfTheHistory() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.reasoningOnlyTurn(), reasoning: .dropped)

        #expect(messages.map(\.role) == [.user, .user])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anEmptyResponseWithNoReasoningStaysOutOfTheHistory() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.emptyTurnWithoutReasoning, reasoning: .everyTurn)

        #expect(messages.map(\.role) == [.user, .user])
    }
}

#endif
