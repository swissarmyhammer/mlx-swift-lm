// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A protocol that replays only a turn that stopped inside its reasoning
/// (``ReasoningConfig/replaysUnclosedReasoningIntoHistory``, DeepSeek-V4) keeps the reasoning of
/// a turn whose response is empty, as `ChatSession` records it, and drops the reasoning of every
/// other turn. The history render of the turn then holds the text the model wrote inside the
/// block its prompt opened, thus the render of the next turn extends the tokens the prompt cache
/// holds.
@Suite
struct TranscriptConverterUnclosedReasoningTests {

    /// The model the configurations name.
    private static let modelID = "test/unclosed-reasoning"

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatStoppedInsideItsReasoningKeepsTheReasoning() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.reasoningOnlyTurn(responseSegments: []),
            reasoning: .reasoningOnlyTurns)

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].content.isEmpty)
        #expect(messages[1].reasoning == ReasoningOnlyTurnFixture.reasoning)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatAnsweredDropsItsReasoning() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.reasoningOnlyTurn(
                responseSegments: [ReasoningOnlyTurnFixture.text("4")]),
            reasoning: .reasoningOnlyTurns)

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].content == "4")
        #expect(messages[1].reasoning == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatCalledAToolDropsItsReasoning() throws {
        let call = Transcript.ToolCall(
            id: "call_1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location":"Paris"}"#))
        let entries: [Transcript.Entry] = [
            ReasoningOnlyTurnFixture.prompt("Weather in Paris?"),
            ReasoningOnlyTurnFixture.reasoningEntry,
            .toolCalls(Transcript.ToolCalls(id: "tc_1", [call])),
        ]

        let messages = try TranscriptConverter.mlxMessages(
            for: entries, reasoning: .reasoningOnlyTurns)

        try #require(messages.map(\.role) == [.user, .assistant])
        #expect(messages[1].tool?.calls?.map(\.function.name) == ["get_weather"])
        #expect(messages[1].reasoning == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anEmptyResponseWithNoReasoningStaysOutOfTheHistory() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: ReasoningOnlyTurnFixture.emptyTurnWithoutReasoning, reasoning: .reasoningOnlyTurns)

        #expect(messages.map(\.role) == [.user, .user])
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theReasoningProtocolNamesHowTheHistoryKeepsReasoning() {
        #expect(
            TranscriptConverter.HistoryReasoning(for: DeepSeekV4EncodingTokenizer.reasoningConfig)
                == .reasoningOnlyTurns)
        #expect(
            TranscriptConverter.HistoryReasoning(for: QwenReasoningProtocol.qwen35) == .everyTurn)
        #expect(
            TranscriptConverter.HistoryReasoning(for: ReasoningConfig.alwaysOnThinking) == .dropped)
        #expect(TranscriptConverter.HistoryReasoning(for: nil) == .dropped)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theHistoryOfAnUnclosedReasoningProtocolKeepsTheReasoningOnlyTurn() throws {
        let resolved = ModelConfiguration(
            id: Self.modelID, reasoningConfig: DeepSeekV4EncodingTokenizer.reasoningConfig)
        let transcript = Transcript(
            entries: ReasoningOnlyTurnFixture.reasoningOnlyTurn(responseSegments: []))

        let messages = try MLXLanguageModel.Executor.historyMessages(
            of: transcript, resolved: resolved,
            otherwise: TranscriptConverter.mlxMessages(for: transcript))

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].reasoning == ReasoningOnlyTurnFixture.reasoning)
    }
}

#endif
