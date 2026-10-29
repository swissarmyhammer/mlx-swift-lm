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

    /// The reasoning the turns write.
    private static let reasoning = "Let me add: 2 plus"

    /// The model the configurations name.
    private static let modelID = "test/unclosed-reasoning"

    /// A prompt entry that asks `question`.
    ///
    /// - Parameter question: The text of the prompt.
    /// - Returns: The entry.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(_ question: String) -> Transcript.Entry {
        .prompt(
            Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: question))], responseFormat: nil))
    }

    /// The reasoning entry of a turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var reasoningEntry: Transcript.Entry {
        .reasoning(Transcript.Reasoning(segments: [.text(Transcript.TextSegment(content: reasoning))]))
    }

    /// The entries of one turn whose response holds `answer`, and the prompt after it, in the
    /// order the framework appends them: the response entry opens with the turn, and the
    /// reasoning entry opens when the first reasoning fragment arrives.
    ///
    /// - Parameter answer: The text of the response, or `nil` for a turn that stopped inside
    ///   its reasoning.
    /// - Returns: The entries.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func turn(answer: String?) -> [Transcript.Entry] {
        let segments = answer.map { [Transcript.Segment.text(Transcript.TextSegment(content: $0))] }
        return [
            prompt("What is 2+2?"),
            .response(Transcript.Response(assetIDs: [], segments: segments ?? [])),
            reasoningEntry,
            prompt("And 3+3?"),
        ]
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatStoppedInsideItsReasoningKeepsTheReasoning() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: Self.turn(answer: nil), reasoning: .reasoningOnlyTurns)

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].content.isEmpty)
        #expect(messages[1].reasoning == Self.reasoning)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatAnsweredDropsItsReasoning() throws {
        let messages = try TranscriptConverter.mlxMessages(
            for: Self.turn(answer: "4"), reasoning: .reasoningOnlyTurns)

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
            Self.prompt("Weather in Paris?"),
            Self.reasoningEntry,
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
        let entries = Self.turn(answer: nil).filter {
            if case .reasoning = $0 { return false }
            return true
        }

        let messages = try TranscriptConverter.mlxMessages(
            for: entries, reasoning: .reasoningOnlyTurns)

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
        let transcript = Transcript(entries: Self.turn(answer: nil))

        let messages = try MLXLanguageModel.Executor.historyMessages(
            of: transcript, resolved: resolved,
            otherwise: TranscriptConverter.mlxMessages(for: transcript))

        try #require(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages[1].reasoning == Self.reasoning)
    }
}

#endif
