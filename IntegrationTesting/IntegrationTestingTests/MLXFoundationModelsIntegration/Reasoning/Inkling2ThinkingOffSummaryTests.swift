// Copyright © 2026 Apple Inc.
//
// The call of the Router compaction summarizer on Inkling 2 Mini: a blank transcript (no
// instructions, no tools), one long user prompt that asks for a summary, thinking off, and a
// ceiling of the summary size with no thinking allowance. The template reads
// `reasoning_effort: none` only as a hint, and before the answer-opening prefill the model still
// thought, spent the ceiling, and left the summary empty or cut. Each call must answer without
// reasoning, with thinking off through `.custom("no_think")` and with reasoning undeclared.
//
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2ThinkingOffSummaryTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXHuggingFace
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Summary calls of Inkling 2 Mini with thinking off, on the real checkpoint: each call must
/// answer without reasoning and with a response that is not empty.
@Suite(.serialized)
struct Inkling2ThinkingOffSummaryTests {
    /// The number of turns of the coding conversation the prompt summarizes.
    private static let turnCount = 24

    /// The number of call sites the turns of the conversation cycle through.
    private static let callSiteCycle = 3

    /// The number of passed tests the first turn of the conversation reports, less one.
    private static let passedTestsBase = 100

    /// The number of calls the test makes for each thinking-off setting.
    private static let attemptsPerSetting = 3

    /// The ceiling of the summary size, with no thinking allowance.
    private static let summaryTokenCeiling = 400

    /// The tag of the result lines in the unified log.
    private static let resultTag = "INKLING2 SUMMARY"

    /// The number of characters of the reasoning and of the response that a result line holds.
    private static let reportedPrefixLength = 300

    /// A conversation of a coding session, long enough to stand for a compaction prompt.
    private static let conversation: String = {
        let turns = (1 ... turnCount).map { index in
            """
            User: Step \(index): open Sources/App/Module\(index).swift and rename `oldName\(index)` \
            to `newName\(index)`, then run the tests.
            Assistant: I renamed `oldName\(index)` to `newName\(index)` in Module\(index).swift, \
            updated the \(index % callSiteCycle + 1) call sites, and the tests pass (\(passedTestsBase + index) passed).
            """
        }
        return turns.joined(separator: "\n")
    }()

    /// The summary request of the compaction summarizer, with the conversation after it.
    private static let prompt = """
        Summarize the conversation below in a few hundred tokens for a future session. List \
        the files changed and the current state of the tests.

        \(conversation)
        """

    /// A transcript of one user prompt, with no instructions and no tools.
    ///
    /// - Parameter text: the text of the prompt.
    /// - Returns: the transcript that holds the prompt only.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func promptTranscript(_ text: String) -> Transcript {
        Transcript(entries: [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: text))],
                    responseFormat: nil))
        ])
    }

    /// One summary call of Inkling 2 Mini, and the text it streamed.
    ///
    /// - Parameters:
    ///   - level: the reasoning level of the call.
    ///   - capabilities: the capabilities the model declares.
    ///   - maxTokens: the ceiling of the response size.
    /// - Returns: the reasoning text and the response text the call streamed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func run(
        level: ContextOptions.ReasoningLevel?, capabilities: [LanguageModelCapabilities.Capability],
        maxTokens: Int
    ) async throws -> (reasoning: String, response: String) {
        let model = MLXLanguageModel(
            configuration: inkling2Mini4Bit, capabilities: capabilities,
            weightsLocation: {
                testWeightsLocation(modelID: $0, revision: inkling2Mini4BitRevision)
            },
            load: { _, _ in try await Inkling2MiniCheckpoint.shared.container() })
        let executor = try makeMLXExecutor(for: model)
        var contextOptions = ContextOptions()
        contextOptions.reasoningLevel = level
        let request = makeExecutorRequest(
            transcript: promptTranscript(Self.prompt),
            generationOptions: GenerationOptions(maximumResponseTokens: maxTokens),
            contextOptions: contextOptions)
        let stream = try await executeResponse(executor, request: request, model: model)
        var reasoning = ""
        var response = ""
        for try await event in stream {
            if case .appendText(let chunk, _, .reasoning) = event {
                reasoning += chunk
            } else if case .appendText(let chunk, _, .response) = event {
                response += chunk
            }
        }
        return (reasoning, response)
    }

    /// Each summary call with thinking off answers without reasoning, through
    /// `.custom("no_think")` and through undeclared reasoning.
    @Test
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func summaryWithThinkingOff() async throws {
        for (label, level, capabilities) in [
            (
                "no_think, .reasoning", ContextOptions.ReasoningLevel.custom("no_think"),
                [.reasoning, .guidedGeneration, .toolCalling]
            ),
            (
                "nil level, no .reasoning", nil,
                [.guidedGeneration, .toolCalling]
            ),
        ] as [(String, ContextOptions.ReasoningLevel?, [LanguageModelCapabilities.Capability])] {
            for attempt in 1 ... Self.attemptsPerSetting {
                let result = try await run(
                    level: level, capabilities: capabilities, maxTokens: Self.summaryTokenCeiling)
                Inkling2ResultLog.report(
                    Self.resultTag,
                    "[\(label)] #\(attempt): reasoning \(result.reasoning.count) chars, "
                        + "response \(result.response.count) chars")
                let reasoningPrefix = result.reasoning.prefix(Self.reportedPrefixLength)
                let responsePrefix = result.response.prefix(Self.reportedPrefixLength)
                let reasoningSummary =
                    "\(result.reasoning.count) chars, \(reasoningPrefix.debugDescription)"
                Inkling2ResultLog.report(
                    Self.resultTag, "reasoning: \(reasoningPrefix.debugDescription)")
                Inkling2ResultLog.report(
                    Self.resultTag, "response: \(responsePrefix.debugDescription)")
                #expect(
                    result.reasoning.isEmpty,
                    "\(label) #\(attempt) reasoned with thinking off: \(reasoningSummary)")
                #expect(!result.response.isEmpty, "\(label) #\(attempt) returned an empty response")
            }
        }
    }
}

#endif
