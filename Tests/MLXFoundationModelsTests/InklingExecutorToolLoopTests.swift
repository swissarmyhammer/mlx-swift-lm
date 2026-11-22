// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// The arguments of the weather tool of the scripted Inkling 2 sessions.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArguments {
    @Guide(description: "The city, for example Rome.")
    var city: String
}

/// A tool round of the executor in the Inkling 2 protocol, on a scripted hybrid model.
///
/// The scripted model writes the frames the real checkpoint writes: a thinking frame, a tool
/// frame, then `<|content_model_end_sampling|>`. The second pass carries the tool result in
/// the transcript the way the framework appends it, and the Inkling template of
/// ``InklingFrameTokenizer`` renders the past tool call as compact JSON, which differs from the
/// call the model wrote. The model has a recurrent layer, thus its caches cannot rewind into
/// the tool call.
@Suite("An Inkling 2 tool round on the executor")
struct InklingExecutorToolLoopTests {

    /// The instructions of the session.
    private static let instructions = "You report the weather."

    /// The question of the user.
    private static let question = "What is the weather in Rome?"

    /// The scripted weather loop: the reasoning before the call, the answer after the result,
    /// and the tokenizer whose vocabulary holds every text of the conversation.
    private static let loop = InklingWeatherToolLoop(
        reasoning: "I will look Rome up.", answer: "Rome is sunny.",
        otherPieces: [instructions, question])

    /// The tokenizer of ``loop``.
    private static var tokenizer: InklingFrameTokenizer { loop.tokenizer }

    /// The identifier of the call for `city` in the transcript of the second pass.
    private static func callID(_ city: String) -> String {
        "call_\(city.lowercased())"
    }

    /// The tool the transcript declares.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var weatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: InklingWeatherToolLoop.toolName,
            description: InklingWeatherToolLoop.toolDescription,
            parameters: WeatherArguments.generationSchema)
    }

    /// A scripted hybrid Inkling 2 model under a fresh identity, whose passes write `rounds`.
    ///
    /// - Parameters:
    ///   - rounds: the tokens each pass writes, in the order of the passes.
    ///   - capabilities: the capabilities the model declares.
    ///   - weights: the directory that makes the model available.
    ///   - stopStrings: the stop strings of the model, or `nil` for none.
    /// - Returns: the model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeModel(
        rounds: [[Int]], capabilities: [LanguageModelCapabilities.Capability], weights: URL,
        stopStrings: Set<String>? = nil
    ) -> MLXLanguageModel {
        let configuration = ModelConfiguration(
            id: "probe/inkling-tool-loop-\(UUID().uuidString)", stopStrings: stopStrings,
            toolCallFormat: .inkling, reasoningConfig: InklingReasoningProtocol.inkling2)
        return MLXLanguageModel(
            configuration: configuration, capabilities: capabilities,
            weightsLocation: { _ in weights },
            load: { _, _ in
                ModelContainer(
                    context: ModelContext(
                        configuration: configuration,
                        model: ScriptedLanguageModel(
                            rounds: rounds, cacheLayerCount: 1, recurrentLayerCount: 1),
                        processor: ScalarTokenizerInputProcessor(
                            tokenizer: tokenizer, configuration: configuration),
                        tokenizer: tokenizer))
            })
    }

    /// The entries of the first pass: the instructions with the tool, then the question.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func firstEntries(sessionID: String) -> [Transcript.Entry] {
        [
            .instructions(
                Transcript.Instructions(
                    id: sessionID,
                    segments: [.text(Transcript.TextSegment(content: instructions))],
                    toolDefinitions: [weatherTool])),
            .prompt(
                Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: question))])),
        ]
    }

    /// The entries the framework appends after the calling turn: its reasoning, its call for
    /// each of `cities` and the result of each call.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func toolRoundEntries(_ cities: [String]) throws -> [Transcript.Entry] {
        let calls = try cities.map { city in
            Transcript.ToolCall(
                id: callID(city), toolName: InklingWeatherToolLoop.toolName,
                arguments: try GeneratedContent(
                    json: #"{"\#(InklingWeatherToolLoop.cityArgument)":"\#(city)"}"#))
        }
        let outputs = cities.map { city in
            Transcript.Entry.toolOutput(
                Transcript.ToolOutput(
                    id: callID(city), toolName: InklingWeatherToolLoop.toolName,
                    segments: [
                        .text(
                            Transcript.TextSegment(
                                content: InklingWeatherToolLoop.weather[city] ?? ""))
                    ]))
        }
        return [
            .reasoning(Transcript.Reasoning(segments: [.text(.init(content: loop.reasoning))])),
            .toolCalls(Transcript.ToolCalls(id: "toolcalls_weather", calls)),
        ] + outputs
    }

    /// The request of one pass over `entries`, with the weather tool enabled.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func request(
        _ entries: [Transcript.Entry], reasoningLevel: ContextOptions.ReasoningLevel?
    ) -> LanguageModelExecutorGenerationRequest {
        makeExecutorRequest(
            transcript: Transcript(entries: entries), enabledTools: [weatherTool],
            contextOptions: ContextOptions(reasoningLevel: reasoningLevel))
    }

    /// The results of the calling pass and of the pass after the tool result, and the entry the
    /// second pass checked in.
    private struct ToolRound {
        let calling: ScriptedPassResult
        let answering: ScriptedPassResult
        let entry: ExecutorPromptCacheEntry?
    }

    /// Runs the calling pass, which calls the tool for each of `cities`, and the pass after the
    /// tool results of one session.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func runToolRound(cities: [String] = ["Rome"]) async throws -> ToolRound {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeModel(
            rounds: [loop.callingTurn(cities), loop.answeringTurn],
            capabilities: [.reasoning, .toolCalling], weights: weights)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorToolLoop")
        let first = firstEntries(sessionID: "inkling-tool-round-\(UUID().uuidString)")
        let calling = try await ScriptedExecutorPass.respond(
            to: request(first, reasoningLevel: .moderate), model: model, inside: store)
        let second = request(first + (try toolRoundEntries(cities)), reasoningLevel: .moderate)
        let answering = try await ScriptedExecutorPass.respond(
            to: second, model: model, inside: store)
        let key = try #require(
            MLXLanguageModel.Executor.sessionCacheKey(
                for: second, modelID: model.modelID, kvCache: nil))
        return ToolRound(calling: calling, answering: answering, entry: await store.peek(key))
    }

    @Test("the calling pass streams the thinking frame as reasoning and the tool frame as a call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theCallingPassStreamsReasoningAndACall() async throws {
        let round = try await Self.runToolRound()

        #expect(round.calling.reasoningText == Self.loop.reasoning)
        #expect(round.calling.toolCallNames == [InklingWeatherToolLoop.toolName])
        #expect(round.calling.responseText.isEmpty)
    }

    @Test("the pass after the tool result reuses every token of the calling pass")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePassAfterTheToolResultReusesTheCallingPass() async throws {
        let round = try await Self.runToolRound()

        #expect(
            round.answering.reusedTokenCount
                == round.calling.promptTokenCount + Self.loop.callingTurn(["Rome"]).count)
        #expect(round.answering.responseText == Self.loop.answer)
    }

    @Test("the pass after the results of parallel calls reuses every token of the calling pass")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func thePassAfterParallelResultsReusesTheCallingPass() async throws {
        let cities = ["Rome", "Paris"]
        let round = try await Self.runToolRound(cities: cities)

        #expect(
            round.calling.toolCallNames == [
                InklingWeatherToolLoop.toolName, InklingWeatherToolLoop.toolName,
            ])
        #expect(
            round.answering.reusedTokenCount
                == round.calling.promptTokenCount + Self.loop.callingTurn(cities).count)
    }

    @Test("the history of the pass after the tool result renders the reasoning of the call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theHistoryRendersTheReasoningOfTheCall() async throws {
        let round = try await Self.runToolRound()

        let render = try #require(round.entry?.renderTokens)
        #expect(render.contains(Self.tokenizer.token(Self.loop.reasoning)))
    }

    @Test("each pass of a tool round reports the completion of its generation")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func eachPassReportsItsCompletion() async throws {
        let round = try await Self.runToolRound()

        #expect(round.calling.completionCount == 1)
        #expect(round.answering.completionCount == 1)
    }

    @Test("a pass that a stop string ends reports the completion and the usage of its generation")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassThatAStopStringEndsReportsItsCompletion() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let stopString = "sunny"
        let model = Self.makeModel(
            rounds: [Self.loop.answeringTurn], capabilities: [.toolCalling], weights: weights,
            stopStrings: [stopString])
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorToolLoop")
        let request = Self.request(
            Self.firstEntries(sessionID: "inkling-stop-string-\(UUID().uuidString)"),
            reasoningLevel: nil)

        let pass = try await ScriptedExecutorPass.respond(
            to: request, model: model, inside: store)

        #expect(!pass.responseText.contains(stopString))
        #expect(pass.completionCount == 1)
        #expect(pass.promptTokenCount > 0)
        // The output usage counts every token the generation produced, the tail after the stop
        // included, as the caches and the prompt-cache ledger hold them.
        #expect(pass.generatedTokenCount == pass.completedTokenCount)
        #expect(pass.generatedTokenCount >= Self.tokensThroughTheAnswer)
        #expect(pass.generatedTokenCount <= Self.loop.answeringTurn.count)
    }

    /// The tokens of the answering turn through its answer: the message and content markers,
    /// then the answer, which holds the stop string.
    private static let tokensThroughTheAnswer = 3

    @Test("a model that does not declare reasoning renders the effort none")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aModelWithoutReasoningRendersTheEffortNone() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = Self.makeModel(
            rounds: [Self.loop.answeringTurn], capabilities: [.toolCalling], weights: weights)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorToolLoop")
        let request = Self.request(
            Self.firstEntries(sessionID: "inkling-thinking-off-\(UUID().uuidString)"),
            reasoningLevel: nil)

        let pass = try await ScriptedExecutorPass.respond(
            to: request, model: model, inside: store)

        let key = try #require(
            MLXLanguageModel.Executor.sessionCacheKey(
                for: request, modelID: model.modelID, kvCache: nil))
        let render = try #require(await store.peek(key)?.renderTokens)
        let effortNone = Self.tokenizer.token(
            InklingFrameTokenizer.effortPrefix + InklingReasoningProtocol.thinkingOffEffort)
        #expect(render.contains(effortNone))
        #expect(pass.responseText == Self.loop.answer)
    }
}

#endif  // FoundationModelsIntegration
