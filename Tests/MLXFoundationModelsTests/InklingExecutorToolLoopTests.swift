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
    ///   - toolCallFormat: the tool-call format of the model.
    /// - Returns: the model.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeModel(
        rounds: [[Int]], capabilities: [LanguageModelCapabilities.Capability], weights: URL,
        toolCallFormat: ToolCallFormat = .inkling
    ) -> MLXLanguageModel {
        let configuration = ModelConfiguration(
            id: "probe/inkling-tool-loop-\(UUID().uuidString)", toolCallFormat: toolCallFormat,
            reasoningConfig: InklingReasoningProtocol.inkling2)
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

    /// The request of one greedy pass over `entries`, with the weather tool enabled in
    /// `toolCallingMode`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func request(
        _ entries: [Transcript.Entry], reasoningLevel: ContextOptions.ReasoningLevel?,
        toolCallingMode: GenerationOptions.ToolCallingMode = .allowed
    ) -> LanguageModelExecutorGenerationRequest {
        makeExecutorRequest(
            transcript: Transcript(entries: entries), enabledTools: [weatherTool],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, toolCallingMode: toolCallingMode),
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

    // MARK: - Required tool mode

    /// The arguments of the weather call for Rome, as the executor streams them.
    private static let romeArguments = #"{"\#(InklingWeatherToolLoop.cityArgument)":"Rome"}"#

    /// A required pass of a fresh session of a model that writes `rounds`.
    ///
    /// - Parameters:
    ///   - rounds: the tokens each generation of the pass writes, in order.
    ///   - capabilities: the capabilities the model declares.
    ///   - reasoningLevel: the reasoning level of the request.
    ///   - toolCallFormat: the tool-call format of the model.
    /// - Returns: what the pass streamed.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func respondRequired(
        rounds: [[Int]], capabilities: [LanguageModelCapabilities.Capability],
        reasoningLevel: ContextOptions.ReasoningLevel?, toolCallFormat: ToolCallFormat = .inkling
    ) async throws -> ScriptedPassResult {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = makeModel(
            rounds: rounds, capabilities: capabilities, weights: weights,
            toolCallFormat: toolCallFormat)
        let store = PromptCacheTestStore.withoutWrites(folderNamed: "InklingExecutorToolLoop")
        return try await ScriptedExecutorPass.respond(
            to: request(
                firstEntries(sessionID: "inkling-required-\(UUID().uuidString)"),
                reasoningLevel: reasoningLevel, toolCallingMode: .required),
            model: model, inside: store)
    }

    @Test(
        "a required pass that thinks reasons inside the thinking frame, then calls in the tool frame"
    )
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRequiredPassThinksThenCallsInTheToolFrame() async throws {
        // The turn opens the thinking frame, thus the reasoning phase writes the thinking and
        // closes the frame; the guided phase writes the tool frame.
        let thinking =
            [Self.tokenizer.token(Self.loop.reasoning)]
            + Self.loop.tokens(.endMessage)
        let call = Self.loop.callFrame("Rome") + Self.loop.tokens(.endSampling)

        let pass = try await Self.respondRequired(
            rounds: [thinking, call], capabilities: [.reasoning, .toolCalling],
            reasoningLevel: .moderate)

        #expect(pass.reasoningText == Self.loop.reasoning)
        #expect(pass.toolCallNames == [InklingWeatherToolLoop.toolName])
        #expect(pass.toolCallArguments == [Self.romeArguments])
        #expect(pass.responseText.isEmpty)
    }

    @Test("a required pass without reasoning calls in the tool frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRequiredPassWithoutReasoningCallsInTheToolFrame() async throws {
        let pass = try await Self.respondRequired(
            rounds: [Self.loop.callFrame("Rome") + Self.loop.tokens(.endSampling)],
            capabilities: [.toolCalling], reasoningLevel: nil)

        #expect(pass.reasoningText.isEmpty)
        #expect(pass.toolCallNames == [InklingWeatherToolLoop.toolName])
        #expect(pass.toolCallArguments == [Self.romeArguments])
        #expect(pass.responseText.isEmpty)
    }

    @Test(
        "a required pass of a frame protocol without a required frame throws before it generates",
        arguments: [ToolCallFormat.gptOSS, .atem])
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRequiredPassOfAnUnsupportedFrameProtocolThrows(format: ToolCallFormat) async throws {
        do {
            let pass = try await Self.respondRequired(
                rounds: [Self.loop.callFrame("Rome")], capabilities: [.toolCalling],
                reasoningLevel: nil, toolCallFormat: format)
            Issue.record("expected a refusal, the pass called \(pass.toolCallNames)")
        } catch let error as LanguageModelError {
            guard case .unsupportedCapability(let unsupported) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(unsupported.capability == .toolCalling)
            #expect(unsupported.debugDescription.contains(format.rawValue))
        }
    }
}

#endif  // FoundationModelsIntegration
