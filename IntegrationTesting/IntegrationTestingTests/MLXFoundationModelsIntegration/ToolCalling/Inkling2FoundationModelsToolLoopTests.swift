// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of a two-step tool loop of Inkling 2 Mini through `MLXLanguageModel`,
// the path the Router and ACPAgent sessions take: the model looks up a customer id with one
// tool, then the balance of that id with a second tool, then answers from the result.
//
// One test runs the loop in a `LanguageModelSession` with two real `Tool`s, as an app does.
// The other drives `MLXLanguageModel.Executor` round by round, the way the framework does
// inside that session, and reads the usage of each round from the channel: each round after a
// tool result, and the user turn after the loop, must serve part of its prompt from the
// prompt cache of the session. Both run with thinking on and with thinking off.
//
// `Inkling2ToolLoopIntegrationTests` runs the same loop through `ChatSession`.
//
// Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2FoundationModelsToolLoopTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import HuggingFace
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// The log every result line of this suite goes to.
private let resultLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "Inkling2FoundationModelsToolLoop")

// MARK: - The tools

/// The argument of the customer tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct FindCustomerArguments {
    @Guide(description: "The full name of the customer")
    var name: String
}

/// The argument of the balance tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct GetBalanceArguments {
    @Guide(description: "The customer id, for example C-1000")
    var customerId: String
}

/// The customer tool of a `LanguageModelSession`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct FindCustomerTool: FoundationModels.Tool {
    let name = Inkling2BankFixture.findCustomerTool
    let description = Inkling2BankFixture.findCustomerDescription
    let log: Inkling2ToolCallLog

    func call(arguments: FindCustomerArguments) async throws -> String {
        log.dispatch(name, argument: arguments.name)
    }
}

/// The balance tool of a `LanguageModelSession`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct GetBalanceTool: FoundationModels.Tool {
    let name = Inkling2BankFixture.getBalanceTool
    let description = Inkling2BankFixture.getBalanceDescription
    let log: Inkling2ToolCallLog

    func call(arguments: GetBalanceArguments) async throws -> String {
        log.dispatch(name, argument: arguments.customerId)
    }
}

// MARK: - One executor round

/// What one executor round streamed.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct Inkling2Round {
    var reasoning = ""
    var text = ""
    var toolCalls: [(id: String, name: String, arguments: String)] = []
    var renderedTokenCount = 0
    var cachedTokenCount = 0
    var completion: GenerateCompletionInfo?

    /// Reads one executor event into the round.
    mutating func consume(_ event: MLXLanguageModel.Executor.GenerationEvent) {
        switch event {
        case .appendText(let chunk, _, .reasoning): reasoning += chunk
        case .appendText(let chunk, _, .response): text += chunk
        case .toolCall(let id, let name, let arguments): toolCalls.append((id, name, arguments))
        case .updateUsage(let input, _, _):
            renderedTokenCount = input.totalTokenCount
            cachedTokenCount = input.cachedTokenCount
        case .completion(let info): completion = info
        case .updateMetadata: break
        }
    }

    /// One line that names the calls, the cached prompt tokens and the decode rate.
    var summary: String {
        let calls = toolCalls.map { "\($0.name)(\($0.arguments))" }
        let decode = completion.map { String(format: "%.1f", $0.tokensPerSecond) } ?? "?"
        return
            "calls \(calls), prompt \(renderedTokenCount) tok (cached \(cachedTokenCount)), "
            + "generated \(completion?.generationTokenCount ?? 0) tok, decode \(decode) tok/s, "
            + "reasoning \(reasoning.count) chars, text <<<\(text)>>>"
    }
}

// MARK: - Tests

@Suite(.serialized, .timeLimit(.minutes(30)))
struct Inkling2FoundationModelsToolLoopTests {

    /// Tokens each round may generate.
    private static let maxTokens = 1_024

    /// The most rounds the executor loop runs before it gives up on an answer.
    private static let maxRounds = 6

    /// The thinking states under test: the default effort of the template, and thinking off.
    static let thinkingStates = [true, false]

    /// Greedy options with the round budget.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var options: GenerationOptions {
        GenerationOptions(samplingMode: .greedy, temperature: 0, maximumResponseTokens: maxTokens)
    }

    /// Writes one result line, which the unified log of the run can be searched for.
    private static func report(_ line: String) {
        resultLog.info("[INKLING2 FM TOOLS] \(line, privacy: .public)")
    }

    /// The snapshot directory of the pinned checkpoint in the Hugging Face cache.
    private static func snapshotDirectory(modelID: String) -> URL {
        let cache = HuggingFace.HubCache.default
        guard let repo = HuggingFace.Repo.ID(rawValue: modelID),
            let snapshot = try? cache.snapshotPath(
                repo: repo, kind: .model, commitHash: inkling2Mini4BitRevision)
        else {
            return cache.cacheDirectory
        }
        return snapshot
    }

    /// The model of the pinned checkpoint. Thinking off is a model that does not declare
    /// `.reasoning`, which the executor renders with the thinking effort `none`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeModel(thinking: Bool) -> MLXLanguageModel {
        MLXLanguageModel(
            configuration: inkling2Mini4Bit,
            capabilities: thinking ? [.reasoning, .toolCalling] : [.toolCalling],
            weightsLocation: { snapshotDirectory(modelID: $0) },
            load: { _, _ in try await Inkling2MiniCheckpoint.shared.container() })
    }

    /// The tool definitions of the executor rounds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var toolDefinitions: [Transcript.ToolDefinition] {
        [
            Transcript.ToolDefinition(
                name: Inkling2BankFixture.findCustomerTool,
                description: Inkling2BankFixture.findCustomerDescription,
                parameters: FindCustomerArguments.generationSchema),
            Transcript.ToolDefinition(
                name: Inkling2BankFixture.getBalanceTool,
                description: Inkling2BankFixture.getBalanceDescription,
                parameters: GetBalanceArguments.generationSchema),
        ]
    }

    /// A text segment.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func segment(_ content: String) -> Transcript.Segment {
        .text(Transcript.TextSegment(content: content))
    }

    /// A prompt entry.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func prompt(_ content: String) -> Transcript.Entry {
        .prompt(Transcript.Prompt(segments: [segment(content)]))
    }

    /// The one string argument of the JSON arguments of a call, or an empty text.
    private static func argument(ofJSON arguments: String) -> String {
        let object = try? JSONSerialization.jsonObject(with: Data(arguments.utf8))
        return (object as? [String: Any])?.values.first as? String ?? ""
    }

    /// The entries the framework appends after a round that called tools: its reasoning, its
    /// calls, and the result of each call.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func entries(
        after round: Inkling2Round, number: Int, log: Inkling2ToolCallLog
    ) throws -> [Transcript.Entry] {
        var added: [Transcript.Entry] = []
        if !round.reasoning.isEmpty {
            added.append(.reasoning(Transcript.Reasoning(segments: [segment(round.reasoning)])))
        }
        let calls = try round.toolCalls.map {
            Transcript.ToolCall(
                id: $0.id, toolName: $0.name, arguments: try GeneratedContent(json: $0.arguments))
        }
        added.append(.toolCalls(Transcript.ToolCalls(id: "toolcalls_\(number)", calls)))
        for call in round.toolCalls {
            let result = log.dispatch(call.name, argument: argument(ofJSON: call.arguments))
            added.append(
                .toolOutput(
                    Transcript.ToolOutput(
                        id: call.id, toolName: call.name, segments: [segment(result)])))
        }
        return added
    }

    /// Runs one executor round over `entries`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func runRound(
        _ entries: [Transcript.Entry], thinking: Bool, model: MLXLanguageModel,
        executor: MLXLanguageModel.Executor
    ) async throws -> Inkling2Round {
        let request = makeExecutorRequest(
            transcript: Transcript(entries: entries), enabledTools: toolDefinitions,
            generationOptions: options,
            contextOptions: ContextOptions(reasoningLevel: thinking ? .moderate : nil))
        var round = Inkling2Round()
        // The stream cancels its producer when it deinitializes, thus it must outlive the
        // iteration.
        let stream = try await executeResponse(executor, request: request, model: model)
        for try await event in stream {
            round.consume(event)
        }
        withExtendedLifetime(stream) {}
        return round
    }

    /// The model calls the customer tool, then the balance tool with the id the first result
    /// gave, then answers with the balance, inside a `LanguageModelSession` with two real
    /// tools. The answer of the loop follows a tool result, thus its usage reports the cached
    /// prompt tokens of the round after a tool result.
    @Test(arguments: thinkingStates)
    func languageModelSessionCompletesATwoStepToolLoop(thinking: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASessionCompletesTheLoop(thinking: thinking)
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// Runs the loop of ``languageModelSessionCompletesATwoStepToolLoop(thinking:)``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASessionCompletesTheLoop(thinking: Bool) async throws {
        let label = thinking ? "thinking" : "no-thinking"
        let log = Inkling2ToolCallLog()
        let session = LanguageModelSession(
            model: Self.makeModel(thinking: thinking),
            tools: [FindCustomerTool(log: log), GetBalanceTool(log: log)],
            instructions: Inkling2BankFixture.instructions)

        let answer = try await session.respond(
            to: Inkling2BankFixture.question, options: Self.options)
        let followUp = try await session.respond(
            to: Inkling2BankFixture.followUp, options: Self.options)

        for entry in session.transcript {
            Self.report("session[\(label)] entry: \(String(describing: entry).prefix(400))")
        }
        Self.report(
            "session[\(label)] answer cached \(answer.usage.input.cachedTokenCount) of "
                + "\(answer.usage.input.totalTokenCount) <<<\(answer.content)>>>")
        Self.report(
            "session[\(label)] follow-up cached \(followUp.usage.input.cachedTokenCount) of "
                + "\(followUp.usage.input.totalTokenCount) <<<\(followUp.content)>>>")
        #expect(log.isTheTwoStepLoop, "[\(label)] calls \(log.calls)")
        #expect(
            Inkling2BankFixture.namesTheBalance(answer.content), "[\(label)] \(answer.content)")
        #expect(answer.usage.input.cachedTokenCount > 0, "[\(label)] answer reused no token")
        #expect(followUp.usage.input.cachedTokenCount > 0, "[\(label)] follow-up reused no token")
        #expect(
            followUp.content.contains(Inkling2BankFixture.customerID),
            "[\(label)] follow-up \(followUp.content)")
    }

    /// The same loop, one executor round at a time, at the default thinking effort and with
    /// thinking off: each round after a tool result, and the user turn after the loop, reuses
    /// the prompt cache of the session. The thinking frames stream as reasoning, and thinking
    /// off (the effort `none`) makes the model reason less.
    @Test
    func executorReusesThePromptCacheAfterEachToolResult() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let thinkingReasoning = try await expectTheExecutorReusesThePromptCache(
                thinking: true)
            let offReasoning = try await expectTheExecutorReusesThePromptCache(thinking: false)
            #expect(!thinkingReasoning.isEmpty, "thinking on streamed no reasoning")
            #expect(
                offReasoning.count < thinkingReasoning.count,
                "thinking off reasoned \(offReasoning.count) chars, on \(thinkingReasoning.count)"
            )
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// Runs the rounds and the follow-up turn of one session, and records an issue unless the
    /// calls, the answers and the cache reuse are right.
    ///
    /// - Parameter thinking: whether the model declares reasoning.
    /// - Returns: the reasoning the rounds streamed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectTheExecutorReusesThePromptCache(thinking: Bool) async throws -> String {
        let label = thinking ? "thinking" : "no-thinking"
        let model = Self.makeModel(thinking: thinking)
        let executor = try makeMLXExecutor(for: model)
        let log = Inkling2ToolCallLog()
        var entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    segments: [Self.segment(Inkling2BankFixture.instructions)],
                    toolDefinitions: Self.toolDefinitions)),
            Self.prompt(Inkling2BankFixture.question),
        ]

        var rounds: [Inkling2Round] = []
        for number in 0 ..< Self.maxRounds {
            let round = try await Self.runRound(
                entries, thinking: thinking, model: model, executor: executor)
            Self.report("executor[\(label)] round \(number): \(round.summary)")
            Self.report("executor[\(label)] round \(number) reasoning: \(round.reasoning)")
            rounds.append(round)
            guard !round.toolCalls.isEmpty else { break }
            entries += try Self.entries(after: round, number: number, log: log)
        }
        let answer = try #require(rounds.last, "[\(label)] no round")
        entries.append(
            .response(Transcript.Response(assetIDs: [], segments: [Self.segment(answer.text)])))
        entries.append(Self.prompt(Inkling2BankFixture.followUp))
        let followUp = try await Self.runRound(
            entries, thinking: thinking, model: model, executor: executor)
        Self.report("executor[\(label)] follow-up: \(followUp.summary)")

        #expect(log.isTheTwoStepLoop, "[\(label)] calls \(log.calls)")
        #expect(Inkling2BankFixture.namesTheBalance(answer.text), "[\(label)] \(answer.text)")
        for (number, round) in rounds.enumerated().dropFirst() {
            #expect(
                round.cachedTokenCount > 0,
                "[\(label)] round \(number) after a tool result reused no prompt token")
        }
        #expect(followUp.cachedTokenCount > 0, "[\(label)] follow-up reused no prompt token")
        #expect(
            followUp.text.contains(Inkling2BankFixture.customerID),
            "[\(label)] follow-up \(followUp.text)")
        let streamed = rounds + [followUp]
        #expect(
            streamed.allSatisfy { !$0.text.contains("<|") }, "[\(label)] a marker reached the text")
        return streamed.map(\.reasoning).joined()
    }
}

#endif  // FoundationModelsIntegration
