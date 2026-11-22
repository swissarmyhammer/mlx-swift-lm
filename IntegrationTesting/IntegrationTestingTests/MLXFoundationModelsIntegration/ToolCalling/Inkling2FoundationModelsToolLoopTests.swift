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
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

// MARK: - The tools

/// The argument of the customer tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct FindCustomerArguments {
    @Guide(description: "The full name of the customer")
    var name: String
}

/// The argument of the balance tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct GetBalanceArguments {
    @Guide(description: "The customer id, for example C-1000")
    var customerId: String
}

/// The customer tool of a `LanguageModelSession`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct FindCustomerTool: FoundationModels.Tool {
    let name = Inkling2BankFixture.findCustomerTool
    let description = Inkling2BankFixture.findCustomerDescription
    let log: Inkling2ToolCallLog

    func call(arguments: FindCustomerArguments) async throws -> String {
        log.dispatch(name, argument: arguments.name)
    }
}

/// The balance tool of a `LanguageModelSession`.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct GetBalanceTool: FoundationModels.Tool {
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
        options(toolCallingMode: .allowed)
    }

    /// Greedy options with the round budget and the tool-calling mode `mode`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func options(toolCallingMode mode: GenerationOptions.ToolCallingMode)
        -> GenerationOptions
    {
        GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: maxTokens,
            toolCallingMode: mode)
    }

    /// The model of the pinned checkpoint. Thinking off is a model that does not declare
    /// `.reasoning`, which the executor renders with the thinking effort `none`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeModel(thinking: Bool) -> MLXLanguageModel {
        MLXLanguageModel(
            configuration: inkling2Mini4Bit,
            capabilities: thinking ? [.reasoning, .toolCalling] : [.toolCalling],
            weightsLocation: {
                testWeightsLocation(modelID: $0, revision: inkling2Mini4BitRevision)
            },
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
        executor: MLXLanguageModel.Executor,
        toolCallingMode: GenerationOptions.ToolCallingMode = .allowed
    ) async throws -> Inkling2Round {
        let request = makeExecutorRequest(
            transcript: Transcript(entries: entries), enabledTools: toolDefinitions,
            generationOptions: options(toolCallingMode: toolCallingMode),
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
            Inkling2ResultLog.report(
                "INKLING2 FM TOOLS",
                "session[\(label)] entry: \(String(describing: entry).prefix(400))")
        }
        Inkling2ResultLog.report(
            "INKLING2 FM TOOLS",
            "session[\(label)] answer cached \(answer.usage.input.cachedTokenCount) of "
                + "\(answer.usage.input.totalTokenCount) <<<\(answer.content)>>>")
        Inkling2ResultLog.report(
            "INKLING2 FM TOOLS",
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
        let session = try await Self.runExecutorSession(
            thinking: thinking, label: label, category: "INKLING2 FM TOOLS",
            toolCallingMode: .allowed)

        for (number, round) in session.rounds.enumerated().dropFirst() {
            #expect(
                round.cachedTokenCount > 0,
                "[\(label)] round \(number) after a tool result reused no prompt token")
        }
        let streamed = session.rounds + [session.followUp]
        Self.expectTheLoopAnswered(
            log: session.log, answer: session.answer, followUp: session.followUp,
            streamed: streamed, label: label)
        return streamed.map(\.reasoning).joined()
    }

    /// The rounds of one executor session: the rounds of the tool loop, the last of which
    /// answers, and the follow-up turn after the answer.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ExecutorSession {
        /// The rounds of the loop, in order; the last one answers the question.
        let rounds: [Inkling2Round]
        /// The round that answered the question, the last of ``rounds``.
        let answer: Inkling2Round
        /// The round of the follow-up turn.
        let followUp: Inkling2Round
        /// The calls the loop dispatched.
        let log: Inkling2ToolCallLog
    }

    /// Runs the tool loop of one session one executor round at a time, then the follow-up turn.
    ///
    /// The loop appends the reasoning, the calls and the results of each round that called a
    /// tool, and stops at the first round that calls none, which answers, or after
    /// ``maxRounds`` rounds. In `.required` mode the first ``requiredRounds`` rounds run in
    /// that mode and each must pass `checkRequiredRound`; every other round, and the follow-up
    /// turn, runs in `.allowed` mode.
    ///
    /// - Parameters:
    ///   - thinking: whether the model declares reasoning.
    ///   - label: the label of the log lines and the issues.
    ///   - category: the category of the log lines.
    ///   - toolCallingMode: the mode of the tool rounds, `.allowed` or `.required`.
    ///   - checkRequiredRound: the checks of a round that ran in `.required` mode, given the
    ///     round and its number.
    /// - Returns: the rounds, the answer, the follow-up turn and the calls of the session.
    /// - Throws: the error of a round, of `checkRequiredRound`, or of a loop without a round.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func runExecutorSession(
        thinking: Bool, label: String, category: String,
        toolCallingMode: GenerationOptions.ToolCallingMode,
        checkRequiredRound: (_ round: Inkling2Round, _ number: Int) throws -> Void = { _, _ in }
    ) async throws -> ExecutorSession {
        let model = makeModel(thinking: thinking)
        let executor = try makeMLXExecutor(for: model)
        let log = Inkling2ToolCallLog()
        var entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    segments: [segment(Inkling2BankFixture.instructions)],
                    toolDefinitions: toolDefinitions)),
            prompt(Inkling2BankFixture.question),
        ]
        let forcedRounds = toolCallingMode.kind == .required ? requiredRounds : 0

        var rounds: [Inkling2Round] = []
        for number in 0 ..< maxRounds {
            let forced = number < forcedRounds
            let clock = ContinuousClock()
            let start = clock.now
            let round = try await runRound(
                entries, thinking: thinking, model: model, executor: executor,
                toolCallingMode: forced ? toolCallingMode : .allowed)
            let seconds = (clock.now - start) / .seconds(1)
            Inkling2ResultLog.report(
                category,
                "executor[\(label)] round \(number) (\(String(format: "%.2f", seconds)) s): "
                    + round.summary)
            Inkling2ResultLog.report(
                category, "executor[\(label)] round \(number) reasoning: \(round.reasoning)")
            if forced {
                try checkRequiredRound(round, number)
            }
            rounds.append(round)
            guard !round.toolCalls.isEmpty else { break }
            entries += try Self.entries(after: round, number: number, log: log)
        }
        let answer = try #require(rounds.last, "[\(label)] no round")
        entries.append(
            .response(Transcript.Response(assetIDs: [], segments: [segment(answer.text)])))
        entries.append(prompt(Inkling2BankFixture.followUp))
        let followUp = try await runRound(
            entries, thinking: thinking, model: model, executor: executor)
        Inkling2ResultLog.report(category, "executor[\(label)] follow-up: \(followUp.summary)")
        return ExecutorSession(rounds: rounds, answer: answer, followUp: followUp, log: log)
    }

    /// Records an issue unless the executor loop called the two tools in order, its answer
    /// names the balance, the follow-up turn reused the prompt cache and names the customer
    /// id, and no protocol marker reached the text of a round.
    ///
    /// - Parameters:
    ///   - log: the calls the loop dispatched.
    ///   - answer: the round that answered the question.
    ///   - followUp: the round of the follow-up turn.
    ///   - streamed: the rounds whose text must hold no marker.
    ///   - label: the label of the issues.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func expectTheLoopAnswered(
        log: Inkling2ToolCallLog, answer: Inkling2Round, followUp: Inkling2Round,
        streamed: [Inkling2Round], label: String
    ) {
        #expect(log.isTheTwoStepLoop, "[\(label)] calls \(log.calls)")
        #expect(Inkling2BankFixture.namesTheBalance(answer.text), "[\(label)] \(answer.text)")
        #expect(followUp.cachedTokenCount > 0, "[\(label)] follow-up reused no prompt token")
        #expect(
            followUp.text.contains(Inkling2BankFixture.customerID),
            "[\(label)] follow-up \(followUp.text)")
        #expect(
            streamed.allSatisfy { !$0.text.contains("<|") }, "[\(label)] a marker reached the text")
    }

    /// The number of rounds of the required-mode loop that must call a tool: the customer
    /// lookup, then the balance read.
    private static let requiredRounds = Inkling2ToolCallLog.twoStepCallCount

    /// The loop in required tool mode: the customer lookup and the balance read each run in
    /// `.required` mode, which must give one call in Inkling 2's own tool frame, then the
    /// answer and the follow-up turn run in `.allowed` mode. A required round runs the guided
    /// loop, which carries no prompt cache, thus the answer round starts cold and the follow-up
    /// turn must reuse the cache of the answer round. Runs with thinking on and with thinking
    /// off.
    @Test(arguments: thinkingStates)
    func requiredModeCallsEachToolOfTheLoop(thinking: Bool) async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectRequiredModeCompletesTheLoop(thinking: thinking)
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// Runs the loop of ``requiredModeCallsEachToolOfTheLoop(thinking:)``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectRequiredModeCompletesTheLoop(thinking: Bool) async throws {
        let label = thinking ? "required-thinking" : "required-no-thinking"
        let session = try await Self.runExecutorSession(
            thinking: thinking, label: label, category: "INKLING2 FM REQUIRED",
            toolCallingMode: .required
        ) { round, number in
            #expect(round.text.isEmpty, "[\(label)] round \(number) answered in text")
            try #require(round.toolCalls.count == 1, "[\(label)] round \(number) \(round.summary)")
        }

        #expect(session.answer.toolCalls.isEmpty, "[\(label)] answer \(session.answer.summary)")
        Self.expectTheLoopAnswered(
            log: session.log, answer: session.answer, followUp: session.followUp,
            streamed: [session.answer, session.followUp], label: label)
    }
}

#endif  // FoundationModelsIntegration
