// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of a two-step tool loop of Inkling 2 Mini through `ChatSession`: the
// model looks up a customer id with one tool, then the balance of that id with a second tool,
// then answers from the result. Each pass after a tool result, and the user turn after the
// loop, must serve part of its prompt from the cache of the passes before it, with thinking on
// and with thinking off.
//
// `Inkling2FoundationModelsToolLoopTests` runs the same loop through `MLXLanguageModel`.
//
// Each test downloads the pinned checkpoint through the Hugging Face downloader (or reuses the
// cached copy), thus none of them skips. Build for Release, or the numbers measure the debug
// build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2ToolLoopIntegrationTests`

import Foundation
import IntegrationTestHelpers
import MLXLMCommon
import MLXVLM
import Testing
import os

/// The minutes the suite may run: two tool loops and their follow-up turns on the real
/// checkpoint, with the download of the checkpoint when it is not in the cache.
private let suiteTimeLimitMinutes = 30

// MARK: - The bank

/// The bank both tool-loop suites query: one customer, a tool that finds the id of a customer
/// by name, and a tool that reads the balance of an id.
enum Inkling2BankFixture {

    /// The name of the one customer.
    static let customerName = "Ada Lovelace"

    /// The id of the one customer.
    static let customerID = "C-7731"

    /// The balance of the one customer, as the balance tool writes it.
    static let balance = "4,812.55 EUR"

    /// The spellings of the balance an answer may use.
    static let balanceSpellings = ["4,812.55", "4812.55"]

    /// The name of the tool that finds the id of a customer.
    static let findCustomerTool = "find_customer"

    /// The argument of ``findCustomerTool``.
    static let findCustomerArgument = "name"

    /// The name of the tool that reads the balance of a customer id.
    static let getBalanceTool = "get_balance"

    /// The argument of ``getBalanceTool``.
    static let getBalanceArgument = "customerId"

    /// The instructions of each session.
    static let instructions =
        "You are a banking assistant. You know no customer data yourself. To answer a question "
        + "about a balance, first call \(findCustomerTool) with the name of the customer to get "
        + "the customer id, then call \(getBalanceTool) with that customer id. Answer in one "
        + "short sentence."

    /// The question that needs both tools.
    static let question = "What is the account balance of the customer named \(customerName)?"

    /// The user turn after the loop, which the history of the loop answers.
    static let followUp = "Which customer id did you look up? Reply with the id only."

    /// The description of ``findCustomerTool``.
    static let findCustomerDescription = "Find the customer id of a customer by full name"

    /// The description of ``getBalanceTool``.
    static let getBalanceDescription = "Read the account balance of a customer id"

    /// The result of the tool `tool` for its one argument `argument`.
    ///
    /// - Parameters:
    ///   - tool: the name of the tool the model called.
    ///   - argument: the value of the one argument of the call.
    /// - Returns: the result as JSON text.
    static func result(of tool: String, argument: String) -> String {
        switch tool {
        case findCustomerTool where argument.localizedCaseInsensitiveContains("lovelace"):
            return #"{"customerId":"\#(customerID)"}"#
        case getBalanceTool where argument == customerID:
            return #"{"customerId":"\#(customerID)","balance":"\#(balance)"}"#
        default:
            return #"{"error":"no record for \#(argument)"}"#
        }
    }

    /// Whether `text` names the balance.
    static func namesTheBalance(_ text: String) -> Bool {
        balanceSpellings.contains { text.contains($0) }
    }

    /// The tool specifications of a `ChatSession`.
    static var toolSpecs: [ToolSpec] {
        [
            spec(findCustomerTool, findCustomerDescription, argument: findCustomerArgument),
            spec(getBalanceTool, getBalanceDescription, argument: getBalanceArgument),
        ]
    }

    /// One tool with one required string argument.
    private static func spec(_ name: String, _ description: String, argument: String) -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": [argument: ["type": "string"] as [String: any Sendable]],
                    "required": [argument],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}

/// The calls a tool loop dispatched, in order: the tool name and its one argument.
///
/// The session dispatches on its generation task and the test reads the log on its own, thus a
/// lock guards the calls.
final class Inkling2ToolCallLog: Sendable {

    /// The number of calls of the two-step loop: the customer lookup, then the balance read.
    static let twoStepCallCount = 2

    private let dispatched = OSAllocatedUnfairLock<[(tool: String, argument: String)]>(
        initialState: [])

    /// Records one call and gives the result the bank answers it with.
    ///
    /// - Parameters:
    ///   - tool: the name of the tool.
    ///   - argument: the value of its one argument.
    /// - Returns: the result of the call.
    func dispatch(_ tool: String, argument: String) -> String {
        dispatched.withLock { $0.append((tool, argument)) }
        return Inkling2BankFixture.result(of: tool, argument: argument)
    }

    /// The calls so far.
    var calls: [(tool: String, argument: String)] { dispatched.withLock { $0 } }

    /// Whether the calls are the two-step loop: the customer id of the customer, then the
    /// balance of that id.
    var isTheTwoStepLoop: Bool {
        let calls = calls
        guard calls.count >= Self.twoStepCallCount else { return false }
        let lookup = calls[0]
        let read = calls[1]
        return lookup.tool == Inkling2BankFixture.findCustomerTool
            && lookup.argument.localizedCaseInsensitiveContains("lovelace")
            && read.tool == Inkling2BankFixture.getBalanceTool
            && read.argument == Inkling2BankFixture.customerID
    }
}

// MARK: - One generation pass

extension CollectedGeneration {
    /// One line that names the calls, the token counts and rates, the reasoning length and the
    /// text of one pass.
    fileprivate var passSummary: String {
        let calls = toolCalls.map { "\($0.function.name)(\($0.function.arguments))" }
        return "calls \(calls), \(summary), reasoning \(reasoning.count) chars, text <<<\(text)>>>"
    }
}

// MARK: - Tests

@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct Inkling2ToolLoopIntegrationTests {

    /// Tokens each pass may generate.
    private static let maxTokens = 1_024

    /// The one string argument of `call`, or an empty text.
    private static func argument(of call: ToolCall) -> String {
        call.function.arguments.values.first?.anyValue as? String ?? ""
    }

    /// A greedy session with both tools, which dispatches each call into `log`.
    private func makeSession(thinking: Bool, log: Inkling2ToolCallLog) async throws
        -> ChatSession
    {
        let container = try await Inkling2MiniCheckpoint.shared.container()
        let session = ChatSession(
            container, instructions: Inkling2BankFixture.instructions,
            generateParameters: GenerateParameters(maxTokens: Self.maxTokens, temperature: 0),
            tools: Inkling2BankFixture.toolSpecs
        ) { call in
            log.dispatch(call.function.name, argument: Self.argument(of: call))
        }
        session.thinkingEnabled = thinking
        return session
    }

    /// The model calls the customer tool, then the balance tool with the id the first result
    /// gave, then answers with the balance, at the default thinking effort and with thinking
    /// off. Each pass after a tool result, and the user turn after the loop, reuses the cache
    /// of the passes before it. The thinking frames stream as reasoning, and thinking off (the
    /// effort `none`) makes the model reason less.
    @Test
    func chatSessionCompletesATwoStepToolLoop() async throws {
        let thinkingReasoning = try await expectTheLoopCompletes(thinking: true)
        let offReasoning = try await expectTheLoopCompletes(thinking: false)

        #expect(!thinkingReasoning.isEmpty, "thinking on streamed no reasoning")
        #expect(
            offReasoning.count < thinkingReasoning.count,
            "thinking off reasoned \(offReasoning.count) chars, on \(thinkingReasoning.count)")
    }

    /// Runs the loop and the follow-up turn of one session, and records an issue unless the
    /// calls, the answers and the cache reuse are right.
    ///
    /// - Parameter thinking: whether thinking is on.
    /// - Returns: the reasoning the session streamed.
    private func expectTheLoopCompletes(thinking: Bool) async throws -> String {
        let label = thinking ? "thinking" : "no-thinking"
        let log = Inkling2ToolCallLog()
        let session = try await makeSession(thinking: thinking, log: log)

        let passes = try await CollectedGeneration.collectPasses(
            session.streamDetails(to: Inkling2BankFixture.question))
        for (index, pass) in passes.enumerated() {
            Inkling2ResultLog.report(
                "INKLING2 TOOLS", "chat[\(label)] pass \(index): \(pass.passSummary)")
            Inkling2ResultLog.report(
                "INKLING2 TOOLS", "chat[\(label)] pass \(index) reasoning: \(pass.reasoning)")
        }
        let followUp = try await CollectedGeneration.collectPasses(
            session.streamDetails(to: Inkling2BankFixture.followUp))
        for pass in followUp {
            Inkling2ResultLog.report(
                "INKLING2 TOOLS", "chat[\(label)] follow-up: \(pass.passSummary)")
        }
        Inkling2ResultLog.report("INKLING2 TOOLS", "chat[\(label)] dispatched \(log.calls)")

        #expect(log.isTheTwoStepLoop, "[\(label)] calls \(log.calls)")
        let answer = try #require(passes.last, "[\(label)] no pass")
        #expect(Inkling2BankFixture.namesTheBalance(answer.text), "[\(label)] \(answer.text)")
        for (index, pass) in passes.enumerated().dropFirst() {
            #expect(
                (pass.info?.cachedPromptTokenCount ?? 0) > 0,
                "[\(label)] pass \(index) after a tool result reused no prompt token")
        }
        let followUpAnswer = try #require(followUp.last, "[\(label)] no follow-up pass")
        #expect(
            (followUpAnswer.info?.cachedPromptTokenCount ?? 0) > 0,
            "[\(label)] the turn after the tool loop reused no prompt token")
        #expect(
            followUpAnswer.text.contains(Inkling2BankFixture.customerID),
            "[\(label)] follow-up \(followUpAnswer.text)")
        let streamed = passes + followUp
        #expect(
            streamed.allSatisfy { !$0.text.contains("<|") }, "[\(label)] a marker reached the text")
        return streamed.map(\.reasoning).joined()
    }
}
