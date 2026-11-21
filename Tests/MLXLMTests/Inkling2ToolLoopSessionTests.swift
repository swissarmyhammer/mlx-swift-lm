// Copyright © 2026 Apple Inc.
//
// Weight-free `ChatSession` tests of a tool loop in the Inkling 2 protocol.
//
// A scripted model writes the frames the real checkpoint writes: a thinking frame, then one tool
// frame for each call, then `<|content_model_end_sampling|>`. The session dispatches each call
// and renders the result back through the Inkling template of ``InklingFrameTokenizer``, whose
// render of a past tool call (compact JSON with sorted keys) differs from the call the model
// wrote. The tests read which tokens each forward pass fed.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The cities of the calls a session dispatched, in order.
///
/// The session dispatches on its generation task and the test reads the log on its own, thus a
/// lock guards the cities. The conformance is `@unchecked` because every access to the one
/// stored list takes the lock.
private final class DispatchLog: @unchecked Sendable {
    private let lock = NSLock()
    private var dispatched: [String] = []

    /// Adds the city of one dispatched call.
    func record(_ city: String) {
        lock.withLock { dispatched.append(city) }
    }

    /// The cities of the calls so far.
    var cities: [String] { lock.withLock { dispatched } }
}

/// A tool loop of a `ChatSession` in the Inkling 2 protocol.
@Suite(.serialized)
struct Inkling2ToolLoopSessionTests {

    /// The question of the user.
    private static let question = "What is the weather in Rome and in Paris?"

    /// The reasoning the model writes before its calls.
    private static let reasoning = "I will look both cities up."

    /// The answer the model writes after the tool results.
    private static let answer = "Rome is sunny and Paris is rainy."

    /// The name of the tool.
    private static let toolName = "get_weather"

    /// The weather of each city the tool knows.
    private static let weather = ["Rome": "Sunny, 21 C", "Paris": "Rainy, 12 C"]

    /// The tool the session declares.
    private static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": toolName,
            "description": "Get the current weather for a city",
            "parameters": [
                "type": "object",
                "properties": ["city": ["type": "string"] as [String: any Sendable]],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// The call of the tool for `city` as the model writes it: JSON with spaces.
    private static func writtenCall(_ city: String) -> String {
        #"{"name": "\#(toolName)", "args": {"city": "\#(city)"}}"#
    }

    /// The call of the tool for `city` as the template renders it.
    private static func renderedCall(_ city: String) -> String {
        InklingFrameTokenizer.renderedCall(name: toolName, arguments: ["city": city])
    }

    /// The tokenizer whose vocabulary holds every text of the conversation.
    private static let tokenizer = InklingFrameTokenizer(
        pieces: [question, reasoning, answer, toolName]
            + weather.keys.flatMap { [writtenCall($0), renderedCall($0)] }
            + Array(weather.values))

    /// The configuration of the scripted Inkling 2 model.
    private static let configuration = ModelConfiguration(
        id: "inkling2-tool-loop-test", toolCallFormat: .inkling,
        reasoningConfig: InklingReasoningProtocol.inkling2)

    /// The thinking frame the model writes.
    private static var thinkingFrame: [Int] {
        tokens(.messageModel, .contentThinking) + [tokenizer.token(reasoning)]
            + tokens(.endMessage)
    }

    /// The tool frame the model writes for `city`.
    private static func callFrame(_ city: String) -> [Int] {
        tokens(.messageModel) + [tokenizer.token(toolName)] + tokens(.contentToolJSON)
            + [tokenizer.token(writtenCall(city))] + tokens(.endMessage)
    }

    /// The tool result frame the template renders for `city`.
    private static func resultFrame(_ city: String) -> [Int] {
        tokens(.messageTool) + [tokenizer.token(toolName)] + tokens(.contentText)
            + [tokenizer.token(weather[city] ?? "")] + tokens(.endMessage)
    }

    /// The tokens the model writes in the turn that calls the tool for each of `cities`.
    private static func callingTurn(_ cities: [String]) -> [Int] {
        thinkingFrame + cities.flatMap(callFrame) + tokens(.endSampling)
    }

    /// The tokens the model writes in the turn that answers from the tool results.
    private static var answeringTurn: [Int] {
        tokens(.messageModel, .contentText) + [tokenizer.token(answer)]
            + tokens(.endMessage, .endSampling)
    }

    /// The tokens of `markers`.
    private static func tokens(_ markers: InklingFrameTokenizer.Marker...) -> [Int] {
        markers.map(tokenizer.token)
    }

    /// The render of the first prompt: the tool declaration, the effort and the question.
    private static func firstRender() throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": question]], tools: [weatherTool],
            additionalContext: nil)
    }

    /// A session of a model that writes `script`, one token for each forward pass, and that
    /// answers each tool call from ``weather``.
    ///
    /// - Parameters:
    ///   - script: the tokens of the forward passes, in order.
    ///   - dispatched: receives the city of each dispatched call.
    /// - Returns: the session and its model.
    private func session(
        script: [Int], dispatched: DispatchLog
    ) -> (session: ChatSession, model: ScriptedTokenModel) {
        let tokenizer = Self.tokenizer
        let model = ScriptedTokenModel(
            script: script, finalToken: tokenizer.token(.endSampling),
            vocabularySize: tokenizer.vocabularySize, toolCallFormat: .inkling)
        let context = ModelContext(
            configuration: Self.configuration, model: model,
            processor: ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: Self.configuration),
            tokenizer: tokenizer)
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 32, temperature: 0),
            tools: [Self.weatherTool]
        ) { call in
            let city = call.function.arguments["city"]?.anyValue as? String ?? ""
            dispatched.record(city)
            return Self.weather[city] ?? ""
        }
        return (session, model)
    }

    /// The pass that fed `tokens` at `offset`, or nil.
    private func pass(
        of model: ScriptedTokenModel, feeding tokens: [Int], at offset: Int
    ) -> ScriptedTokenModel.Pass? {
        model.passes.first { $0.offset == offset && $0.tokens == tokens }
    }

    @Test func `the pass after a tool result feeds the result alone`() async throws {
        let calling = Self.callingTurn(["Rome"])
        let dispatched = DispatchLog()
        let (session, model) = session(script: calling + Self.answeringTurn, dispatched: dispatched)

        let reply = try await session.respond(to: Self.question)

        let ledger = try Self.firstRender().count + calling.count
        #expect(dispatched.cities == ["Rome"])
        #expect(reply == Self.answer)
        #expect(
            pass(of: model, feeding: Self.resultFrame("Rome"), at: ledger) != nil,
            "the passes were \(model.passes.map { ($0.offset, $0.tokens) })")
    }

    @Test func `the pass after parallel calls feeds every result alone`() async throws {
        let cities = ["Rome", "Paris"]
        let calling = Self.callingTurn(cities)
        let dispatched = DispatchLog()
        let (session, model) = session(script: calling + Self.answeringTurn, dispatched: dispatched)

        _ = try await session.respond(to: Self.question)

        let ledger = try Self.firstRender().count + calling.count
        #expect(dispatched.cities == cities)
        #expect(
            pass(of: model, feeding: cities.flatMap(Self.resultFrame), at: ledger) != nil,
            "the passes were \(model.passes.map { ($0.offset, $0.tokens) })")
    }

    @Test func `the thinking frame streams as reasoning`() async throws {
        let dispatched = DispatchLog()
        let (session, _) = session(
            script: Self.callingTurn(["Rome"]) + Self.answeringTurn, dispatched: dispatched)

        var reasoning = ""
        var text = ""
        for try await event in session.streamDetails(to: Self.question) {
            if case .reasoning(let chunk) = event { reasoning += chunk }
            text += event.chunk ?? ""
        }

        #expect(reasoning == Self.reasoning)
        #expect(text == Self.answer)
    }

    @Test func `thinking off renders the effort none`() async throws {
        let dispatched = DispatchLog()
        let (session, model) = session(script: Self.answeringTurn, dispatched: dispatched)
        session.thinkingEnabled = false

        _ = try await session.respond(to: Self.question)

        let effortNone = Self.tokenizer.token(
            InklingFrameTokenizer.effortPrefix + InklingReasoningProtocol.thinkingOffEffort)
        #expect(model.passes.first?.tokens.contains(effortNone) == true)
    }
}
