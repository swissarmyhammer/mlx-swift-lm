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
import os

@testable import MLXLLM
@testable import MLXLMCommon

/// The cities of the calls a session dispatched, in order.
///
/// The session dispatches on its generation task and the test reads the log on its own, thus a
/// lock guards the cities.
private struct DispatchLog: Sendable {
    private let dispatched = OSAllocatedUnfairLock<[String]>(initialState: [])

    /// Adds the city of one dispatched call.
    func record(_ city: String) {
        dispatched.withLock { $0.append(city) }
    }

    /// The cities of the calls so far.
    var cities: [String] { dispatched.withLock { $0 } }
}

/// A tool loop of a `ChatSession` in the Inkling 2 protocol.
@Suite(.serialized)
struct Inkling2ToolLoopSessionTests {

    /// The question of the user.
    private static let question = "What is the weather in Rome and in Paris?"

    /// The tokens each response may generate.
    private static let maxTokens = 32

    /// The scripted weather loop: the reasoning before the calls, the answer after the results,
    /// and the tokenizer whose vocabulary holds every text of the conversation.
    private static let loop = InklingWeatherToolLoop(
        reasoning: "I will look both cities up.", answer: "Rome is sunny and Paris is rainy.",
        otherPieces: [question])

    /// The tokenizer of ``loop``.
    private static var tokenizer: InklingFrameTokenizer { loop.tokenizer }

    /// The configuration of the scripted Inkling 2 model.
    private static let configuration = ModelConfiguration(
        id: "inkling2-tool-loop-test", toolCallFormat: .inkling,
        reasoningConfig: InklingReasoningProtocol.inkling2)

    /// The render of the first prompt: the tool declaration, the effort and the question.
    private static func firstRender() throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": question]],
            tools: [InklingWeatherToolLoop.toolSpec],
            additionalContext: nil)
    }

    /// A session of a model that writes `script`, one token for each forward pass, and that
    /// answers each tool call from ``InklingWeatherToolLoop/weather``.
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
            context,
            generateParameters: GenerateParameters(maxTokens: Self.maxTokens, temperature: 0),
            tools: [InklingWeatherToolLoop.toolSpec]
        ) { call in
            let argument = call.function.arguments[InklingWeatherToolLoop.cityArgument]
            let city = argument?.anyValue as? String ?? ""
            dispatched.record(city)
            return InklingWeatherToolLoop.weather[city] ?? ""
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
        let calling = Self.loop.callingTurn(["Rome"])
        let dispatched = DispatchLog()
        let (session, model) = session(
            script: calling + Self.loop.answeringTurn, dispatched: dispatched)

        let reply = try await session.respond(to: Self.question)

        let ledger = try Self.firstRender().count + calling.count
        #expect(dispatched.cities == ["Rome"])
        #expect(reply == Self.loop.answer)
        #expect(
            pass(of: model, feeding: Self.loop.resultFrame("Rome"), at: ledger) != nil,
            "the passes were \(model.passes.map { ($0.offset, $0.tokens) })")
    }

    @Test func `the pass after parallel calls feeds every result alone`() async throws {
        let cities = ["Rome", "Paris"]
        let calling = Self.loop.callingTurn(cities)
        let dispatched = DispatchLog()
        let (session, model) = session(
            script: calling + Self.loop.answeringTurn, dispatched: dispatched)

        _ = try await session.respond(to: Self.question)

        let ledger = try Self.firstRender().count + calling.count
        #expect(dispatched.cities == cities)
        #expect(
            pass(of: model, feeding: cities.flatMap(Self.loop.resultFrame), at: ledger) != nil,
            "the passes were \(model.passes.map { ($0.offset, $0.tokens) })")
    }

    @Test func `the thinking frame streams as reasoning`() async throws {
        let dispatched = DispatchLog()
        let (session, _) = session(
            script: Self.loop.callingTurn(["Rome"]) + Self.loop.answeringTurn,
            dispatched: dispatched)

        var reasoning = ""
        var text = ""
        for try await event in session.streamDetails(to: Self.question) {
            if case .reasoning(let chunk) = event { reasoning += chunk }
            text += event.chunk ?? ""
        }

        #expect(reasoning == Self.loop.reasoning)
        #expect(text == Self.loop.answer)
    }

    @Test func `thinking off renders the effort none`() async throws {
        let dispatched = DispatchLog()
        let (session, model) = session(script: Self.loop.answeringTurn, dispatched: dispatched)
        session.thinkingEnabled = false

        _ = try await session.respond(to: Self.question)

        let effortNone = Self.tokenizer.token(
            InklingFrameTokenizer.effortPrefix + InklingReasoningProtocol.thinkingOffEffort)
        #expect(model.passes.first?.tokens.contains(effortNone) == true)
    }
}
