// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMTestSupport
import MLXNN
import MLXOptimizers
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase, ChatSessionFixtures {

    private actor SteeringGate {
        private var open = false
        private var waiter: CheckedContinuation<Void, Never>?

        func wait() async {
            if !open { await withCheckedContinuation { waiter = $0 } }
        }

        func release() {
            open = true
            waiter?.resume()
            waiter = nil
        }
    }

    private struct GatedSteeringProcessor: UserInputProcessor {
        let base: TestInputProcessor
        let ready: AsyncStream<Void>.Continuation
        let gate: SteeringGate
        var rejectInstruction: String? = nil

        func prepare(input: UserInput) async throws -> LMInput {
            ready.yield(())
            await gate.wait()
            try Task.checkCancellation()
            if let rejectInstruction, case .chat(let chat) = input.prompt,
                chat.last?.content == rejectInstruction
            {
                throw SteeringTestError.preparationFailed
            }
            return try base.prepare(input: input)
        }
    }

    private actor SteeringGateSequence {
        let gates: [SteeringGate]
        var step = 0

        init(_ gates: [SteeringGate]) { self.gates = gates }

        func next() -> SteeringGate {
            defer { step += 1 }
            return gates[step]
        }
    }

    private struct SteppedSteeringProcessor: UserInputProcessor {
        let base: TestInputProcessor
        let ready: AsyncStream<Void>.Continuation
        let gates: SteeringGateSequence

        func prepare(input: UserInput) async throws -> LMInput {
            let gate = await gates.next()
            ready.yield(())
            await gate.wait()
            try Task.checkCancellation()
            return try base.prepare(input: input)
        }
    }

    func testSteeringReusesPrefixAndKeepsOneTurn() async throws {
        try await checkSteering(policy: .nextSafeBoundary, speculative: false)
    }

    func testQueuedSteeringFinishesCurrentStep() async throws {
        try await checkSteering(policy: .nextStepBoundary, speculative: false)
    }

    func testSteeringFinalizesSpeculativeLookahead() async throws {
        try await checkSteering(policy: .nextSafeBoundary, speculative: true)
    }

    func testSteeringRebuildsWhenTemplateRewritesPrefix() async throws {
        try await checkSteering(policy: .nextSafeBoundary, speculative: false, rewritePrefix: true)
    }

    private func checkSteering(
        policy: SteeringPolicy, speculative: Bool, rewritePrefix: Bool = false
    ) async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let (lengths, lengthContinuation) = AsyncStream<Int>.makeStream()
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let gate = SteeringGate()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: lengthContinuation,
            rewritesPrefixOnContinuation: rewritePrefix)
        let configuration = ModelConfiguration(id: "steering-test")
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: configuration, tokenizer: tokenizer)
        let draft = speculative ? ModelContainer(context: model(processor: base)) : nil
        let session = ChatSession(
            context,
            speculativeDecoding: draft.map { SpeculativeDecodingConfig(draftModel: $0) },
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let stream = session.streamDetails(to: "original task")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        let first = try session.steer("keep the original task", policy: policy)
        let second = try session.steer("also include tests", policy: policy)
        await gate.release()

        var infos: [GenerateCompletionInfo] = []
        var applied: [UUID] = []
        for try await event in stream {
            switch event {
            case .info(let info): infos.append(info)
            case .steering(.applied(let ids)):
                XCTAssertEqual(infos.count, 1)
                applied += ids
            default: break
            }
        }
        XCTAssertEqual(applied, [first, second])
        XCTAssertEqual(infos.count, 2)
        guard infos.count == 2 else { return }
        if case .nextSafeBoundary = policy {
            XCTAssertEqual(infos[0].stopReason, .steered)
            XCTAssertEqual(infos[0].generationTokenCount, 1)
        } else {
            XCTAssertEqual(infos[0].stopReason, .length)
            XCTAssertEqual(infos[0].generationTokenCount, 8)
        }
        XCTAssertEqual(infos[1].stopReason, .length)
        XCTAssertEqual(
            infos[1].cachedPromptTokenCount,
            rewritePrefix ? 0 : infos[0].promptTokenCount + infos[0].generationTokenCount)
        var lengthIterator = lengths.makeAsyncIterator()
        _ = await lengthIterator.next()
        let secondLength = await lengthIterator.next()
        XCTAssertEqual(infos[1].totalPromptTokenCount, secondLength)
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        let rendered = await messageIterator.next()
        XCTAssertEqual(rendered?.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(rendered?.first?.content, "original task")
        XCTAssertEqual(rendered?.last?.content, "keep the original task\n\nalso include tests")
        XCTAssertFalse(session.canSteer())
        XCTAssertThrowsError(try session.steer("too late")) {
            XCTAssertEqual($0 as? SteeringError, .noActiveResponse)
        }
        // Subsequent calls reuse the same cache after all speculative cleanup.
        let followup = try await collectGeneration(session.streamDetails(to: "next turn"))
        XCTAssertEqual(followup.info.generationTokenCount, 8)
        if !speculative { XCTAssertGreaterThan(followup.info.cachedPromptTokenCount, 0) }
    }

    func testSteeringDuringToolDispatchWaitsForResult() async throws {
        let output = "{\"name\":\"weather\",\"arguments\":{}}"
        let tokenizer = LiteralTokenizer(output: output + "done")
        let configuration = ModelConfiguration(
            id: "steering-tool-test", eosTokenIds: [99], toolCallFormat: .json)
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let calls = TokenSequenceState(tokens: [0, 1])
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let session = ChatSession(
            model(processor: base),
            generateParameters: GenerateParameters(maxTokens: 100, temperature: 0),
            components: GenerationComponents {
                let text = calls.current == 0 ? output : "done"
                calls.advance()
                return ForceTokenSequenceProcessor(
                    state: TokenSequenceState(tokens: tokenizer.tokens(for: text) + [99]))
            },
            tools: [["function": ["name": "weather"] as [String: any Sendable]]],
            toolDispatch: { _ in
                readyContinuation.yield(())
                await gate.wait()
                try Task.checkCancellation()
                return "sunny"
            })
        let stream = session.streamDetails(to: "weather please")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        let id = try session.steer("give temperatures in Celsius")
        await gate.release()
        var applied: [UUID] = []
        var toolCount = 0
        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            switch event {
            case .steering(.applied(let ids)): applied += ids
            case .toolCall: toolCount += 1
            case .info(let info): infos.append(info)
            default: break
            }
        }
        XCTAssertEqual(toolCount, 0)
        XCTAssertEqual(applied, [id])
        XCTAssertEqual(infos.map(\.stopReason), [.stop, .steered, .stop])
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        let toolContinuation = await messageIterator.next()
        let rendered = await messageIterator.next()
        XCTAssertEqual(toolContinuation?.map(\.role), [.user, .assistant, .tool])
        XCTAssertEqual(rendered?.map(\.role), [.user, .assistant, .tool, .assistant, .user])
        XCTAssertEqual(rendered?[2].content, "sunny")
        XCTAssertEqual(rendered?.last?.content, "give temperatures in Celsius")
    }

    func testInterruptKeepsTheReplyAndContinuesTheCache() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let gate = SteeringGate()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: AsyncStream<Int>.makeStream().continuation)
        let configuration = ModelConfiguration(id: "interrupt-test")
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let session = ChatSession(
            Self.makeModel(
                processor: processor, configuration: configuration, tokenizer: tokenizer),
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let stream = session.streamDetails(to: "original task")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        let unapplied = try session.steer("never applied", policy: .nextStepBoundary)
        try session.interrupt()
        XCTAssertFalse(session.canSteer())
        await gate.release()

        var infos: [GenerateCompletionInfo] = []
        var failures: [SteeringFailure] = []
        for try await event in stream {
            if let info = event.info { infos.append(info) }
            if case .steering(.failed(let failure)) = event { failures.append(failure) }
            if case .steering(.applied) = event {
                XCTFail("An interrupted response applies nothing")
            }
        }
        XCTAssertEqual(infos.map(\.stopReason), [.interrupted])
        XCTAssertEqual(infos.first?.generationTokenCount, 1)
        XCTAssertEqual(failures.map(\.ids), [[unapplied]])
        XCTAssertEqual(failures.first?.reason, .responseEnded)
        XCTAssertThrowsError(try session.interrupt()) {
            XCTAssertEqual($0 as? SteeringError, .noActiveResponse)
        }

        let followup = try await collectGeneration(session.streamDetails(to: "next turn"))
        XCTAssertEqual(
            followup.info.cachedPromptTokenCount,
            infos.first.map { $0.promptTokenCount + $0.generationTokenCount })
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        let rendered = await messageIterator.next()
        XCTAssertEqual(rendered?.map(\.role), [.user, .assistant, .user])
    }

    func testInterruptAnswersToolResultsBeforeTheResponseEnds() async throws {
        let output = "{\"name\":\"weather\",\"arguments\":{}}"
        let tokenizer = LiteralTokenizer(output: output + "done")
        let configuration = ModelConfiguration(
            id: "interrupt-tool-test", eosTokenIds: [99], toolCallFormat: .json)
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let calls = TokenSequenceState(tokens: [0, 1])
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let session = ChatSession(
            model(processor: base),
            generateParameters: GenerateParameters(maxTokens: 100, temperature: 0),
            components: GenerationComponents {
                let text = calls.current == 0 ? output : "done"
                calls.advance()
                return ForceTokenSequenceProcessor(
                    state: TokenSequenceState(tokens: tokenizer.tokens(for: text) + [99]))
            },
            tools: [["function": ["name": "weather"] as [String: any Sendable]]],
            toolDispatch: { _ in
                readyContinuation.yield(())
                await gate.wait()
                return "sunny"
            })
        let stream = session.streamDetails(to: "weather please")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.interrupt()
        await gate.release()
        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            if let info = event.info { infos.append(info) }
        }
        XCTAssertEqual(infos.map(\.stopReason), [.stop, .interrupted])
        XCTAssertEqual(infos.last?.generationTokenCount, 1)

        _ = try await session.respond(to: "thanks")
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        _ = await messageIterator.next()
        let rendered = await messageIterator.next()
        XCTAssertEqual(rendered?.map(\.role), [.user, .assistant, .tool, .assistant, .user])
    }

    func testSessionWithoutTranscriptCanStillBeInterrupted() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: AsyncStream<Int>.makeStream().continuation)
        let configuration = ModelConfiguration(id: "raw-cache-interrupt-test")
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: DefaultMessageGenerator())
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: configuration, tokenizer: tokenizer)
        let session = ChatSession(
            context, cache: try context.model.newCache(parameters: nil),
            generateParameters: GenerateParameters(maxTokens: 8, temperature: 0),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let stream = session.streamDetails(to: "hello")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.interrupt()
        await gate.release()

        var infos: [GenerateCompletionInfo] = []
        for try await event in stream {
            if let info = event.info { infos.append(info) }
        }
        XCTAssertEqual(infos.map(\.stopReason), [.interrupted])
        XCTAssertEqual(infos.first?.generationTokenCount, 1)
    }

    func testFailedSteeringPreparationRetainsCompletedToolResult() async throws {
        let output = "{\"name\":\"weather\",\"arguments\":{}}"
        let tokenizer = LiteralTokenizer(output: output + "done")
        let configuration = ModelConfiguration(
            id: "steering-tool-failure-test", eosTokenIds: [99], toolCallFormat: .json)
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let base = TestInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let prepareGate = SteeringGate()
        await prepareGate.release()
        let processor = GatedSteeringProcessor(
            base: base, ready: AsyncStream<Void>.makeStream().continuation,
            gate: prepareGate, rejectInstruction: "fail preparation")
        let context = Self.makeModel(
            processor: processor, configuration: configuration, tokenizer: tokenizer)
        let calls = TokenSequenceState(tokens: [0, 1])
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let toolGate = SteeringGate()
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 100, temperature: 0),
            components: GenerationComponents {
                let text = calls.current == 0 ? output : "done"
                calls.advance()
                return ForceTokenSequenceProcessor(
                    state: TokenSequenceState(tokens: tokenizer.tokens(for: text) + [99]))
            },
            tools: [["function": ["name": "weather"] as [String: any Sendable]]],
            toolDispatch: { _ in
                readyContinuation.yield(())
                await toolGate.wait()
                return "sunny"
            })
        let stream = session.streamDetails(to: "weather please")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.steer("fail preparation")
        await toolGate.release()
        do {
            for try await _ in stream {}
            XCTFail("Expected preparation failure")
        } catch SteeringTestError.preparationFailed {
        }
        _ = try await session.respond(to: "recover")
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        _ = await messageIterator.next()
        let recovered = await messageIterator.next()
        XCTAssertEqual(recovered?.map(\.role), [.user, .assistant, .tool, .assistant, .user])
        XCTAssertEqual(recovered?[2].content, "sunny")
        XCTAssertEqual(recovered?.last?.content, "recover")
    }

    /// Manual tool handling is a supported pattern. An instruction that needs
    /// results the session cannot produce is reported, and the caller keeps the
    /// tool call it asked for.
    func testUnappliedSteeringReportsFailureAndKeepsTheResponse() async throws {
        let session = rejectionSession(output: "{\"name\":\"weather\",\"arguments\":{}}")
        let stream = session.streamDetails(to: "weather please")
        let id = try session.steer("also explain the forecast")
        var toolCount = 0
        var failures: [SteeringFailure] = []
        for try await event in stream {
            if case .toolCall = event { toolCount += 1 }
            if case .steering(.failed(let failure)) = event { failures.append(failure) }
        }
        XCTAssertEqual(toolCount, 1)
        XCTAssertEqual(failures.count, 1)
        XCTAssertEqual(failures.first?.ids, [id])
        XCTAssertEqual(failures.first?.instructions, ["also explain the forecast"])
        XCTAssertEqual(failures.first?.reason, .toolResultsRequired)
        XCTAssertThrowsError(try session.steer("late"))
    }

    func testSessionWithoutTranscriptRejectsSteeringUpFront() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(
            context, cache: try context.model.newCache(parameters: nil),
            generateParameters: GenerateParameters(maxTokens: 3))
        let stream = session.streamDetails(to: "hello")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        XCTAssertFalse(session.canSteer())
        XCTAssertThrowsError(try session.steer("more detail")) {
            XCTAssertEqual($0 as? SteeringError, .notSteerable)
        }
        await gate.release()
        var text = ""
        for try await event in stream {
            if let chunk = event.chunk { text += chunk }
            if case .steering = event { XCTFail("No instruction was accepted") }
        }
        XCTAssertFalse(text.isEmpty)
    }

    func testChainedSteeringAddsOneStepPerInstruction() async throws {
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gates = [SteeringGate(), SteeringGate(), SteeringGate()]
        await gates[2].release()
        let base = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(), configuration: ModelConfiguration(id: "chained"),
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let processor = SteppedSteeringProcessor(
            base: base, ready: readyContinuation, gates: SteeringGateSequence(gates))
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(context, generateParameters: GenerateParameters(maxTokens: 4))
        let stream = session.streamDetails(to: "start")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        let first = try session.steer("more 1", policy: .nextStepBoundary)
        await gates[0].release()
        _ = await readyIterator.next()
        let second = try session.steer("more 2", policy: .nextStepBoundary)
        await gates[1].release()

        var infos: [GenerateCompletionInfo] = []
        var applied: [UUID] = []
        for try await event in stream {
            if let info = event.info { infos.append(info) }
            if case .steering(.applied(let ids)) = event { applied += ids }
        }
        XCTAssertEqual(applied, [first, second])
        XCTAssertEqual(infos.count, 3)
        XCTAssertEqual(infos.map(\.generationTokenCount), [4, 4, 4])
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        _ = await messageIterator.next()
        let rendered = await messageIterator.next()
        XCTAssertEqual(rendered?.map(\.role), [.user, .assistant, .user, .assistant, .user])
        XCTAssertEqual(rendered?.last?.content, "more 2")
    }

    func testLatestResponseTargetsExactlyOneResponse() async throws {
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(
            base: base, ready: AsyncStream<Void>.makeStream().continuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(context, generateParameters: GenerateParameters(maxTokens: 3))
        XCTAssertNil(session.latestResponse)
        let first = session.streamDetails(to: "first")
        let firstID = session.latestResponse
        let second = session.streamDetails(to: "second")
        let secondID = session.latestResponse
        XCTAssertNotNil(firstID)
        XCTAssertNotEqual(firstID, secondID)

        var appliedToSecond: [UUID] = []
        let id = try session.steer("only the second", response: secondID)
        await gate.release()
        for try await _ in first {}
        for try await event in second {
            if case .steering(.applied(let ids)) = event { appliedToSecond += ids }
        }
        XCTAssertEqual(appliedToSecond, [id])
    }

    @MainActor
    func testActorOwnedRespondAcceptsSteering() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 3),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let response = Task { @MainActor in try await session.respond(to: "original") }
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.steer("continue", policy: .nextStepBoundary)
        await gate.release()
        let text = try await response.value
        await session.synchronize()
        XCTAssertEqual(
            text, String(repeating: base.tokenizer.decode(tokenIds: [7, 7, 7]), count: 2))
    }

    func testSteeringKeepsStandardStringStream() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 3),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let stream = session.streamResponse(to: "original")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.steer("continue", policy: .nextStepBoundary)
        await gate.release()
        var response = ""
        for try await chunk in stream { response += chunk }
        XCTAssertEqual(
            response, String(repeating: base.tokenizer.decode(tokenIds: [7, 7, 7]), count: 2))
        XCTAssertThrowsError(try session.steer("late"))
    }

    func testObservedResponseReportsSteeringAndEndsEarly() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 3),
            components: GenerationComponents {
                ForceTokenSequenceProcessor(state: TokenSequenceState(tokens: [7]))
            })
        let stream = session.observe(to: "original", using: .logProbabilities())
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        let id = try session.steer("continue")
        await gate.release()
        var applied: [UUID] = []
        var infos: [GenerateCompletionInfo] = []
        var probabilityCount = 0
        for try await event in stream {
            if event.logProbabilities != nil { probabilityCount += 1 }
            if case .steering(.applied(let ids)) = event.generation { applied += ids }
            if let info = event.generation?.info { infos.append(info) }
        }
        XCTAssertEqual(applied, [id])
        XCTAssertEqual(infos.map(\.stopReason), [.steered, .length])
        XCTAssertEqual(probabilityCount, infos.map(\.generationTokenCount).reduce(0, +))
    }

    func testUnsteeredResponseStillSupportsManualTools() async throws {
        let session = rejectionSession(output: "{\"name\":\"weather\",\"arguments\":{}}")
        var calls = 0
        for try await event in session.streamDetails(to: "weather please") {
            if case .toolCall = event { calls += 1 }
            if case .steering(.applied) = event { XCTFail("No instruction was submitted") }
        }
        XCTAssertEqual(calls, 1)
    }

    private enum SteeringTestError: Error { case preparationFailed }

    func testFailedSteeringPreparationPreservesCommittedConversation() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let (messages, messageContinuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(), configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: messageContinuation))
        let processor = GatedSteeringProcessor(
            base: base, ready: readyContinuation, gate: gate, rejectInstruction: "fail preparation")
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(context, generateParameters: GenerateParameters(maxTokens: 3))
        let stream = session.streamDetails(to: "original")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.steer("fail preparation")
        await gate.release()
        do {
            for try await event in stream {
                if case .steering(.applied) = event { XCTFail("Failed input cannot be applied") }
            }
            XCTFail("Expected preparation failure")
        } catch SteeringTestError.preparationFailed {
        }
        _ = try await session.respond(to: "recover")
        var messageIterator = messages.makeAsyncIterator()
        _ = await messageIterator.next()
        let recovered = await messageIterator.next()
        XCTAssertEqual(recovered?.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(recovered?.first?.content, "original")
        XCTAssertEqual(recovered?.last?.content, "recover")
        XCTAssertThrowsError(try session.steer("too late"))
    }

    func testCancelAndJoinTurnBeforeRunnerStarts() async throws {
        let session = ChatSession(model(), generateParameters: GenerateParameters(maxTokens: 3))
        let stream = session.streamDetails(to: "cancel immediately")
        let consumer = Task { for try await _ in stream {} }
        consumer.cancel()
        _ = await consumer.result
        await session.synchronize()
        XCTAssertThrowsError(try session.steer("late"))
        let result = try await session.respond(to: "new task")
        XCTAssertFalse(result.isEmpty)
    }

    func testCancelSteerableTurnWhilePreparingThenReuseSession() async throws {
        let (ready, readyContinuation) = AsyncStream<Void>.makeStream()
        let gate = SteeringGate()
        let base = TestInputProcessor()
        let processor = GatedSteeringProcessor(base: base, ready: readyContinuation, gate: gate)
        let context = Self.makeModel(
            processor: processor, configuration: base.configuration, tokenizer: base.tokenizer)
        let session = ChatSession(context, generateParameters: GenerateParameters(maxTokens: 3))
        let stream = session.streamDetails(to: "cancel me")
        var readyIterator = ready.makeAsyncIterator()
        _ = await readyIterator.next()
        try session.steer("pending instruction")
        let consumer = Task { for try await _ in stream {} }
        consumer.cancel()
        _ = await consumer.result
        await gate.release()
        await session.synchronize()
        XCTAssertThrowsError(try session.steer("late"))
        let result = try await session.respond(to: "new task")
        XCTAssertFalse(result.isEmpty)
    }

    private struct RecordedMessage: Equatable, Sendable {
        var role: Chat.Message.Role
        var content: String
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func generate(messages: [Chat.Message]) -> [Message] {
            continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    /// Produces a transcript token stream where a rendered follow-up is an
    /// exact extension of the prompt and generated tokens from the prior turn.
    private struct PrefixPreservingTokenizer: Tokenizer {
        let renderedLengthContinuation: AsyncStream<Int>.Continuation
        var rewritesPrefixOnContinuation = false
        var rewritesCachedTailOnContinuation = false

        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil
        var eosTokenId: Int? { 101 }
        var unknownTokenId: Int? { 102 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            text.unicodeScalars.map { scalar in
                if (0xE000 ..< 0xE064).contains(Int(scalar.value)) {
                    return Int(scalar.value) - 0xE000
                }
                return Int(scalar.value) % 80 + 1
            }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(
                String.UnicodeScalarView(
                    tokenIds.compactMap { UnicodeScalar(0xE000 + $0) }))
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? {
            UnicodeScalar(0xE000 + id).map(String.init)
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            var tokens: [Int] = []
            for message in messages {
                let role = message["role"] as? String
                let content = message["content"] as? String ?? ""
                switch role {
                case "system":
                    tokens.append(96)
                case "user":
                    tokens.append(97)
                case "tool":
                    tokens.append(98)
                case "assistant":
                    tokens.append(94)
                default:
                    tokens.append(99)
                }
                tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
                if role != "assistant" {
                    tokens.append(95)
                }
            }
            tokens.append(94)  // assistant generation marker
            if rewritesPrefixOnContinuation {
                tokens.insert(messages.count > 1 ? 93 : 92, at: 0)
            }
            if rewritesCachedTailOnContinuation, let marker = tokens.firstIndex(of: 94) {
                tokens[marker] = messages.count > 1 ? 93 : 92
            }

            renderedLengthContinuation.yield(tokens.count)
            return tokens
        }
    }

    private struct MediaAwareInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration = ModelConfiguration(id: "test")
        let messageGenerator = DefaultMessageGenerator()

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext)
            let image =
                input.images.isEmpty
                ? nil : LMInput.ProcessedImage(pixels: MLXArray([Float(0)]))
            return LMInput(text: .init(tokens: MLXArray(tokens)), image: image)
        }
    }

    private struct MaskedInputProcessor: UserInputProcessor {
        let tokenizer: Tokenizer
        let configuration = ModelConfiguration(id: "test")
        let messageGenerator = DefaultMessageGenerator()

        func prepare(input: UserInput) throws -> LMInput {
            let messages = messageGenerator.generate(from: input)
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext)
            let mask = MLXArray(Array(repeating: 1, count: tokens.count))
                .reshaped(1, tokens.count)
            return LMInput(text: .init(tokens: MLXArray(tokens), mask: mask))
        }
    }

    /// The synthetic text model does not consume attention masks itself.
    /// Strip the mask only after `ChatSession` has made its cache-reuse
    /// decision so the regression test can exercise that decision with a
    /// valid masked processor input.
    private final class MaskTolerantLanguageModel: Module, LanguageModel {
        let base: any LanguageModel

        init(_ base: any LanguageModel) {
            self.base = base
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(
                LMInput(tokens: input.text.tokens),
                cache: cache,
                state: state,
                prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters)
        }
    }

    /// A model that opts into append-only media cache reuse.
    ///
    /// The synthetic text model underneath cannot consume a media payload, so the
    /// split keeps the token suffix and the payload is stripped in `prepare` -- the
    /// same shape as `MaskTolerantLanguageModel`. What these tests measure is
    /// `ChatSession`'s decision, which is made before the model is called.
    private final class SplittingLanguageModel: Module, LanguageModel, PreparedInputSplitting {
        let base: any LanguageModel
        let declinesToSplit: Bool

        init(_ base: any LanguageModel, declinesToSplit: Bool = false) {
            self.base = base
            self.declinesToSplit = declinesToSplit
            super.init()
        }

        func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
            -> LMInput?
        {
            guard !declinesToSplit else { return nil }
            let ids = input.text.tokens.asArray(Int.self)
            guard prefixTokenCount > 0, prefixTokenCount < ids.count else { return nil }
            return LMInput(
                text: .init(tokens: MLXArray(Array(ids[prefixTokenCount...]))),
                image: input.image)
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(
                LMInput(tokens: input.text.tokens),
                cache: cache,
                state: state,
                prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters)
        }
    }

    /// A conformer that honors the protocol's shape but not its contract: it returns
    /// a suffix starting `extraDrop` tokens past the boundary it was handed. It never
    /// returns `nil`, so a nil-only check would let it through.
    private final class MisSplittingLanguageModel: Module, LanguageModel,
        PreparedInputSplitting
    {
        let base: any LanguageModel
        let extraDrop: Int

        init(_ base: any LanguageModel, extraDrop: Int) {
            self.base = base
            self.extraDrop = extraDrop
            super.init()
        }

        func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
            -> LMInput?
        {
            let ids = input.text.tokens.asArray(Int.self)
            let wrongStart = prefixTokenCount + extraDrop
            guard wrongStart > 0, wrongStart < ids.count else { return nil }
            return LMInput(
                text: .init(tokens: MLXArray(Array(ids[wrongStart...]))),
                image: input.image)
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(
                LMInput(tokens: input.text.tokens),
                cache: cache,
                state: state,
                prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters)
        }
    }

    /// Counts `KVCache.copy()` calls across every cache a model creates.
    private final class CopyCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        func increment() {
            lock.withLock { count += 1 }
        }

        var value: Int {
            lock.withLock { count }
        }
    }

    private final class CopyCountingCache: KVCacheSimple {
        let copies: CopyCounter

        init(copies: CopyCounter) {
            self.copies = copies
            super.init()
        }

        override func copy() -> any KVCache {
            copies.increment()
            return super.copy()
        }
    }

    /// Serves `base` over caches that count their copies.
    private final class CopyCountingLanguageModel: Module, LanguageModel {
        let base: any LanguageModel
        let copies: CopyCounter

        init(_ base: any LanguageModel, copies: CopyCounter) {
            self.base = base
            self.copies = copies
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            try base.prepare(input, cache: cache, state: state, prefill: prefill)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            base(input, cache: cache, state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters).map { _ in
                CopyCountingCache(copies: copies)
            }
        }
    }

    private struct EmptyChatTemplateTokenizer: Tokenizer {
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil
        var eosTokenId: Int? { 101 }
        var unknownTokenId: Int? { 102 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            []
        }
    }

    private struct UnexpectedDraftModelLoadError: Error {}

    private struct PreparedImageInputProcessor: UserInputProcessor {
        private let base = TestInputProcessor()

        func prepare(input: UserInput) throws -> LMInput {
            let text = try base.prepare(input: input).text
            return LMInput(
                text: text,
                image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
        }
    }

    /// Attaches media to the first prepared turn only. That turn refuses speculation and drops
    /// the draft cache; later text-only turns are what should be able to pick it back up.
    ///
    /// `@unchecked Sendable` because the counter is serialised by the lock.
    private final class FirstTurnImageInputProcessor: UserInputProcessor, @unchecked Sendable {
        private let base: TestInputProcessor
        private let lock = NSLock()
        private var prepared = 0

        init(base: TestInputProcessor) {
            self.base = base
        }

        private func consumeFirst() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            prepared += 1
            return prepared == 1
        }

        func prepare(input: UserInput) async throws -> LMInput {
            let text = try base.prepare(input: input).text
            guard consumeFirst() else { return LMInput(text: text) }
            return LMInput(
                text: text,
                image: .init(pixels: MLXArray.zeros([1, 1, 1, 1])))
        }
    }

    /// A model that hands back model state on every prefill, the way the Qwen
    /// vision families and GLM-OCR hand back an M-RoPE anchor. The state is
    /// what a session has to carry across turns and persist with its cache;
    /// this stands in for a VLM without needing image inputs.
    private final class StateProducingModel: Module, LanguageModel, KVCacheDimensionProvider {
        static let anchorKey = LMOutput.Key<MLXArray>("test.ropeDeltas")

        @ModuleInfo var inner: Gemma3TextModel

        var kvHeads: [Int] { (inner as? KVCacheDimensionProvider)?.kvHeads ?? [] }

        init(_ inner: Gemma3TextModel) {
            self.inner = inner
        }

        private func batched(_ tokens: MLXArray) -> MLXArray {
            tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            let logits = inner(batched(input.text.tokens), cache: cache.isEmpty ? nil : cache)
            // Carry a seeded anchor forward, or start one at zero — the shape
            // of every wired model's cold prefill.
            var produced = LMOutput.State()
            produced[Self.anchorKey] = state?[Self.anchorKey] ?? MLXArray([Int32(0)])
            let total = input.text.tokens.dim(-1)
            prefill.progress?(total, total)
            return .logits(LMOutput(logits: logits, state: produced))
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            LMOutput(logits: inner(batched(input.tokens), cache: cache), state: state)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try inner.newCache(parameters: parameters)
        }
    }

    /// Advances its state on every decode step, not just on prefill. A session
    /// must carry the state the model actually ended on.
    private final class DecodeStateAdvancingModel: Module, LanguageModel, KVCacheDimensionProvider {
        static let stepKey = LMOutput.Key<MLXArray>("test.decodeSteps")

        @ModuleInfo var inner: Gemma3TextModel

        var kvHeads: [Int] { (inner as? KVCacheDimensionProvider)?.kvHeads ?? [] }

        init(_ inner: Gemma3TextModel) {
            self.inner = inner
        }

        private func batched(_ tokens: MLXArray) -> MLXArray {
            tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            let logits = inner(batched(input.text.tokens), cache: cache.isEmpty ? nil : cache)
            var produced = LMOutput.State()
            produced[Self.stepKey] = state?[Self.stepKey] ?? MLXArray([Int32(0)])
            let total = input.text.tokens.dim(-1)
            prefill.progress?(total, total)
            return .logits(LMOutput(logits: logits, state: produced))
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            var advanced = LMOutput.State()
            advanced[Self.stepKey] = (state?[Self.stepKey] ?? MLXArray([Int32(0)])) + 1
            return LMOutput(logits: inner(batched(input.tokens), cache: cache), state: advanced)
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try inner.newCache(parameters: parameters)
        }
    }

    private static func makeDecodeStateAdvancingModel() -> ModelContext {
        let base = makeModel()
        guard let inner = base.model as? Gemma3TextModel else {
            fatalError("expected the test model to be a Gemma3TextModel")
        }
        return .init(
            configuration: base.configuration,
            model: DecodeStateAdvancingModel(inner),
            processor: base.processor,
            tokenizer: base.tokenizer)
    }

    private static func makeStateProducingModel() -> ModelContext {
        let base = makeModel()
        guard let inner = base.model as? Gemma3TextModel else {
            fatalError("expected the test model to be a Gemma3TextModel")
        }
        return .init(
            configuration: base.configuration,
            model: StateProducingModel(inner),
            processor: base.processor,
            tokenizer: base.tokenizer)
    }

    private actor DraftModelLoadCounter {
        private var count = 0

        func increment() {
            count += 1
        }

        var value: Int {
            count
        }
    }

    private static func makeModel(
        processor: any UserInputProcessor,
        configuration: ModelConfiguration,
        tokenizer: any Tokenizer
    )
        -> ModelContext
    {
        let config = Gemma3TextConfiguration.smallConfiguration(
            vocabularySize: 100)
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        return .init(
            configuration: configuration,
            model: model,
            processor: processor,
            tokenizer: tokenizer)
    }

    private static func makeModel(processor: TestInputProcessor = TestInputProcessor())
        -> ModelContext
    {
        makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    private func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        Self.makeModel(processor: processor)
    }

    private func model(processor: MediaAwareInputProcessor) -> ModelContext {
        Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
    }

    private func model(
        processor: MediaAwareInputProcessor, splitting: Bool, declinesToSplit: Bool = false
    ) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        if splitting {
            context.model = SplittingLanguageModel(
                context.model, declinesToSplit: declinesToSplit)
        }
        return context
    }

    private func model(processor: MaskedInputProcessor) -> ModelContext {
        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        context.model = MaskTolerantLanguageModel(context.model)
        return context
    }

    private func collectGeneration(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var completionInfo: GenerateCompletionInfo?
        for try await item in stream {
            if let chunk = item.chunk {
                text += chunk
            }
            if let info = item.info {
                completionInfo = info
            }
        }
        return (text, try XCTUnwrap(completionInfo))
    }

    /// The values a realized cache holds, read out so they can leave the session.
    private static func cachedValues(_ cache: [KVCache]?) -> [[Float]] {
        (cache ?? []).flatMap(\.state).map { $0.asType(.float32).asArray(Float.self) }
    }

    private let generationParameters = GenerateParameters(maxTokens: 50)

    private let targetLength = 1

    func testExistingSessionAPIAndExhaustiveEventSwitchesRemainCompatible() async throws {
        func classify(_ error: ChatSessionError) -> Int {
            switch error {
            case .noCacheAvailable: 0
            case .emptyPreparedInput: 1
            case .kvCacheConfigurationChanged: 2
            }
        }
        func classify(_ event: Generation) -> Int {
            switch event {
            case .chunk: 0
            case .toolCall: 1
            case .rejectedToolCall: 2
            case .info: 3
            case .reasoning: 4
            case .steering: 5
            }
        }
        func classify(_ event: TokenGeneration) -> Int {
            switch event {
            case .token: 0
            case .info: 1
            }
        }
        XCTAssertEqual(classify(ChatSessionError.noCacheAvailable), 0)
        XCTAssertEqual(classify(ChatSessionError.emptyPreparedInput), 1)
        XCTAssertEqual(
            classify(ChatSessionError.kvCacheConfigurationChanged(previous: nil, requested: nil)), 2
        )
        XCTAssertEqual(classify(Generation.chunk("text")), 0)
        XCTAssertEqual(classify(Generation.reasoning("thought")), 4)
        XCTAssertEqual(classify(TokenGeneration.token(1)), 0)

        let session = ChatSession(
            model(), generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            components: GenerationComponents { ForceTokenProcessor(token: 1) })
        let promptDetails:
            (String, Chat.Message.Role, [UserInput.Image], [UserInput.Video], [UserInput.Audio]) ->
                AsyncThrowingStream<Generation, Error> = session.streamDetails
        let messageDetails: ([Chat.Message]) -> AsyncThrowingStream<Generation, Error> = session
            .streamDetails
        let prompt = try await collectGeneration(promptDetails("first", .user, [], [], []))
        let messages = try await collectGeneration(messageDetails([.user("second")]))
        XCTAssertEqual(prompt.info.generationTokenCount, 1)
        XCTAssertEqual(messages.info.generationTokenCount, 1)
    }

    private func collectObservedGeneration(
        _ stream: AsyncThrowingStream<LogProbabilityGeneration<Generation>, Error>
    ) async throws -> (
        text: String, info: GenerateCompletionInfo, probabilities: [GenerateTokenLogProbabilities]
    ) {
        var text = ""
        var info: GenerateCompletionInfo?
        var probabilities: [GenerateTokenLogProbabilities] = []
        for try await event in stream {
            XCTAssertNotEqual(event.generation != nil, event.logProbabilities != nil)
            if let values = event.logProbabilities { probabilities.append(values) }
            if let generation = event.generation {
                if let chunk = generation.chunk { text += chunk }
                if let completion = generation.info { info = completion }
            }
        }
        return (text, try XCTUnwrap(info), probabilities)
    }

    func testTokenObservationPreservesCachedContinuationAndIsScopedToTheCall() async throws {
        let (_, continuation) = AsyncStream<Int>.makeStream()
        defer { continuation.finish() }
        let processor = TestInputProcessor(
            tokenizer: PrefixPreservingTokenizer(renderedLengthContinuation: continuation),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let container = ModelContainer(context: model(processor: processor))
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let components = GenerationComponents { ForceTokenProcessor(token: 1) }
        let plain = ChatSession(container, generateParameters: parameters, components: components)
        let observed = ChatSession(
            container, generateParameters: parameters, components: components)
        for prompt in ["first", "second"] {
            let expected = try await collectGeneration(plain.streamDetails(to: prompt))
            let actual = try await collectObservedGeneration(
                observed.observe(
                    to: prompt, using: .logProbabilities(topCount: 4)))
            XCTAssertEqual(actual.text, expected.text)
            XCTAssertEqual(actual.info.generationTokenCount, expected.info.generationTokenCount)
            XCTAssertEqual(actual.info.cachedPromptTokenCount, expected.info.cachedPromptTokenCount)
            if prompt == "second" { XCTAssertGreaterThan(actual.info.cachedPromptTokenCount, 0) }
            XCTAssertEqual(actual.probabilities.count, 3)
            XCTAssertTrue(
                actual.probabilities.allSatisfy {
                    $0.chosen.token == 1 && $0.chosen.logProbability == 0
                        && $0.topLogProbabilities.count == 4
                })
            XCTAssertNil(observed.generateParameters.logProbabilities)
        }
        let ordinary = try await collectGeneration(
            observed.streamDetails(to: "without observation"))
        XCTAssertEqual(ordinary.info.generationTokenCount, 3)
    }

    func testTokenObservationOmitsStopAndLookaheadForBothEntryPoints() async throws {
        for limit in [1, 3, 4, 10] {
            let session = rejectionSession(output: "abc")
            session.generateParameters.maxTokens = limit
            let stream =
                limit.isMultiple(of: 2)
                ? session.observe(
                    to: [.user("test")],
                    using: .logProbabilities(topCount: 0))
                : session.observe(
                    to: "test", using: .logProbabilities(topCount: 0))
            let result = try await collectObservedGeneration(stream)
            XCTAssertEqual(result.text, String("abc".prefix(min(limit, 3))))
            XCTAssertEqual(result.probabilities.count, result.info.generationTokenCount)
            XCTAssertEqual(result.probabilities.count, min(limit, 3))
            XCTAssertTrue(
                result.probabilities.allSatisfy {
                    $0.topLogProbabilities.isEmpty && $0.chosen.logProbability == 0
                })
            XCTAssertFalse(result.probabilities.contains { $0.chosen.token == 99 })
        }
    }

    func testTokenObservationOverridesReportingOnlyForThisCall() async throws {
        let session = rejectionSession(output: "abc")
        session.generateParameters.logProbabilities = 3
        let result = try await collectObservedGeneration(
            session.observe(
                to: "test", using: .logProbabilities(topCount: 0)))
        XCTAssertEqual(result.text, "abc")
        XCTAssertEqual(result.probabilities.count, 3)
        XCTAssertTrue(result.probabilities.allSatisfy { $0.topLogProbabilities.isEmpty })
        XCTAssertEqual(session.generateParameters.logProbabilities, 3)
    }

    func testTextTransformerPreservesReportingForBothEntryPoints() async throws {
        let session = rejectionSession(output: "abc")
        session.generateParameters.logProbabilities = 3
        let validations = CallCounter()
        session.components.parameterValidator = { parameters in
            XCTAssertEqual(parameters.logProbabilities, 3)
            validations.increment()
        }
        for structured in [false, true] {
            let stream =
                structured
                ? session.observe(
                    to: [.user("test")], using: .text)
                : session.observe(to: "test", using: .text)
            let result = try await collectGeneration(stream)
            XCTAssertEqual(result.text, "abc")
            XCTAssertEqual(result.info.generationTokenCount, 3)
            XCTAssertEqual(session.generateParameters.logProbabilities, 3)
        }
        XCTAssertEqual(validations.value, 2)
        session.components.parameterValidator = nil
        let observed = try await collectObservedGeneration(
            session.observe(to: "test", using: .logProbabilities()))
        XCTAssertEqual(observed.probabilities.count, 3)
    }

    func testTransformersAreCapturedForEachRequest() async throws {
        let session = rejectionSession(output: "abc")
        var transformer = LogProbabilityGenerationTransformer()
        let stream = session.observe(to: "first", using: transformer)
        transformer.topCount = 2
        let first = try await collectObservedGeneration(stream)
        let second = try await collectObservedGeneration(
            session.observe(to: "second", using: transformer))
        XCTAssertEqual(first.probabilities.count, 3)
        XCTAssertEqual(second.probabilities.count, 3)
        XCTAssertTrue(first.probabilities.allSatisfy { $0.topLogProbabilities.isEmpty })
        XCTAssertTrue(second.probabilities.allSatisfy { $0.topLogProbabilities.count == 2 })
        XCTAssertNil(session.generateParameters.logProbabilities)
    }

    func testOrdinarySpeculationDoesNotRequireUnusedLogProbabilities() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()), numDraftTokens: 2),
            generateParameters: GenerateParameters(
                maxTokens: 3, temperature: 0, logProbabilities: 2),
            components: GenerationComponents { ForceTokenProcessor(token: 1) })
        for prompt in ["first", "second"] {
            let result = try await collectGeneration(session.streamDetails(to: prompt))
            XCTAssertEqual(result.info.generationTokenCount, 3)
        }
        XCTAssertEqual(session.generateParameters.logProbabilities, 2)
    }

    func testTextTransformerAllowsActiveSpeculation() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()), numDraftTokens: 2),
            generateParameters: GenerateParameters(maxTokens: 3, temperature: 0),
            components: GenerationComponents { ForceTokenProcessor(token: 1) })
        _ = try await collectGeneration(session.streamDetails(to: "warm"))
        let result = try await collectGeneration(
            session.observe(to: "test", using: .text))
        XCTAssertEqual(result.info.generationTokenCount, 3)
    }

    func testCustomTransformerConfiguresOnlyItsRequestAndPreservesCustomOutput() async throws {
        let session = rejectionSession(output: "abc")
        session.generateParameters.maxTokens = 10
        let factories = TransformerFactoryCount()
        for structured in [false, true] {
            let transformer = TokenCountingTransformer(factories: factories, maxTokens: 2)
            let stream =
                structured
                ? session.observe(to: [.user("test")], using: transformer)
                : session.observe(to: "test", using: transformer)
            var indices: [Int] = []
            var text = ""
            var completion: GenerateCompletionInfo?
            for try await event in stream {
                switch event {
                case .token(let index): indices.append(index)
                case .output(let generation, let count):
                    if let chunk = generation.chunk { text += chunk }
                    if let info = generation.info {
                        completion = info
                        XCTAssertEqual(count, 2)
                    }
                }
            }
            XCTAssertEqual(indices, [1, 2])
            XCTAssertEqual(text, "ab")
            XCTAssertEqual(completion?.generationTokenCount, 2)
            XCTAssertEqual(session.generateParameters.maxTokens, 10)
        }
        XCTAssertEqual(factories.value, 2)
        let ordinary = try await collectGeneration(session.streamDetails(to: "ordinary"))
        XCTAssertEqual(ordinary.text, "abc")
    }

    func testCustomTransformerWorksWithActiveSpeculation() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()), numDraftTokens: 2),
            generateParameters: GenerateParameters(maxTokens: 3, temperature: 0),
            components: GenerationComponents { ForceTokenProcessor(token: 1) })
        let factories = TransformerFactoryCount()
        for prompt in ["first", "second"] {
            var indices: [Int] = []
            var completion: GenerateCompletionInfo?
            for try await event in session.observe(
                to: prompt, using: TokenCountingTransformer(factories: factories))
            {
                if case .token(let index) = event { indices.append(index) }
                if let info = event.generation?.info { completion = info }
            }
            XCTAssertEqual(indices, [1, 2, 3])
            XCTAssertEqual(completion?.generationTokenCount, 3)
        }
        XCTAssertEqual(factories.value, 2)
    }

    @MainActor
    func testTokenObservationStopsAfterCancellationAndReleasesTheCache() async throws {
        let session = ChatSession(
            model(), generateParameters: GenerateParameters(maxTokens: 1000, temperature: 0),
            components: GenerationComponents { ForceTokenProcessor(token: 1) })
        let (observations, continuation) = AsyncStream<Void>.makeStream()
        var count = 0
        let task = Task {
            for try await event in session.observe(
                to: "test", using: .logProbabilities(topCount: 1))
            {
                if event.logProbabilities != nil {
                    count += 1
                    continuation.yield(())
                }
            }
        }
        var iterator = observations.makeAsyncIterator()
        let first: Void? = await iterator.next()
        XCTAssertNotNil(first)
        task.cancel()
        _ = await task.result
        continuation.finish()
        session.generateParameters.maxTokens = 1
        var completion: GenerateCompletionInfo?
        for try await event in session.streamDetails(to: "again") {
            if let info = event.info { completion = info }
        }
        XCTAssertEqual(completion?.generationTokenCount, 1)
        XCTAssertGreaterThan(count, 0)
        XCTAssertLessThan(count, 1000)
    }

    func testTokenObservationRejectsActiveSpeculationButAllowsMemoryFallback() async throws {
        for fallback in [false, true] {
            let session = ChatSession(
                model(),
                speculativeDecoding: SpeculativeDecodingConfig(
                    draftModel: ModelContainer(context: model()), numDraftTokens: 2,
                    memoryPolicy: fallback
                        ? SpeculativeDecodingMemoryPolicy(limitBytes: 0, action: .fallbackToDefault)
                        : nil),
                generateParameters: GenerateParameters(maxTokens: 3, temperature: 0),
                components: GenerationComponents { ForceTokenProcessor(token: 1) })
            let warm = try await collectGeneration(session.streamDetails(to: "warm"))
            XCTAssertEqual(warm.info.generationTokenCount, 3)
            do {
                let result = try await collectObservedGeneration(
                    session.observe(
                        to: "test", using: .logProbabilities(topCount: 1)))
                XCTAssertTrue(fallback)
                XCTAssertEqual(result.probabilities.count, result.info.generationTokenCount)
                XCTAssertEqual(result.probabilities.count, 3)
            } catch ChatSession.ObservationError.unsupportedSpeculativeDecoding {
                XCTAssertFalse(fallback)
            }
            let resumed = try await collectGeneration(session.streamDetails(to: "resume"))
            XCTAssertEqual(resumed.info.generationTokenCount, 3)
        }
    }

    func testTokenObservationPreservesRejectionAndPreventsPartialToolDispatch() async throws {
        for automatic in [false, true] {
            let dispatched = CallCounter()
            let dispatch: @Sendable (ToolCall) async throws -> String = { _ in
                dispatched.increment()
                return "unexpected"
            }
            let session = rejectionSession(
                output: #"<tool_call>{"name":"allowed","arguments":{}}</tool_call><tool_call>{"#,
                tools: [["function": ["name": "allowed"] as [String: any Sendable]]],
                toolDispatch: automatic ? dispatch : nil)
            var rejection: RejectedToolCall?
            var completion: GenerateCompletionInfo?
            var tokenCount = 0
            do {
                for try await event in session.observe(
                    to: "test", using: .logProbabilities(topCount: 0))
                {
                    if event.logProbabilities != nil { tokenCount += 1 }
                    if let generation = event.generation {
                        if let value = generation.rejectedToolCall { rejection = value }
                        if let value = generation.info { completion = value }
                    }
                }
                XCTAssertFalse(automatic)
            } catch let error as RejectedToolCallError {
                XCTAssertTrue(automatic)
                XCTAssertEqual(error.rejection.reason, .incompleteOutput)
            }
            XCTAssertEqual(rejection?.reason, .incompleteOutput)
            XCTAssertEqual(completion?.rejectedToolCallCount, 1)
            XCTAssertEqual(tokenCount, completion?.generationTokenCount)
            XCTAssertEqual(dispatched.value, 0)
        }
    }

    func testChatSessionSync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        let result1 = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result1.count, targetLength, result1)
        let result2 = try await session.respond(to: "hello again")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        var result1 = ""
        for try await part in session.streamResponse(to: "hello") {
            result1 += part
        }
        XCTAssertGreaterThan(result1.count, targetLength, result1)

        var result2 = ""
        for try await part in session.streamResponse(to: "hello again") {
            result2 += part
        }
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionRespondToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        let result = try await session.respond(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ])
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChatSessionStreamResponseToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        var result = ""
        for try await part in session.streamResponse(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ]) {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChangingInstructionsUpdatesRetainedConversation() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            instructions: "first instruction",
            generateParameters: GenerateParameters(maxTokens: 1))

        _ = try await session.respond(to: "first question")
        session.instructions = "replacement instruction"
        _ = try await session.respond(to: "second question")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.system, .user])
        XCTAssertEqual(calls[0].first?.content, "first instruction")
        XCTAssertEqual(calls[1].map(\.role), [.system, .user, .assistant, .user])
        XCTAssertEqual(calls[1].first?.content, "replacement instruction")
        XCTAssertFalse(calls[1].contains { $0.content == "first instruction" })
    }

    func testEmptyGenerationRollsBackIncompleteTurn() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 0))

        let firstResponse = try await session.respond(to: "first question")
        let secondResponse = try await session.respond(to: "second question")
        XCTAssertEqual(firstResponse, "")
        XCTAssertEqual(secondResponse, "")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.user])
        XCTAssertEqual(calls[1].map(\.role), [.user])
        XCTAssertEqual(calls[1].first?.content, "second question")
    }

    /// A turn that produces only reasoning is still a turn.
    ///
    /// Reasoning is deliberately excluded from the recorded assistant content, so a model
    /// that runs out of `maxTokens` inside its thought block yields an empty `content`.
    /// Without `producedReasoning` that is indistinguishable from a cancelled or empty
    /// generation, and the rollback in `respond` would take the USER's message out of the
    /// transcript with it - the caller's prompt would simply vanish. Contrast
    /// `testEmptyGenerationRollsBackIncompleteTurn`, where rolling back IS correct.
    func testReasoningOnlyGenerationStillRecordsTheTurn() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: ReasoningOnlyTokenizer(),
            configuration: ModelConfiguration(
                id: "test", reasoningConfig: .alwaysOnThinking),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            Self.makeModel(
                processor: processor,
                configuration: processor.configuration,
                tokenizer: processor.tokenizer),
            generateParameters: GenerateParameters(maxTokens: 4))

        // The thought block never closes, so nothing reaches `.chunk`.
        let response = try await session.respond(to: "first question")
        XCTAssertEqual(response, "")

        _ = try await session.respond(to: "second question")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.user])
        // The user turn survived and an assistant turn was recorded for it, so the
        // second request renders user/assistant/user rather than a bare user.
        XCTAssertEqual(calls[1].map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(calls[1].first?.content, "first question")
        XCTAssertEqual(calls[1].last?.content, "second question")
        // The other half of the contract: the turn is recorded, but the reasoning is
        // NOT replayed as assistant content. Without this the test would also pass for
        // a "fix" that accumulated reasoning into `content`, which is the behavior this
        // change deliberately removes.
        XCTAssertEqual(calls[1][1].content, "")
    }

    func testCancelledGenerationRollsBackIncompleteTurn() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 50))

        for try await _ in session.streamResponse(to: "first question") {
            break
        }
        await session.synchronize()

        session.generateParameters = GenerateParameters(maxTokens: 1)
        _ = try await session.respond(to: "second question")
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].map(\.role), [.user])
        XCTAssertEqual(calls[1].map(\.role), [.user])
        XCTAssertEqual(calls[1].first?.content, "second question")
    }

    func testEmptyPreparedInputThrowsClearError() async throws {
        let tokenizer = EmptyChatTemplateTokenizer()
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 1))

        do {
            _ = try await session.respond(to: "ignored")
            XCTFail("expected ChatSessionError.emptyPreparedInput")
        } catch ChatSessionError.emptyPreparedInput {
            // expected
        }
    }

    /// something that looks like a view model
    @MainActor class ChatModel {
        let session: ChatSession

        public var messages = [Chat.Message]()

        private var task: Task<Void, Error>?
        public var isBusy: Bool {
            task != nil
        }

        init(model: ModelContext) {
            self.session = ChatSession(model)
        }

        public func cancel() {
            task?.cancel()
        }

        public func respond(_ message: String) {
            guard task == nil else { return }

            self.messages.append(.init(role: .user, content: message))
            self.messages.append(.init(role: .assistant, content: "..."))
            let lastIndex = self.messages.count - 1

            self.task = Task {
                var first = true
                for try await item in session.streamResponse(to: message) {
                    if first {
                        self.messages[lastIndex].content = item
                        first = false
                    } else {
                        self.messages[lastIndex].content += item
                    }
                }
                self.task = nil
            }
        }
    }

    @MainActor
    func testViewModel() async throws {
        let model = ChatModel(model: model())

        // start producing a response but interrupt it
        // triggers https://github.com/ml-explore/mlx-swift/pull/323
        model.respond("message1")
        try await Task.sleep(for: .milliseconds(50))
        model.cancel()

        // wait for it to finish
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }

        // try another message, wait for full completion (but cap the length)
        model.session.generateParameters = self.generationParameters
        model.respond("message2")
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}
