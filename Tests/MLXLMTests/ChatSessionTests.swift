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

    func testInterruptedGenerationRollsBackIncompleteTurn() async throws {
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
