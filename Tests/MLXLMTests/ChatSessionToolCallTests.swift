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

final class ChatSessionToolCallTests: XCTestCase, ChatSessionFixtures {
    func testStreamDetailsEmitsRejectedToolCallAndCompletionCount() async throws {
        let session = rejectionSession(output: #"<tool_call>{"#)
        var rejection: RejectedToolCall?
        var info: GenerateCompletionInfo?
        var chunks: [String] = []

        for try await event in session.streamDetails(to: "trigger") {
            if let value = event.rejectedToolCall { rejection = value }
            if let value = event.info { info = value }
            if let value = event.chunk { chunks.append(value) }
        }

        XCTAssertEqual(rejection?.reason, .incompleteOutput)
        XCTAssertEqual(info?.rejectedToolCallCount, 1)
        XCTAssertTrue(chunks.isEmpty)
    }

    func testStreamDetailsReportsRecoveredToolCallCount() async throws {
        let tools: [ToolSpec] = [
            ["function": ["name": "weather"] as [String: any Sendable]]
        ]
        let session = rejectionSession(
            output: "<function=weather><parameter=city>Paris</parameter></function>",
            tools: tools)
        var call: ToolCall?
        var info: GenerateCompletionInfo?

        for try await event in session.streamDetails(to: "trigger") {
            if let value = event.toolCall { call = value }
            if let value = event.info { info = value }
        }

        XCTAssertEqual(call?.function.name, "weather")
        XCTAssertEqual(info?.recoveredToolCallCount, 1)
        XCTAssertEqual(info?.rejectedToolCallCount, 0)
    }

    func testTextStreamThrowsRejectedToolCallError() async throws {
        let session = rejectionSession(output: #"<tool_call>{"#)

        do {
            for try await _ in session.streamResponse(to: "trigger") {}
            XCTFail("Expected RejectedToolCallError")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.reason, .incompleteOutput)
        }
    }

    func testRejectedGenerationIsNotCommittedToConversation() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let session = rejectionSession(
            output: #"<tool_call>{"#,
            messageGenerator: RecordingMessageGenerator(continuation: continuation))

        for prompt in ["first", "second"] {
            do {
                for try await _ in session.streamResponse(to: prompt) {}
                XCTFail("Expected RejectedToolCallError")
            } catch is RejectedToolCallError {
                // Expected. The next request must start from a clean transcript.
            }
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages { calls.append(call) }
        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0], [.init(role: .user, content: "first")])
        XCTAssertEqual(calls[1], [.init(role: .user, content: "second")])
    }

    func testRejectedGenerationPreventsPartialToolDispatch() async throws {
        let dispatchCount = CallCounter()
        let output =
            #"<tool_call>{"name":"allowed","arguments":{}}</tool_call><tool_call>{"#
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": ["name": "allowed"] as [String: any Sendable],
            ]
        ]
        let session = rejectionSession(
            output: output,
            tools: tools,
            toolDispatch: { _ in
                dispatchCount.increment()
                return "should not execute"
            })

        do {
            for try await _ in session.streamDetails(to: "trigger") {}
            XCTFail("Expected RejectedToolCallError")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.reason, .incompleteOutput)
        }
        XCTAssertEqual(dispatchCount.value, 0)
    }

    func testToolDispatchWithoutSchemasFailsClosed() async throws {
        let dispatchCount = CallCounter()
        let session = rejectionSession(
            output: #"<tool_call>{"name":"undeclared","arguments":{}}</tool_call>"#,
            toolDispatch: { _ in
                dispatchCount.increment()
                return "should not execute"
            })

        do {
            for try await _ in session.streamDetails(to: "trigger") {}
            XCTFail("Expected RejectedToolCallError")
        } catch let error as RejectedToolCallError {
            XCTAssertEqual(error.rejection.reason, .undeclaredTool)
        }
        XCTAssertEqual(dispatchCount.value, 0)
    }

    /// A custom ``LogitProcessor`` injected via ``GenerationComponents`` must ride
    /// the real ``ChatSession`` generation path -- the reason this API exists.
    func testChatSessionUsesCustomLogitProcessor() async throws {
        let forcedToken = 7
        let inputProcessor = TestInputProcessor()
        let model = model(processor: inputProcessor)

        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var components = GenerationComponents()
        components.logitProcessorFactory = {
            ForceTokenProcessor(token: Int32(forcedToken))
        }

        let session = ChatSession(model, generateParameters: parameters, components: components)
        let result = try await session.respond(to: "hello")

        // with all other logits masked, every generated token must be the forced token
        let expectedWord = try XCTUnwrap(inputProcessor.tokenizer.convertIdToToken(forcedToken))
        let words = result.split(separator: " ").map(String.init)
        XCTAssertFalse(words.isEmpty, result)
        XCTAssertTrue(words.allSatisfy { $0 == expectedWord }, result)
    }

    /// A single ``ChatSession`` reuses one ``GenerationComponents`` across turns.
    /// The factory MUST be invoked fresh for each generation so a stateful
    /// processor never leaks state between turns.
    func testChatSessionInvokesLogitProcessorFactoryPerGeneration() async throws {
        let counter = CallCounter()
        let model = model()

        let parameters = GenerateParameters(maxTokens: 5, temperature: 0)
        var components = GenerationComponents()
        components.logitProcessorFactory = {
            counter.increment()
            return ForceTokenProcessor(token: 7)
        }

        let session = ChatSession(model, generateParameters: parameters, components: components)
        _ = try await session.respond(to: "hello")
        _ = try await session.respond(to: "hello again")

        // two turns -> two fresh processor instances
        XCTAssertEqual(counter.value, 2)
    }

    /// Parameter-dependent components must fail through the real session API
    /// before prompt prefill starts.
    func testChatSessionRunsGenerationComponentValidation() async throws {
        let inputProcessor = TestInputProcessor()
        let components = try GenerationComponents().applyingThinkingBudget(
            ThinkingBudgetConfiguration(
                maximumTokenCount: 100,
                transitionOverride: .immediate),
            reasoning: .alwaysOnThinking,
            tokenizer: inputProcessor.tokenizer)
        let session = ChatSession(
            model(processor: inputProcessor),
            generateParameters: GenerateParameters(maxTokens: 1),
            components: components)

        do {
            _ = try await session.respond(to: "hello")
            XCTFail("Expected generation-component validation to reject maxTokens")
        } catch let error as ThinkingBudgetError {
            guard case .insufficientGenerationTokenLimit = error else {
                XCTFail("Unexpected thinking-budget error: \(error)")
                return
            }
        }
    }

    /// Passing an empty ``GenerationComponents()`` must be non-breaking: the
    /// session still generates normally, exactly as when no components are
    /// supplied. (The exact processor-equivalence guarantee is proven
    /// deterministically by `testEmptyGenerationComponentsMatchesParametersProcessor`;
    /// full token-sequence equality is not asserted here because argmax over
    /// randomly-initialized weights is not bitwise-reproducible on GPU.)
    func testChatSessionEmptyComponentsMatchesDefault() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 50),
            components: GenerationComponents())
        let result = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChatSessionAsyncInterrupt() async throws {
        // interrupt the streamResponse and continue with another request
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        for _ in 0 ..< 10 {
            var result1 = ""
            for try await part in session.streamResponse(to: "hello") {
                result1 += part
                break
            }

            // at this point the performStreaming/generate code may still be running.
            // the next call can corrupt the state if not thread safe

            var result2 = ""
            for try await part in session.streamResponse(to: "hello again") {
                result2 += part
                if result2.count > 100 {
                    break
                }
            }
        }

        // since we are interrupting we need to wait for everything to finish
        // (avoids shutdown issues if this is the last/only test). because the
        // streaming task is not a synchronous shutdown
        await session.synchronize()
    }

    func testChatSessionWithTools() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        let result = try await session.respond(to: "What is the weather in SF?")
        XCTAssertGreaterThan(result.count, targetLength, result)

        // second turn to verify tools persist through cache
        let result2 = try await session.respond(to: "How about NYC?")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionWithToolsStreaming() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        var result = ""
        for try await part in session.streamResponse(to: "hello") {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }
}
