// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLMCommon

/// The Qwen history of reasoning: ``Qwen3VLMessageGenerator`` moves a completed
/// think block at the start of an imported assistant turn to
/// `reasoning_content`, and ``ChatSession`` records the reasoning of a turn it
/// generated apart from the content, so that every message generator writes it
/// as `reasoning_content`.
final class QwenReasoningHistoryTests: XCTestCase {
    private func text(in message: MLXLMCommon.Message) throws -> String {
        let parts = try XCTUnwrap(message["content"] as? [[String: String]])
        return parts.filter { $0["type"] == "text" }.compactMap { $0["text"] }.joined()
    }

    func testCompletedReasoningIsSeparatedWithoutTrimming() throws {
        let raw = Qwen3VLMessageGenerator().generate(
            message: .assistant("<think>  thought\n</think>\n\n answer "))
        XCTAssertEqual(raw["reasoning_content"] as? String, "  thought\n")
        XCTAssertEqual(try text(in: raw), "\n\n answer ")
    }

    func testEmptyCompletedReasoningIsPresent() throws {
        let raw = Qwen3VLMessageGenerator().generate(message: .assistant("<think></think>answer"))
        XCTAssertEqual(raw["reasoning_content"] as? String, "")
        XCTAssertEqual(try text(in: raw), "answer")
    }

    func testRecordedReasoningIsKeptAndTheContentIsNotParsedAgain() throws {
        let raw = Qwen3VLMessageGenerator().generate(
            message: .assistant("<think>literal</think>answer", reasoning: "recorded"))
        XCTAssertEqual(raw["reasoning_content"] as? String, "recorded")
        XCTAssertEqual(try text(in: raw), "<think>literal</think>answer")
    }

    func testUnmarkedLiteralAndIncompleteContentRemainUnchanged() throws {
        let messages: [Chat.Message] = [
            .assistant("The closing tag is </think>, followed by an answer."),
            .assistant("An example: <think>thought</think>answer"),
            .assistant(" <think>thought</think>answer"),
            .assistant("```xml\n<think>thought</think>\n```"),
            .assistant("thought</think>answer"),
            .assistant("<think>unfinished"),
        ]
        for message in messages {
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertNil(raw["reasoning_content"], message.content)
            XCTAssertEqual(try text(in: raw), message.content)
        }
    }

    func testImplicitToolBoundaryDoesNotInventAReasoningClose() throws {
        let call = ToolCall(
            function: .init(name: "read_file", arguments: ["path": .string("README.md")]))
        let message = Chat.Message.assistant("<think>unfinished thought", toolCalls: [call])
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        XCTAssertNil(raw["reasoning_content"])
        XCTAssertEqual(try text(in: raw), message.content)
        let calls = try XCTUnwrap(raw["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
    }

    func testOnlyAssistantMessagesAreNormalized() throws {
        for role in [Chat.Message.Role.user, .system, .tool] {
            let message = Chat.Message(role: role, content: "<think>thought</think>answer")
            let raw = Qwen3VLMessageGenerator().generate(message: message)
            XCTAssertEqual(raw["role"] as? String, role.rawValue)
            XCTAssertNil(raw["reasoning_content"])
            XCTAssertEqual(try text(in: raw), message.content)
        }
    }

    func testNormalizationPreservesMediaAndToolMetadata() throws {
        let call = ToolCall(
            function: .init(
                name: "read_file", arguments: ["path": .string("README.md"), "offset": .int(1)]),
            id: "call_1")
        let message = Chat.Message.assistant(
            "<think>thought</think>answer",
            images: [.url(URL(fileURLWithPath: "/tmp/reasoning-image.png"))],
            videos: [.url(URL(fileURLWithPath: "/tmp/reasoning-video.mp4"))],
            toolCalls: [call])
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        let parts = try XCTUnwrap(raw["content"] as? [[String: String]])
        XCTAssertEqual(parts.map { $0["type"] }, ["image", "video", "text"])
        XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
        XCTAssertEqual(try text(in: raw), "answer")
        let calls = try XCTUnwrap(raw["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0]["id"] as? String, "call_1")
        let function = try XCTUnwrap(calls[0]["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "read_file")
        let arguments = try XCTUnwrap(function["arguments"] as? [String: any Sendable])
        XCTAssertEqual(arguments["path"] as? String, "README.md")
        XCTAssertEqual(arguments["offset"] as? Int, 1)

        let tool = Chat.Message.tool("<think>literal</think>", id: "call_1", name: "read_file")
        let rawTool = Qwen3VLMessageGenerator().generate(message: tool)
        XCTAssertEqual(rawTool["tool_call_id"] as? String, "call_1")
        XCTAssertEqual(rawTool["name"] as? String, "read_file")
        XCTAssertEqual(try text(in: rawTool), tool.content)
        XCTAssertNil(rawTool["reasoning_content"])
    }

    func testOtherMessageGeneratorsKeepTheirExistingContent() throws {
        let message = Chat.Message.assistant("<think>thought</think>answer")
        let standard = DefaultMessageGenerator().generate(message: message)
        XCTAssertEqual(standard["content"] as? String, message.content)
        XCTAssertNil(standard["reasoning_content"])
        let qwen2 = Qwen2VLMessageGenerator().generate(message: message)
        XCTAssertEqual(try text(in: qwen2), message.content)
        XCTAssertNil(qwen2["reasoning_content"])
    }

    func testSessionRecordsPromptPrimedReasoningAndLeavesStreamedTextUnchanged() async throws {
        let output = "thought</think>\n\nanswer"
        for whitespace in [
            "\n \t", String(repeating: " ", count: 61), String(repeating: " ", count: 130),
        ] {
            let retained = try await retainedAssistant(
                promptTail: "<think>" + whitespace, output: output)
            XCTAssertEqual(
                retained.reasoning, "thought", "Trailing whitespace length: \(whitespace.count)")
            XCTAssertEqual(retained.content, "answer")
            let raw = Qwen3VLMessageGenerator().generate(message: retained)
            XCTAssertEqual(raw["reasoning_content"] as? String, "thought")
            XCTAssertEqual(try text(in: raw), "answer")
        }
    }

    func testSessionDoesNotInferPrimingFromAClosedBlock() async throws {
        for tail in ["", "<think>\n\n</think>\n\n"] {
            let retained = try await retainedAssistant(promptTail: tail, output: "literal</think>")
            XCTAssertNil(retained.reasoning, tail)
            XCTAssertEqual(retained.content, "literal</think>")
            let raw = Qwen3VLMessageGenerator().generate(message: retained)
            XCTAssertNil(raw["reasoning_content"])
            XCTAssertEqual(try text(in: raw), retained.content)
        }
    }

    func testSessionRequiresTheConfiguredDelimiter() async throws {
        let unconfigured = try await retainedAssistant(
            promptTail: "<think>\n", output: "thought</think>", reasoning: nil)
        XCTAssertNil(unconfigured.reasoning)
        XCTAssertEqual(unconfigured.content, "thought</think>")

        var custom = ReasoningConfig(
            startDelimiter: "<reason>", endDelimiter: "</reason>", promptStrategy: .alwaysOn)
        custom.replaysReasoningIntoHistory = true
        let mismatched = try await retainedAssistant(
            promptTail: "<think>\n", output: "thought</think>", reasoning: custom)
        XCTAssertNil(mismatched.reasoning)
        XCTAssertEqual(mismatched.content, "thought</think>")

        let matching = try await retainedAssistant(
            promptTail: "<reason>\n", output: "thought</reason>answer", reasoning: custom)
        XCTAssertEqual(matching.reasoning, "thought")
        XCTAssertEqual(matching.content, "answer")
    }

    /// A turn cut inside the block its prompt opened keeps its whole text as
    /// the reasoning, so that the history render extends the cached tokens.
    func testLengthLimitedOpenReasoningIsRecordedAsReasoning() async throws {
        let retained = try await retainedAssistant(
            promptTail: "<think>\n", output: "unfinished", maxTokens: 4)
        XCTAssertEqual(retained.reasoning, "unfi")
        XCTAssertEqual(retained.content, "")
    }

    private struct RecordedMessage: Sendable {
        let role: Chat.Message.Role
        let content: String
        let reasoning: String?
    }

    private struct RecordingProcessor: UserInputProcessor {
        let promptTail: String
        let tokenizer: ByteTokenizer
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func prepare(input: UserInput) throws -> LMInput {
            guard case .chat(let messages) = input.prompt else {
                throw TokenizerError.missingChatTemplate
            }
            continuation.yield(
                messages.map {
                    RecordedMessage(role: $0.role, content: $0.content, reasoning: $0.reasoning)
                })
            let transcript = messages.map { "\($0.role.rawValue): \($0.content)\n" }.joined()
            let prompt = transcript + "assistant:\n" + promptTail
            return LMInput(
                tokens: MLXArray(tokenizer.encode(text: prompt, addSpecialTokens: false)))
        }
    }

    private struct ByteTokenizer: Tokenizer {
        var bosToken: String? { nil }
        var eosToken: String? { nil }
        var unknownToken: String? { nil }
        var eosTokenId: Int? { 256 }
        var unknownTokenId: Int? { 257 }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.compactMap { UInt8(exactly: $0) }, as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? {
            token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil
        }

        func convertIdToToken(_ id: Int) -> String? {
            UInt8(exactly: id).map { String(decoding: [$0], as: UTF8.self) }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    private final class ScriptedModel: Module, LanguageModel, KVCacheDimensionProvider {
        let output: [Int]
        private var index = 0
        var kvHeads: [Int] { [1] }

        init(output: [Int]) {
            self.output = output
            super.init()
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            index = 0
            return .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            let length = inputs.size
            if let cache = cache?.first {
                let entries = MLXArray.zeros([1, 1, length, 1])
                _ = cache.update(keys: entries, values: entries)
            }
            let token = output[min(index, output.count - 1)]
            index += 1
            var logits = Array(repeating: Float(-100), count: length * 258)
            for position in 0 ..< length { logits[position * 258 + token] = 100 }
            return MLXArray(logits, [1, length, 258])
        }
    }

    /// The assistant message that the session retained from its first turn, as
    /// the processor received it on the second turn.
    private func retainedAssistant(
        promptTail: String, output: String,
        reasoning: ReasoningConfig? = QwenReasoningProtocol.qwen35,
        maxTokens: Int? = nil
    ) async throws -> Chat.Message {
        let (observations, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let tokenizer = ByteTokenizer()
        let outputTokens = tokenizer.encode(text: output, addSpecialTokens: false) + [256]
        let processor = RecordingProcessor(
            promptTail: promptTail, tokenizer: tokenizer, continuation: continuation)
        let context = ModelContext(
            configuration: ModelConfiguration(
                id: "reasoning-history-test", eosTokenIds: [256], reasoningConfig: reasoning),
            model: ScriptedModel(output: outputTokens), processor: processor, tokenizer: tokenizer)
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(
                maxTokens: maxTokens ?? outputTokens.count, temperature: 0))
        let firstOutput = try await session.respond(to: "first")
        XCTAssertEqual(firstOutput, String(output.prefix(maxTokens ?? output.count)))
        _ = try await session.respond(to: "second")
        continuation.finish()
        var calls: [[RecordedMessage]] = []
        for await messages in observations { calls.append(messages) }
        XCTAssertEqual(calls.count, 2)
        let assistant = try XCTUnwrap(calls.last?.first { $0.role == .assistant })
        return Chat.Message.assistant(assistant.content, reasoning: assistant.reasoning)
    }
}
