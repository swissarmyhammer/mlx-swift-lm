// Copyright © 2026 Apple Inc.

import MLXLMCommon
import MLXVLM
import Testing

/// `Chat.Message.reasoning` → `reasoning_content` emission by the default
/// `MessageGenerator` dictionary conversion.
///
/// History-preserving chat templates (Qwen3.6's `preserve_thinking`) read a
/// past assistant turn's chain-of-thought from the raw message dictionary's
/// `reasoning_content` key; every other family must keep producing exactly
/// the `role`/`content` (+ tool metadata) dictionaries it always has.
struct MessageGeneratorReasoningTests {

    @Test("Assistant reasoning is emitted as reasoning_content")
    func assistantReasoningEmitsReasoningContent() {
        let message = Chat.Message.assistant("4", reasoning: "2 + 2 is 4")
        let raw = DefaultMessageGenerator().generate(message: message)
        #expect(raw["role"] as? String == "assistant")
        #expect(raw["content"] as? String == "4")
        #expect(raw["reasoning_content"] as? String == "2 + 2 is 4")
    }

    @Test("Without reasoning the dictionary carries no reasoning_content key")
    func nilReasoningOmitsKey() {
        let message = Chat.Message.assistant("4")
        let raw = DefaultMessageGenerator().generate(message: message)
        #expect(raw["reasoning_content"] == nil)
        #expect(Set(raw.keys) == ["role", "content"])
    }

    @Test("Reasoning rides alongside tool-call metadata, not instead of it")
    func reasoningCoexistsWithToolCalls() {
        let call = ToolCall(function: .init(name: "lookup", arguments: [:]))
        var message = Chat.Message.assistant("", toolCalls: [call])
        message.reasoning = "let me check"
        let raw = DefaultMessageGenerator().generate(message: message)
        #expect(raw["reasoning_content"] as? String == "let me check")
        #expect(raw["tool_calls"] != nil)
    }

    /// Qwen3.6 checkpoints ship a VLM processor config and load through
    /// `Qwen3VLMessageGenerator`, so the live render's generator must emit
    /// `reasoning_content` exactly like the default one — otherwise the
    /// preserved-thinking history render silently drops the replayed
    /// reasoning on the VLM path.
    @Test("Qwen3VLMessageGenerator emits reasoning_content like the default generator")
    func qwen3VLGeneratorEmitsReasoningContent() {
        let message = Chat.Message.assistant("4", reasoning: "2 + 2 is 4")
        let raw = Qwen3VLMessageGenerator().generate(message: message)
        #expect(raw["reasoning_content"] as? String == "2 + 2 is 4")

        let plain = Qwen3VLMessageGenerator().generate(
            message: Chat.Message.assistant("4"))
        #expect(plain["reasoning_content"] == nil)
    }
}
