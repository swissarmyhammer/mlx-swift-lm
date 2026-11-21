// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMTestSupport
import Testing

/// The render of ``InklingFrameTokenizer``, the Inkling 2 test double that the tool-loop suites
/// share.
struct InklingFrameTokenizerTests {

    /// A value that `JSONSerialization` cannot write.
    private struct NotJSON: Sendable {}

    @Test func `a tool call whose arguments are not JSON fails the render`() {
        let tokenizer = InklingFrameTokenizer(pieces: [InklingWeatherToolLoop.toolName])
        let call: [String: any Sendable] = [
            "function": [
                "name": InklingWeatherToolLoop.toolName,
                "arguments": [InklingWeatherToolLoop.cityArgument: NotJSON()]
                    as [String: any Sendable],
            ] as [String: any Sendable]
        ]
        let assistant: [String: any Sendable] = [
            "role": "assistant", "content": "", "tool_calls": [call],
        ]

        #expect(
            throws: InklingFrameTokenizer.ArgumentsAreNotJSON(
                toolName: InklingWeatherToolLoop.toolName)
        ) {
            try tokenizer.applyChatTemplate(
                messages: [assistant], tools: nil, additionalContext: nil)
        }
    }
}
