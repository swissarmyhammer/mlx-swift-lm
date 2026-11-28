// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// ``PromptCacheCheckpoint/leadingSystemAndLastMessageEnds(of:messages:render:)`` finds the end
/// of the leading instructions and the end of the last message of a prompt, as
/// ``PromptCacheCheckpoint/leadingSystemEnd(of:messages:render:)`` and
/// ``PromptCacheCheckpoint/lastMessageEnd(of:messages:render:)`` find them, from one render of
/// the sample conversation they share.
///
/// The template of these tests writes each message as a role token, the scalars of its content
/// and an end token, and a generation prompt after the last message.
@Suite("The ends of the instructions and of the last message come from one sample render")
struct PromptCacheCheckpointEndsTests {

    /// The tokens below this value are the scalars of the contents.
    private static let scalarTokenCount = 100

    /// The token of each role, and the end of each message.
    private enum Marker: Int {
        case system = 100
        case user
        case assistant
        case end
        case think
    }

    /// The messages of the prompt of each test: instructions and one user message.
    private static var messages: [Chat.Message] {
        [.system("be brief"), .user("hello")]
    }

    /// Counts the renders of one test.
    private final class RenderLog {
        private(set) var renderCount = 0

        /// The tokens of `messages` under the template of these tests.
        ///
        /// - Parameter messages: the messages to render.
        /// - Returns: the prompt.
        func render(_ messages: [Chat.Message]) -> LMInput {
            renderCount += 1
            return LMInput(tokens: MLXArray(PromptCacheCheckpointEndsTests.tokens(of: messages)))
        }
    }

    /// The tokens of `messages` under the template of these tests, the generation prompt
    /// included.
    private static func tokens(of messages: [Chat.Message]) -> [Int32] {
        let body = messages.flatMap { message -> [Int] in
            let role: Marker =
                switch message.role {
                case .system: .system
                case .user, .tool: .user
                case .assistant: .assistant
                }
            let content = message.content.unicodeScalars.map {
                Int($0.value) % scalarTokenCount
            }
            return [role.rawValue] + content + [Marker.end.rawValue]
        }
        return (body + [Marker.assistant.rawValue, Marker.think.rawValue]).map(Int32.init)
    }

    /// The prompt of ``messages``.
    private static var prompt: LMInput {
        LMInput(tokens: MLXArray(tokens(of: messages)))
    }

    @Test("the end of the instructions is the one leadingSystemEnd finds")
    func theEndOfTheInstructionsIsTheOneLeadingSystemEndFinds() async {
        let log = RenderLog()
        let expected = await PromptCacheCheckpoint.leadingSystemEnd(
            of: Self.prompt.text.tokens.asArray(Int.self), messages: Self.messages,
            render: log.render)

        let ends = await PromptCacheCheckpoint.leadingSystemAndLastMessageEnds(
            of: Self.prompt, messages: Self.messages, render: log.render)

        #expect(ends.leadingSystemEnd == expected)
    }

    @Test("the end of the last message is the one lastMessageEnd finds")
    func theEndOfTheLastMessageIsTheOneLastMessageEndFinds() async {
        let log = RenderLog()
        let expected = await PromptCacheCheckpoint.lastMessageEnd(
            of: Self.prompt, messages: Self.messages, render: log.render)

        let ends = await PromptCacheCheckpoint.leadingSystemAndLastMessageEnds(
            of: Self.prompt, messages: Self.messages, render: log.render)

        #expect(ends.lastMessageEnd == expected)
    }

    @Test("both ends cost one render fewer than the two functions of their own")
    func bothEndsCostOneRenderFewerThanTheTwoFunctions() async {
        let separate = RenderLog()
        _ = await PromptCacheCheckpoint.leadingSystemEnd(
            of: Self.prompt.text.tokens.asArray(Int.self), messages: Self.messages,
            render: separate.render)
        _ = await PromptCacheCheckpoint.lastMessageEnd(
            of: Self.prompt, messages: Self.messages, render: separate.render)
        let combined = RenderLog()

        _ = await PromptCacheCheckpoint.leadingSystemAndLastMessageEnds(
            of: Self.prompt, messages: Self.messages, render: combined.render)

        #expect(combined.renderCount == separate.renderCount - 1)
    }

    @Test("a prompt with no instructions has no end of the instructions")
    func aPromptWithNoInstructionsHasNoEndOfTheInstructions() async {
        let log = RenderLog()
        let messages: [Chat.Message] = [.user("hello")]

        let ends = await PromptCacheCheckpoint.leadingSystemAndLastMessageEnds(
            of: LMInput(tokens: MLXArray(Self.tokens(of: messages))), messages: messages,
            render: log.render)

        #expect(ends.leadingSystemEnd == nil)
    }
}
