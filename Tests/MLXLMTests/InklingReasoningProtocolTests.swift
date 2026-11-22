// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon

/// The thinking switch, the history and the answer opening of the Inkling 2 reasoning protocol.
///
/// The published template reads the thinking effort from `reasoning_effort`, a level and not a
/// Boolean: `none` turns thinking off, no value keeps the default level of the template, and
/// `true` would fail its range check. The template writes the reasoning of every past assistant
/// message, thus the history keeps it. The template writes no generation prompt: the model
/// writes the header of its answer frame after the prompt, thus a guided pass appends that
/// header before its grammar starts.
struct InklingReasoningProtocolTests {

    /// The template variable of the thinking effort.
    private static let effortKey = "reasoning_effort"

    /// The question of the prompts of the answer-opening tests.
    private static let question = "Pick a tool."

    /// The tokenizer of the answer-opening tests.
    private static let tokenizer = InklingFrameTokenizer(pieces: [question])

    /// The tokens of a prompt of ``question`` rendered with thinking off.
    private static func thinkingOffPrompt() throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": question]], tools: nil,
            additionalContext: [effortKey: InklingReasoningProtocol.thinkingOffEffort])
    }

    /// The header of the answer frame, in tokens.
    private static var answerHeader: [Int] {
        [tokenizer.token(.messageModel), tokenizer.token(.contentText)]
    }

    @Test func `thinking off sets the effort to none`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: false)

        #expect(render.additionalContext?[Self.effortKey] as? String == "none")
    }

    @Test func `thinking on keeps the default effort of the template`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: true)

        #expect(render.additionalContext == nil)
    }

    @Test func `thinking on keeps the effort level the caller set`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: true)

        let context = render.templateContext(over: [Self.effortKey: "low"])

        #expect(context?[Self.effortKey] as? String == "low")
    }

    @Test func `an unset switch keeps the default effort of the template`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: nil)

        #expect(render.additionalContext == nil)
    }

    @Test func `thinking off closes no block at the end of the prompt`() {
        #expect(
            InklingReasoningProtocol.inkling2.promptStrategy.closedBlock(
                forThinkingEnabled: false) == nil)
    }

    @Test func `the history replays the reasoning of every turn`() {
        #expect(InklingReasoningProtocol.inkling2.replaysReasoningIntoHistory)
    }

    @Test func `the answer opens with the header of the text frame of the model`() {
        #expect(
            InklingReasoningProtocol.inkling2.answerOpening
                == InklingFrameTokenizer.Marker.messageModel.rawValue
                + InklingFrameTokenizer.Marker.contentText.rawValue)
    }

    @Test func `opening the answer appends the header of the answer frame to the prompt`() throws {
        let prompt = try Self.thinkingOffPrompt()

        let opened = InklingReasoningProtocol.inkling2.openingAnswer(
            after: prompt, tokenizer: Self.tokenizer)

        #expect(opened == prompt + Self.answerHeader)
    }

    @Test func `a protocol without an answer opening keeps the prompt`() throws {
        let prompt = try Self.thinkingOffPrompt()

        let opened = QwenReasoningProtocol.qwen35.openingAnswer(
            after: prompt, tokenizer: Self.tokenizer)

        #expect(opened == prompt)
    }

    @Test func `opening the answer of a batched prompt keeps its rank and its mask`() throws {
        let prompt = try Self.thinkingOffPrompt()
        let tokens = MLXArray(prompt.map { Int32($0) })[.newAxis, 0...]
        let input = LMInput(text: .init(tokens: tokens, mask: ones(like: tokens)))

        let opened = ThinkingRender.openingAnswer(
            of: input, config: InklingReasoningProtocol.inkling2, tokenizer: Self.tokenizer)

        #expect(opened.text.tokens.ndim == LMInput.Text.batchedTokenRank)
        #expect(opened.text.tokens.asArray(Int.self) == prompt + Self.answerHeader)
        #expect(opened.text.mask?.size == prompt.count + Self.answerHeader.count)
    }
}
