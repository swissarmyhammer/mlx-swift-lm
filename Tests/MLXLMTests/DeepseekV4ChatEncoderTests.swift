// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift @ 4546a5d720e7013adffdbddd728c6106e4f9e637
// Manual transcription; no git ancestry.
//
// Every expected string below is a hand transcription of the output of
// DeepSeek's own Python encoder, `encoding/encoding_dsv4.py`, in the model
// repository `deepseek-ai/DeepSeek-V4-Flash` @
// 60d8d70770c6776ff598c94bb586a859a38244f1 (sha256 of that file is
// bdbd57c132a1b372…). No expected string comes from ``DeepSeekV4ChatEncoder``,
// thus a defect in the encoder cannot write itself into the expectation.
//
// ``expectedFixtureTwo`` is a character for character transcription of the
// published golden file `encoding/tests/test_output_2.txt` in the same model
// repository (sha256 9b366d9d2eac842a…). The other 15 conversations were
// rendered by running that Python file over the message lists this file
// declares.
//
// The special tokens in each expected string use the FULLWIDTH VERTICAL LINE
// U+FF5C and the LOWER ONE EIGHTH BLOCK U+2581, not the ASCII `|` and `_`.
// ``testSpecialTokensUseFullwidthVerticalLine`` examines those code points on
// their own, because the two pairs look almost the same on a screen.
//
// The published DeepSeek fixtures 1 and 3 need tool schemas and DSML tool
// calls, which this task leaves to the follow-on task. They are not here.

import Foundation
import MLXLMCommon
import XCTest

/// Examines ``DeepSeekV4ChatEncoder`` against the output of the Python encoder
/// that DeepSeek ships in the DeepSeek-V4-Flash model repository.
final class DeepseekV4ChatEncoderTests: XCTestCase {

    private let encoder = DeepSeekV4ChatEncoder()

    private let systemMessage = DeepSeekV4ChatEncoder.Message.system(
        content: "You are a helpful assistant.")

    /// The three turns that the multi-turn tests share.
    private var threeTurns: [DeepSeekV4ChatEncoder.Message] {
        [
            systemMessage,
            .user(content: "One"),
            .assistant(content: "First answer.", reasoning: "First reasoning."),
            .user(content: "Two"),
            .assistant(content: "Second answer.", reasoning: "Second reasoning."),
            .user(content: "Three"),
        ]
    }

    // MARK: - Thinking mode and the generation tail

    func testChatModeEndsWithClosedThinkTail() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Hello")], thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>")
    }

    func testThinkingModeEndsWithOpenThinkTail() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Hello")], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜><think>")
    }

    // MARK: - Reasoning effort

    func testReasoningEffortMaxPrefacesTheFirstMessage() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Hello")], thinkingMode: .thinking,
            reasoningEffort: .max)
        XCTAssertEqual(
            prompt,
            """
            <｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted.
            You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
            Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

            You are a helpful assistant.<｜User｜>Hello<｜Assistant｜><think>
            """)
    }

    func testReasoningEffortHighAddsNoPreface() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Hello")], thinkingMode: .thinking,
            reasoningEffort: .high)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜><think>")
    }

    func testReasoningEffortMaxAddsNoPrefaceInChatMode() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Hello")], thinkingMode: .chat,
            reasoningEffort: .max)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>")
    }

    // MARK: - drop_earlier_reasoning

    func testDropEarlierReasoningRemovesEveryEarlierThinkBlock() {
        let prompt = encoder.encode(messages: threeTurns, thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>One<｜Assistant｜></think>First answer.<｜end▁of▁sentence｜><｜User｜>Two<｜Assistant｜></think>Second answer.<｜end▁of▁sentence｜><｜User｜>Three<｜Assistant｜><think>"
        )
    }

    func testKeepEarlierReasoningRendersEveryThinkBlock() {
        let prompt = encoder.encode(
            messages: threeTurns, thinkingMode: .thinking, dropsEarlierReasoning: false)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>One<｜Assistant｜><think>First reasoning.</think>First answer.<｜end▁of▁sentence｜><｜User｜>Two<｜Assistant｜><think>Second reasoning.</think>Second answer.<｜end▁of▁sentence｜><｜User｜>Three<｜Assistant｜><think>"
        )
    }

    func testChatModeRendersNoThinkBlockOnAnyTurn() {
        let prompt = encoder.encode(messages: threeTurns, thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>One<｜Assistant｜></think>First answer.<｜end▁of▁sentence｜><｜User｜>Two<｜Assistant｜></think>Second answer.<｜end▁of▁sentence｜><｜User｜>Three<｜Assistant｜></think>"
        )
    }

    // MARK: - developer and latest_reminder roles

    func testDeveloperRoleRendersWithTheUserMarker() {
        let prompt = encoder.encode(
            messages: [systemMessage, .developer(content: "Answer in one word.")],
            thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Answer in one word.<｜Assistant｜><think>"
        )
    }

    func testDeveloperMessageBeforeTheLastUserIsRemoved() {
        let prompt = encoder.encode(
            messages: [
                systemMessage,
                .developer(content: "Answer in one word."),
                .assistant(content: "Yes.", reasoning: "Earlier reasoning."),
                .user(content: "Again?"),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.Yes.<｜end▁of▁sentence｜><｜User｜>Again?<｜Assistant｜><think>"
        )
    }

    func testLatestReminderRendersWithItsOwnMarker() {
        let prompt = encoder.encode(
            messages: [
                systemMessage,
                .latestReminder(content: "2026-08-11, London"),
                .user(content: "Hello"),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜latest_reminder｜>2026-08-11, London<｜User｜>Hello<｜Assistant｜><think>"
        )
    }

    // MARK: - Quick instruction tasks

    func testActionTaskRendersAssistantMarkerThenTaskToken() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Weather?", task: .action)],
            thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Weather?<｜Assistant｜></think><｜action｜>"
        )
    }

    func testQueryTaskRendersOnlyTheTaskToken() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "Weather?", task: .query)],
            thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Weather?<｜query｜>")
    }

    // MARK: - Sentence markers and message merging

    func testAssistantWithoutEndOfSentenceAndWithoutBeginOfSentence() {
        let prompt = encoder.encode(
            messages: [
                systemMessage,
                .user(content: "Hello"),
                .assistant(content: "Hi", endsWithEndOfSentence: false),
            ], thinkingMode: .chat, addsBeginOfSentence: false)
        XCTAssertEqual(prompt, "You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi")
    }

    func testConsecutiveUserMessagesJoinIntoOneTurn() {
        let prompt = encoder.encode(
            messages: [systemMessage, .user(content: "One"), .user(content: "Two")],
            thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>One\n\nTwo<｜Assistant｜></think>"
        )
    }

    func testContextMessagesAreNotRendered() {
        let prompt = encoder.encode(
            messages: [.user(content: "Two")], thinkingMode: .chat,
            context: [systemMessage, .user(content: "One")])
        XCTAssertEqual(prompt, "<｜User｜>Two<｜Assistant｜></think>")
    }

    // MARK: - The published golden fixture

    func testPublishedFixtureTwoRendersByteIdentically() {
        let prompt = encoder.encode(
            messages: [
                systemMessage,
                .user(content: "Hello"),
                .assistant(
                    content: "Hi there! How can I help you?",
                    reasoning: "The user said hello, I should greet back."),
                .user(content: "What is the capital of France?"),
                .assistant(
                    content: "The capital of France is Paris.",
                    reasoning: "The user asks about the capital of France. It is Paris."),
            ], thinkingMode: .thinking)
        XCTAssertEqual(prompt, Self.expectedFixtureTwo)
        XCTAssertFalse(
            prompt.contains("The user said hello"),
            "drop_earlier_reasoning must remove the reasoning of the first assistant turn")
    }

    /// The published golden output `encoding/tests/test_output_2.txt`.
    private static let expectedFixtureTwo =
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜><think>The user asks about the capital of France. It is Paris.</think>The capital of France is Paris.<｜end▁of▁sentence｜>"

    // MARK: - Special token spelling

    func testSpecialTokensUseFullwidthVerticalLine() {
        let expected: [(String, [Unicode.Scalar])] = [
            (
                DeepSeekV4ChatEncoder.SpecialToken.beginOfSentence,
                [
                    "<", "\u{FF5C}", "b", "e", "g", "i", "n", "\u{2581}", "o", "f", "\u{2581}", "s",
                    "e", "n", "t", "e", "n", "c", "e", "\u{FF5C}", ">",
                ]
            ),
            (
                DeepSeekV4ChatEncoder.SpecialToken.endOfSentence,
                [
                    "<", "\u{FF5C}", "e", "n", "d", "\u{2581}", "o", "f", "\u{2581}", "s", "e", "n",
                    "t", "e", "n", "c", "e", "\u{FF5C}", ">",
                ]
            ),
            (
                DeepSeekV4ChatEncoder.SpecialToken.user,
                [
                    "<", "\u{FF5C}", "U", "s", "e", "r", "\u{FF5C}", ">",
                ]
            ),
            (
                DeepSeekV4ChatEncoder.SpecialToken.assistant,
                [
                    "<", "\u{FF5C}", "A", "s", "s", "i", "s", "t", "a", "n", "t", "\u{FF5C}", ">",
                ]
            ),
            (DeepSeekV4ChatEncoder.SpecialToken.thinkStart, ["<", "t", "h", "i", "n", "k", ">"]),
            (
                DeepSeekV4ChatEncoder.SpecialToken.thinkEnd,
                [
                    "<", "/", "t", "h", "i", "n", "k", ">",
                ]
            ),
        ]
        for (token, scalars) in expected {
            XCTAssertEqual(Array(token.unicodeScalars), scalars, "wrong spelling of `\(token)`")
        }
    }

    func testLatestReminderAndTaskTokensUseFullwidthVerticalLine() {
        let fullwidth: Unicode.Scalar = "\u{FF5C}"
        let named: [(String, String)] = [
            (DeepSeekV4ChatEncoder.SpecialToken.latestReminder, "latest_reminder"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.action), "action"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.query), "query"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.authority), "authority"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.domain), "domain"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.title), "title"),
            (DeepSeekV4ChatEncoder.SpecialToken.task(.readURL), "read_url"),
        ]
        for (token, name) in named {
            var scalars: [Unicode.Scalar] = ["<", fullwidth]
            scalars.append(contentsOf: name.unicodeScalars)
            scalars.append(contentsOf: [fullwidth, ">"])
            XCTAssertEqual(Array(token.unicodeScalars), scalars, "wrong spelling of `\(token)`")
        }
    }
}
