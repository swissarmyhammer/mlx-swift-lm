// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// A named reasoning effort level gets to the template under the effort key of the protocol,
/// only when thinking is on.
///
/// The Qwen3.8 template reads `reasoning_effort` (`xhigh` by default, `medium`, `low`). The
/// Inkling 2 template reads its level and its off value `none` from the same key. A protocol
/// with no effort key ignores the level.
struct ReasoningEffortRenderTests {

    /// The template variable of the effort level.
    private static let effortKey = "reasoning_effort"

    /// The template flag that turns Qwen thinking on and off.
    private static let thinkingFlag = "enable_thinking"

    /// The level of the tests.
    private static let level = "low"

    @Test func `the Qwen 3.5 protocol and the Inkling protocol read the effort key`() {
        #expect(QwenReasoningProtocol.qwen35.effortKey == Self.effortKey)
        #expect(InklingReasoningProtocol.inkling2.effortKey == Self.effortKey)
        #expect(QwenReasoningProtocol.qwen3.effortKey == nil)
        #expect(ReasoningConfig.gemma4.effortKey == nil)
    }

    @Test(arguments: [true, nil] as [Bool?])
    func `a Qwen 3.5 render with thinking on states the level`(thinking: Bool?) throws {
        let render = try ThinkingRender(
            config: QwenReasoningProtocol.qwen35, thinkingEnabled: thinking, effort: Self.level)

        #expect(render.additionalContext?[Self.effortKey] as? String == Self.level)
        #expect(render.additionalContext?[Self.thinkingFlag] as? Bool == true)
    }

    @Test func `a Qwen 3.5 render with no level states no level`() throws {
        let render = try ThinkingRender(
            config: QwenReasoningProtocol.qwen35, thinkingEnabled: true, effort: nil)

        #expect(render.additionalContext?[Self.effortKey] == nil)
        #expect(render.additionalContext?[Self.thinkingFlag] as? Bool == true)
    }

    @Test func `a Qwen 3.5 render with thinking off states no level`() throws {
        let withLevel = try ThinkingRender(
            config: QwenReasoningProtocol.qwen35, thinkingEnabled: false, effort: Self.level)
        let withoutLevel = try ThinkingRender(
            config: QwenReasoningProtocol.qwen35, thinkingEnabled: false)

        #expect(withLevel.additionalContext?[Self.effortKey] == nil)
        #expect(withLevel.additionalContext?.keys.sorted() == [Self.thinkingFlag])
        #expect(withoutLevel.additionalContext?.keys.sorted() == [Self.thinkingFlag])
    }

    @Test func `a protocol with no effort key ignores the level`() throws {
        let render = try ThinkingRender(
            config: QwenReasoningProtocol.qwen3, thinkingEnabled: true, effort: Self.level)

        #expect(render.additionalContext?.keys.sorted() == [Self.thinkingFlag])
    }

    @Test func `an Inkling render with thinking on states the level`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: true,
            effort: Self.level)

        #expect(render.additionalContext?.keys.sorted() == [Self.effortKey])
        #expect(render.additionalContext?[Self.effortKey] as? String == Self.level)
    }

    @Test func `an Inkling render with thinking off keeps the off value`() throws {
        let render = try ThinkingRender(
            config: InklingReasoningProtocol.inkling2, thinkingEnabled: false,
            effort: Self.level)

        #expect(
            render.additionalContext?[Self.effortKey] as? String
                == InklingReasoningProtocol.thinkingOffEffort)
    }

    @Test func `the level replaces a level that the caller set`() throws {
        let render = try ThinkingRender(
            config: QwenReasoningProtocol.qwen35, thinkingEnabled: true, effort: Self.level)

        let context = render.templateContext(over: [Self.effortKey: "medium", "other": 1])

        #expect(context?[Self.effortKey] as? String == Self.level)
        #expect(context?["other"] as? Int == 1)
    }

    @Test func `an always-on protocol ignores the level`() throws {
        let render = try ThinkingRender(
            config: .alwaysOnThinking, thinkingEnabled: true, effort: Self.level)

        #expect(render.additionalContext == nil)
    }
}
