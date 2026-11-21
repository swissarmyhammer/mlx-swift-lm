// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// The thinking switch and the history of the Inkling 2 reasoning protocol.
///
/// The published template reads the thinking effort from `reasoning_effort`, a level and not a
/// Boolean: `none` turns thinking off, no value keeps the default level of the template, and
/// `true` would fail its range check. The template writes the reasoning of every past assistant
/// message, thus the history keeps it.
struct InklingReasoningProtocolTests {

    /// The template variable of the thinking effort.
    private static let effortKey = "reasoning_effort"

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
}
