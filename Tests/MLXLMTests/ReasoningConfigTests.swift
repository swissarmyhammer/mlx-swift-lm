// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite
struct ReasoningConfigTests {

    // MARK: - Reasoning replayed into history

    @Test func `the Qwen 3.5 protocol replays reasoning into history`() {
        // Its chat template keeps the `<think>` block of a past turn, thus the
        // history render of a turn can hold what the model read.
        #expect(QwenReasoningProtocol.qwen35.replaysReasoningIntoHistory)
    }

    @Test func `the other Qwen protocols drop reasoning from history`() {
        #expect(!QwenReasoningProtocol.tagged.replaysReasoningIntoHistory)
        #expect(!QwenReasoningProtocol.qwen3.replaysReasoningIntoHistory)
    }

    @Test func `a protocol drops reasoning from history unless it says otherwise`() {
        #expect(!ReasoningConfig.thinkTagsWithEnableThinking.replaysReasoningIntoHistory)
        #expect(!ReasoningConfig.alwaysOnThinking.replaysReasoningIntoHistory)
    }

    // MARK: - ReasoningPromptStrategy.additionalContext

    @Test func templateFlagThinkingOn() throws {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: "enable_thinking", defaultOn: true)
        let ctx = try strategy.additionalContext(forThinkingEnabled: true)
        #expect(ctx?["enable_thinking"] as? Bool == true)
    }

    @Test func templateFlagThinkingOff() throws {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: "enable_thinking", defaultOn: true)
        let ctx = try strategy.additionalContext(forThinkingEnabled: false)
        #expect(ctx?["enable_thinking"] as? Bool == false)
    }

    @Test func templateFlagUnspecifiedUsesDefaultOn() throws {
        let defaultsOn = ReasoningPromptStrategy.templateFlag(
            key: "enable_thinking", defaultOn: true)
        let defaultsOff = ReasoningPromptStrategy.templateFlag(
            key: "enable_thinking", defaultOn: false)
        #expect(
            try defaultsOn.additionalContext(forThinkingEnabled: nil)?["enable_thinking"] as? Bool
                == true)
        #expect(
            try defaultsOff.additionalContext(forThinkingEnabled: nil)?["enable_thinking"] as? Bool
                == false)
    }

    /// The kwarg name is data: a non-Qwen3 family using a different key works
    /// through the same strategy without a new enum case.
    @Test func templateFlagHonorsCustomKey() throws {
        let strategy = ReasoningPromptStrategy.templateFlag(
            key: "use_chain_of_thought", defaultOn: false)
        let ctx = try strategy.additionalContext(forThinkingEnabled: true)
        #expect(ctx?["use_chain_of_thought"] as? Bool == true)
        #expect(ctx?["enable_thinking"] == nil)
    }

    @Test func alwaysOnIgnoresEnabledLevels() throws {
        let on = try ReasoningPromptStrategy.alwaysOn.additionalContext(forThinkingEnabled: true)
        let unspecified = try ReasoningPromptStrategy.alwaysOn.additionalContext(
            forThinkingEnabled: nil)
        #expect(on == nil)
        #expect(unspecified == nil)
    }

    @Test func alwaysOnThrowsWhenDisabled() {
        #expect(throws: ReasoningError.cannotDisableReasoning) {
            try ReasoningPromptStrategy.alwaysOn.additionalContext(forThinkingEnabled: false)
        }
    }

    /// `.none` is non-suppressible: like `.alwaysOn`, asking to disable
    /// thinking on a `.none` strategy must throw `cannotDisableReasoning`
    /// rather than silently returning nil. The capability gate in the FM
    /// adapter relies on this throw to surface `unsupportedCapability` for
    /// any future configuration that resolves `.none` (nothing declares it
    /// today, but a model or a registered resolver could).
    @Test func noneStrategyThrowsWhenDisabled() {
        #expect(throws: ReasoningError.cannotDisableReasoning) {
            try ReasoningPromptStrategy.none.additionalContext(forThinkingEnabled: false)
        }
    }

    @Test func noneStrategyReturnsNilWhenEnabledOrUnspecified() throws {
        let on = try ReasoningPromptStrategy.none.additionalContext(forThinkingEnabled: true)
        let unspecified = try ReasoningPromptStrategy.none.additionalContext(
            forThinkingEnabled: nil)
        #expect(on == nil)
        #expect(unspecified == nil)
    }

    // MARK: - Conformances (rides on ModelConfiguration: Sendable + Equatable)

    @Test func equatable() {
        let a = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)
        let b = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)
        let c = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))
        #expect(a == b)
        #expect(a != c)
    }
}
