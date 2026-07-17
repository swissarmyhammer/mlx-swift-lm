// Copyright © 2025 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite
struct ReasoningConfigTests {

    // MARK: - infer

    @Test func inferQwen3() {
        let config = ReasoningConfig.infer(from: "qwen3", modelID: "mlx-community/Qwen3-4B-4bit")
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.endDelimiter == "</think>")
        #expect(config?.promptStrategy == .templateFlag(key: "enable_thinking", defaultOn: true))
    }

    @Test func inferDeepSeekV3IsAlwaysOn() {
        let config = ReasoningConfig.infer(
            from: "deepseek_v3", modelID: "mlx-community/DeepSeek-R1-4bit")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.endDelimiter == "</think>")
    }

    /// R1-Distill reports `model_type == "qwen2"` — indistinguishable from plain
    /// Qwen2.5 by type alone. It must be recognized by repo id (the load-bearing
    /// `modelID` parameter).
    @Test func inferR1DistillByIdNotType() {
        let config = ReasoningConfig.infer(
            from: "qwen2", modelID: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
    }

    /// MiniMax-M2 (model_type "minimax") is an interleaved-thinking model:
    /// its chat template pre-opens `<think>\n` in every generation prompt
    /// (there is no thinking-toggle kwarg), so reasoning is always on.
    @Test func inferMiniMaxM2IsAlwaysOn() {
        let config = ReasoningConfig.infer(
            from: "minimax", modelID: "mlx-community/MiniMax-M2-4bit")
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.endDelimiter == "</think>")
        #expect(config?.promptStrategy == .alwaysOn)
    }

    /// The Qwen3 entry is keyed on the model_type *prefix*, so family
    /// variants (e.g. `qwen3_moe`) resolve the same configuration as plain
    /// `qwen3`.
    @Test func inferQwen3PrefixVariant() {
        let config = ReasoningConfig.infer(
            from: "qwen3_moe", modelID: "mlx-community/Qwen3-30B-A3B-4bit")
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.promptStrategy == .templateFlag(key: "enable_thinking", defaultOn: true))
    }

    /// A checkpoint reporting `model_type == "deepseek_v3"` resolves
    /// always-on by type alone — no repo-id signal needed (a plain
    /// DeepSeek-V3 upload carries no "deepseek-r1"/"r1-distill" id marker).
    @Test func inferDeepSeekV3TypeAloneIsAlwaysOn() {
        let config = ReasoningConfig.infer(from: "deepseek_v3")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.endDelimiter == "</think>")
    }

    /// A checkpoint reporting `model_type == "deepseek_r1"` resolves
    /// always-on by type alone — no repo-id signal needed.
    @Test func inferDeepSeekR1TypeIsAlwaysOn() {
        let config = ReasoningConfig.infer(from: "deepseek_r1")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
        #expect(config?.endDelimiter == "</think>")
    }

    /// An id carrying the "deepseek-r1" signal without any "r1-distill"
    /// substring, on a base model_type, resolves via that id fallback alone.
    @Test func inferDeepSeekR1IdWithoutDistillSuffix() {
        let config = ReasoningConfig.infer(
            from: "qwen2", modelID: "someorg/DeepSeek-R1-Custom-4bit")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
    }

    /// An id carrying only the "r1-distill" signal (no "deepseek-r1"
    /// substring) still resolves always-on — the two id signals are
    /// independent.
    @Test func inferR1DistillIdWithoutDeepSeekPrefix() {
        let config = ReasoningConfig.infer(
            from: "llama", modelID: "someorg/R1-Distill-Llama-70B-4bit")
        #expect(config?.promptStrategy == .alwaysOn)
        #expect(config?.startDelimiter == "<think>")
    }

    /// "minimax" is an exact-type match: earlier MiniMax families
    /// (minimax_text_01, minimax_m1) must not resolve M2's configuration.
    @Test func inferMiniMaxIsExactNotPrefixMatch() {
        #expect(
            ReasoningConfig.infer(
                from: "minimax_text_01", modelID: "MiniMaxAI/MiniMax-Text-01") == nil)
        #expect(ReasoningConfig.infer(from: "minimax_m1") == nil)
    }

    /// Both signals are matched case-insensitively: `model_type` and the
    /// repo id are lowercased before inference runs.
    @Test func inferLowercasesTypeAndId() {
        #expect(ReasoningConfig.infer(from: "MiniMax")?.promptStrategy == .alwaysOn)
        #expect(
            ReasoningConfig.infer(
                from: "qwen2", modelID: "MLX-COMMUNITY/DEEPSEEK-R1-DISTILL-QWEN-7B-4BIT")?
                .promptStrategy == .alwaysOn)
    }

    @Test func inferPlainQwen2IsNil() {
        #expect(
            ReasoningConfig.infer(
                from: "qwen2", modelID: "mlx-community/Qwen2.5-3B-Instruct-4bit") == nil)
    }

    @Test func inferGemmaIsNil() {
        #expect(
            ReasoningConfig.infer(from: "gemma3", modelID: "mlx-community/gemma-3-270m-it-4bit")
                == nil)
    }

    @Test func inferLlamaIsNil() {
        #expect(
            ReasoningConfig.infer(
                from: "llama", modelID: "mlx-community/Llama-3.2-3B-Instruct-4bit") == nil)
    }

    /// `modelID` defaults to nil; type-only inference must still work for the
    /// VLM-style bare call site.
    @Test func inferWithoutModelID() {
        #expect(
            ReasoningConfig.infer(from: "qwen3")?.promptStrategy
                == .templateFlag(key: "enable_thinking", defaultOn: true))
        #expect(ReasoningConfig.infer(from: "gemma3") == nil)
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
    /// any future configuration that resolves `.none` (today nothing in
    /// `ReasoningConfig.infer` does, but a custom resolver could).
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
