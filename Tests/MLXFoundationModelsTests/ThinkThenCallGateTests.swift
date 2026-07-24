// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Gates for the tool-calling path's think-then-call Phase 1.
///
/// Two gates cooperate (see `runToolCalling`):
///
/// 1. `makeThinkThenCallConfig` — the static, pre-render gate: `.reasoning`
///    must be declared, the resolved configuration must carry a reasoning
///    config with a prompt-level strategy (`.templateFlag` or `.alwaysOn`),
///    and the request must not have disabled thinking.
/// 2. `thinkThenCallPhase1Engages` — the post-render gate: `.templateFlag`
///    families always run Phase 1 (the kwarg primed the template), while
///    `.alwaysOn` families run it only when the rendered prompt actually ends
///    inside an open reasoning span (e.g. MiniMax-M2's template pre-opens
///    `<think>\n`). An `.alwaysOn` configuration whose prompt is NOT primed
///    (e.g. a DeepSeek-V3-style model that does not actually think) stays on
///    the single-phase path unchanged — Phase 1 would otherwise burn the
///    whole budget waiting for a `</think>` that never comes.
@Suite
struct ThinkThenCallGateTests {

    private let templateFlagConfig = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true))

    private let alwaysOnConfig = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .alwaysOn)

    /// MiniMax-M3's toggleable-but-string-valued strategy. Behaves like
    /// `.templateFlag` for both gates below (see `ReasoningConfig.swift`'s
    /// `minimaxM3ThinkConfig` doc): the template both renders the tool block
    /// and honors `thinking_mode`, and -- unlike `.alwaysOn` -- reasoning is
    /// never forced, so priming is irrelevant to whether Phase 1 engages.
    private let templateStringFlagConfig = ReasoningConfig(
        startDelimiter: "<mm:think>", endDelimiter: "</mm:think>",
        promptStrategy: .templateStringFlag(
            key: "thinking_mode", onValue: "enabled", offValue: "disabled",
            defaultValue: "adaptive"))

    private func configuration(reasoning: ReasoningConfig?) -> ModelConfiguration {
        ModelConfiguration(id: "test/model", reasoningConfig: reasoning)
    }

    // MARK: - makeThinkThenCallConfig (static, pre-render gate)

    @Test("templateFlag families still qualify")
    func templateFlagQualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
            declaresReasoning: true,
            resolved: configuration(reasoning: templateFlagConfig),
            reasoningLevel: nil)
        #expect(config == templateFlagConfig)
    }

    @Test("alwaysOn families qualify (MiniMax-M2 is alwaysOn AND tool-aware)")
    func alwaysOnQualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // MiniMax-M2's template renders the tool block AND pre-opens
        // `<think>`, so an alwaysOn model must reach the (post-render)
        // primed-inside gate rather than being ruled out statically here.
        let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
            declaresReasoning: true,
            resolved: configuration(reasoning: alwaysOnConfig),
            reasoningLevel: nil)
        #expect(config == alwaysOnConfig)
    }

    @Test("undeclared .reasoning capability never qualifies")
    func undeclaredReasoningDisqualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        for reasoning in [templateFlagConfig, alwaysOnConfig] {
            let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
                declaresReasoning: false,
                resolved: configuration(reasoning: reasoning),
                reasoningLevel: nil)
            #expect(config == nil)
        }
    }

    @Test("no resolved reasoning config never qualifies")
    func missingReasoningConfigDisqualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
            declaresReasoning: true,
            resolved: configuration(reasoning: nil),
            reasoningLevel: nil)
        #expect(config == nil)
    }

    @Test("thinking-disabled requests stay single-phase")
    func thinkingDisabledDisqualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // `.custom("no_think")` is the one level `thinkingEnabled(for:)` maps
        // to `false` (see its `.custom` normalization).
        for reasoning in [templateFlagConfig, alwaysOnConfig] {
            let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
                declaresReasoning: true,
                resolved: configuration(reasoning: reasoning),
                reasoningLevel: .custom("no_think"))
            #expect(config == nil)
        }
    }

    @Test("templateStringFlag families still qualify (MiniMax-M3)")
    func templateStringFlagQualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
            declaresReasoning: true,
            resolved: configuration(reasoning: templateStringFlagConfig),
            reasoningLevel: nil)
        #expect(config == templateStringFlagConfig)
    }

    @Test(".none prompt strategy never qualifies")
    func noneStrategyDisqualifies() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let noneConfig = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .none)
        let config = MLXLanguageModel.Executor.makeThinkThenCallConfig(
            declaresReasoning: true,
            resolved: configuration(reasoning: noneConfig),
            reasoningLevel: nil)
        #expect(config == nil)
    }

    // MARK: - thinkThenCallPhase1Engages (post-render, primed-inside gate)

    @Test("templateFlag engages Phase 1 regardless of prompt priming")
    func templateFlagAlwaysEngages() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Qwen3-style templates don't pre-open the think block (the model
        // emits `<think>` itself), so Phase 1 must not depend on priming.
        #expect(
            MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: templateFlagConfig, primedInside: false))
        #expect(
            MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: templateFlagConfig, primedInside: true))
    }

    @Test("templateStringFlag engages Phase 1 regardless of prompt priming (MiniMax-M3)")
    func templateStringFlagAlwaysEngages() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // M3 only pre-opens `<mm:think>` when the caller forces
        // `thinking_mode: "enabled"`; the unspecified/adaptive default does
        // not prime the prompt, yet the model may still choose to think, so
        // Phase 1 must engage regardless of priming, exactly like Qwen3.
        #expect(
            MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: templateStringFlagConfig, primedInside: false))
        #expect(
            MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: templateStringFlagConfig, primedInside: true))
    }

    @Test("alwaysOn engages Phase 1 only when the prompt ends inside an open think block")
    func alwaysOnEngagesOnlyWhenPrimed() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // MiniMax-M2's generation prompt ends `…]~b]ai\n<think>\n` — primed.
        #expect(
            MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: alwaysOnConfig, primedInside: true))
        // An alwaysOn-configured model whose template does NOT pre-open the
        // block (and may never emit `</think>`) must stay single-phase.
        #expect(
            !MLXLanguageModel.Executor.thinkThenCallPhase1Engages(
                reasoningConfig: alwaysOnConfig, primedInside: false))
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
