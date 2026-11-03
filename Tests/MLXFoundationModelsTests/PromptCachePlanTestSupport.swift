// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon

@testable import MLXFoundationModels

/// Fixtures for the suites that plan passes over prompt cache entries. A suite conforms to get
/// them as instance members.
///
/// No weights are needed. Each entry carries caches that ``feed(_:tokenCount:)`` moved to the
/// position of its ledger.
protocol PromptCachePlanFixtures {}

extension PromptCachePlanFixtures {

    /// An entry that names `tokens` and carries one cache fed with one position for each token,
    /// thus the entry holds real bytes.
    ///
    /// - Parameters:
    ///   - tokens: The ledger of the entry.
    ///   - state: The model state that the entry carries, or nil for none.
    /// - Returns: The entry.
    func entry(tokens: [Int], state: LMOutput.State? = nil) -> ExecutorPromptCacheEntry {
        let caches: [KVCache] = [KVCacheSimple()]
        feed(caches, tokenCount: tokens.count)
        return ExecutorPromptCacheEntry(caches: caches, tokens: tokens, state: state)
    }
}

/// Plans `render` against `entry`.
///
/// - Parameters:
///   - render: The tokens of the prompt of the pass.
///   - entry: The entry that the pass carries, or nil for none.
///   - model: The model that makes fresh caches. The default owns no attention layer, thus a
///     fresh cache is empty, which is all a plan needs.
///   - protocolRules: The protocol rules of the pass.
/// - Returns: The plan, or nil when the pass carries no cache.
/// - Throws: The error of the plan.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
func planPromptCache(
    render: [Int], reusing entry: ExecutorPromptCacheEntry?,
    model: any MLXLMCommon.LanguageModel = ScriptedLanguageModel(rounds: []),
    protocolRules: [any PromptCacheReuseRule] = []
) throws -> ExecutorPromptCachePlan? {
    try ExecutorPromptCachePlan.make(
        reusing: entry, input: LMInput(tokens: MLXArray(render)), model: model,
        parameters: GenerateParameters(), protocolRules: protocolRules)
}

/// A rule that keeps the tokens the model wrote and feeds the tail of the render after its
/// commit token, the way a committed-turn rule splices.
struct SplicingRule: PromptCacheReuseRule {

    /// The token that ends the turn the model wrote.
    let commit: Int

    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard turn.promptTokens.starts(with: cache.previousRenderTokens),
            let commitIndex = turn.promptTokens.firstIndex(of: commit)
        else { return nil }
        let suffixStart = commitIndex + 1
        return .appendSuffix(
            suffixStart: suffixStart,
            representedTokens: cache.cachedTokens + turn.promptTokens[suffixStart...])
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
