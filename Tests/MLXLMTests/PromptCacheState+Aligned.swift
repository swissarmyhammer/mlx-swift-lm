// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon

extension PromptCacheState {

    /// A cache whose timeline agrees with its ledger: it processed every cached token, and its
    /// main cache is aligned.
    ///
    /// Shared by the rule suites `CommittedTurnThinkBlockTests` and
    /// `InklingCommittedTurnRuleTests`.
    ///
    /// - Parameters:
    ///   - cached: the tokens of the ledger.
    ///   - previousRender: the render of the turn that wrote the ledger.
    /// - Returns: the state of the cache.
    static func aligned(_ cached: [Int], previousRender: [Int]) -> PromptCacheState {
        PromptCacheState(
            cachedTokens: cached, previousRenderTokens: previousRender,
            processedTokenCount: cached.count, mainCacheIsAligned: true)
    }
}
