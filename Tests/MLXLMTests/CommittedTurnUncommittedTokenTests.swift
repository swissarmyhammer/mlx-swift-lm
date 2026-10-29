// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests of the splice the two committed-turn rules make after a turn whose last
/// generated tokens never reached the caches (card `^g2t7caa`).
///
/// An MTP round returns its final verified token before any forward pass feeds it. A turn that
/// stops on its token budget there leaves that token out of the ledger, and the session keeps it
/// as an uncommitted token. The splice feeds it ahead of the commit that closes the turn, thus
/// the model reads its whole answer before the commit, as it does after a plain decode.
///
/// Each row of ``CommittedTurnThinkBlockTests/ruleCases`` runs against each test. The token 70
/// below stands for a token that only the model wrote, 71 for the token the render writes in its
/// place, and 72 for the last token the model wrote, which the caches do not hold.
struct CommittedTurnUncommittedTokenTests {

    /// Stands in for the commit token id of each rule.
    private static let commit = 1

    /// Stands in for the `<think>` token id.
    private static let openThink = 2

    /// Stands in for the `</think>` token id.
    private static let closeThink = 3

    /// Stands in for a token of the conversation before the generation prompt.
    private static let conversation = 10

    /// Stands in for the token that opens the assistant turn.
    private static let assistant = 11

    /// Stands in for the token that the render adds after the commit.
    private static let nextTurn = 20

    /// Stands in for a token that only the model wrote.
    private static let generatedOnly = 70

    /// Stands in for the token the render writes where the model wrote ``generatedOnly``.
    private static let renderedOnly = 71

    /// Stands in for the last token the model wrote, which no forward pass fed.
    private static let unfed = 72

    /// The render of the previous prompt with thinking on: it ends inside an open think block.
    private static let previousRender = [conversation, assistant, openThink]

    /// The ledger of a turn that closed its reasoning and answered, and stopped on its budget
    /// before the caches held its last token.
    private static let answeredLedger = previousRender + [generatedOnly, closeThink, generatedOnly]

    /// The render of the next turn after ``answeredLedger``: the history of the turn, its commit,
    /// and the new message.
    private static let followUpRender =
        previousRender + [renderedOnly, closeThink, renderedOnly, unfed, commit, nextTurn]

    /// The index of the commit in ``followUpRender``.
    private static let commitIndex = followUpRender.count - 2

    /// The rule `rule` makes from a tokenizer that names the markers of this suite.
    ///
    /// - Parameter rule: The rule case.
    /// - Returns: The rule.
    /// - Throws: An issue when the rule makes nothing from the tokenizer.
    private static func made(_ rule: CommittedTurnThinkBlockTests.RuleCase) throws
        -> any PromptCacheReuseRule
    {
        let tokenizer = MarkerTokenizer(identifierOfMarker: [
            rule.endOfTurnMarker: commit,
            rule.openThinkMarker: openThink,
            rule.closeThinkMarker: closeThink,
        ])
        return try #require(rule.rule(tokenizer))
    }

    /// A cache whose timeline agrees with its ledger, and whose drafter state can follow it.
    ///
    /// - Parameter cached: The tokens the caches hold.
    /// - Returns: The cache state.
    private static func alignedCache(_ cached: [Int]) -> PromptCacheState {
        PromptCacheState(
            cachedTokens: cached, previousRenderTokens: previousRender,
            processedTokenCount: cached.count, mainCacheIsAligned: true,
            speculativeReuseCapability: .restartsOnRewind)
    }

    /// The turn of ``followUpRender`` after a generation that left `uncommitted` out of the
    /// caches.
    ///
    /// - Parameters:
    ///   - uncommitted: The tokens the previous generation returned and the caches miss.
    ///   - prompt: The render of the turn.
    ///   - speculates: Whether the session speculates.
    /// - Returns: The turn.
    private static func turn(
        uncommitted: [Int], prompt: [Int] = followUpRender, speculates: Bool = false
    ) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt, previousGenerationUncommittedTokens: uncommitted,
            usesSpeculativeDecoding: speculates)
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `feeds the unfed last token ahead of the commit`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.unfed]),
            cache: Self.alignedCache(Self.answeredLedger))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: Self.commitIndex,
                    representedTokens: Self.answeredLedger + [
                        Self.unfed, Self.commit, Self.nextTurn,
                    ],
                    uncommittedTokens: [Self.unfed]))
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `feeds the unfed tokens and then the commit of the render in place of the model's commit`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        // The round returned the last answer token and the commit, and fed neither. The render's
        // commit stands for the model's, thus the model reads it once.
        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.unfed, Self.commit]),
            cache: Self.alignedCache(Self.answeredLedger))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: Self.commitIndex,
                    representedTokens: Self.answeredLedger + [
                        Self.unfed, Self.commit, Self.nextTurn,
                    ],
                    uncommittedTokens: [Self.unfed]))
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `splices a turn whose unfed token closes its reasoning`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        // The caches end inside the think block, but the model closed it with the token they
        // miss. The model wrote no answer, thus it reads the close and then the commit.
        let ledger = Self.previousRender + [Self.generatedOnly]
        let prompt =
            Self.previousRender + [Self.renderedOnly, Self.closeThink, Self.commit, Self.nextTurn]

        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.closeThink], prompt: prompt),
            cache: Self.alignedCache(ledger))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: prompt.count - 2,
                    representedTokens: ledger + [Self.closeThink, Self.commit, Self.nextTurn],
                    uncommittedTokens: [Self.closeThink]))
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `feeds the unfed token to the main cache alone when the drafter cannot follow`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        var cache = Self.alignedCache(Self.answeredLedger)
        cache.speculativeReuseCapability = .mainOnlyFallback

        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.unfed], speculates: true), cache: cache)

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: Self.commitIndex,
                    representedTokens: Self.answeredLedger + [
                        Self.unfed, Self.commit, Self.nextTurn,
                    ],
                    uncommittedTokens: [Self.unfed]))
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `declines unfed tokens after a ledger that holds the commit`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        // The model closed its turn and the caches hold the commit, thus a token past it is
        // lookahead the render cannot explain.
        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.unfed]),
            cache: Self.alignedCache(Self.answeredLedger + [Self.commit]))

        #expect(decision == nil)
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `declines unfed tokens that hold a commit before their end`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        let decision = try Self.made(rule).reuse(
            turn: Self.turn(uncommitted: [Self.commit, Self.unfed]),
            cache: Self.alignedCache(Self.answeredLedger))

        #expect(decision == nil)
    }

    @Test(arguments: CommittedTurnThinkBlockTests.ruleCases)
    func `reconciling live caches hands the unfed token to the caller`(
        _ rule: CommittedTurnThinkBlockTests.RuleCase
    ) throws {
        // The function reads the offset of each cache alone, thus a cache placed at the end of
        // the ledger by hand serves.
        let ledger = Self.answeredLedger
        let cache = KVCacheSimple()
        cache.offset = ledger.count
        let caches: [KVCache] = [cache]

        let reuse = try #require(
            reconcilePromptCache(
                promptTokens: Self.followUpRender, cachedTokens: ledger,
                previousRenderTokens: Self.previousRender, caches: caches,
                protocolRules: [try Self.made(rule)],
                previousGenerationUncommittedTokens: [Self.unfed]))

        #expect(reuse.kind == .splice)
        #expect(reuse.uncommittedTokens == [Self.unfed])
        #expect(
            reuse.fedTokens(of: Self.followUpRender) == [Self.unfed, Self.commit, Self.nextTurn])
        #expect(reuse.representedTokens == ledger + reuse.fedTokens(of: Self.followUpRender))
    }
}
