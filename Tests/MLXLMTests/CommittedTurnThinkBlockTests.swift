// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests of the think-block check that the two committed-turn rules share
/// through ``CommittedTurnSplice``.
///
/// A turn that stops inside its reasoning leaves no close of the think block in
/// the caches. The render of that turn closes the block before its commit, and
/// a splice at the commit would make the model read its unclosed reasoning and
/// then the commit. Each rule thus declines such a cache. Card `^ef2zynk`
/// measured that on Qwen 3.8, and card `^mf1mwaw` on DeepSeek-V4.
///
/// Each row of ``ruleCases`` makes one rule, and every row of
/// ``reasoningEndCases`` runs against each rule. The token 70 below stands for
/// a token that only the model wrote, and 71 for the token the render writes in
/// its place.
struct CommittedTurnThinkBlockTests {

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

    /// Stands in for the token the render writes where the model wrote
    /// ``generatedOnly``.
    private static let renderedOnly = 71

    /// The render of a prompt with thinking on: it ends inside an open think
    /// block.
    private static let thinkingOnRender = [conversation, assistant, openThink]

    /// The render of a prompt with thinking off: it ends with a closed, empty
    /// think block.
    private static let thinkingOffRender = [conversation, assistant, openThink, closeThink]

    /// One committed-turn rule, and the markers its tokenizer names.
    struct RuleCase: Sendable, CustomTestStringConvertible {
        /// The name of the rule in the test report.
        let name: String
        /// The marker that closes an assistant turn.
        let endOfTurnMarker: String
        /// The marker that opens a think block.
        let openThinkMarker: String
        /// The marker that closes a think block.
        let closeThinkMarker: String
        /// The rule as its response format makes it from a tokenizer.
        let rule: @Sendable (any Tokenizer) -> (any PromptCacheReuseRule)?
        /// The rule made from the commit token alone, with no think markers.
        let markerlessRule: @Sendable (Int) -> any PromptCacheReuseRule

        var testDescription: String { name }
    }

    /// The rules, one row for each response format that splices a committed
    /// turn.
    static let ruleCases: [RuleCase] = [
        RuleCase(
            name: "qwen35",
            endOfTurnMarker: QwenCommittedTurnRule.endOfTurnMarker,
            openThinkMarker: QwenCommittedTurnRule.openThinkMarker,
            closeThinkMarker: QwenCommittedTurnRule.closeThinkMarker,
            rule: { QwenCommittedTurnRule(tokenizer: $0) },
            markerlessRule: { QwenCommittedTurnRule(endOfTurnToken: $0) }),
        RuleCase(
            name: "dsml",
            endOfTurnMarker: DeepSeekV4ChatEncoder.SpecialToken.endOfSentence,
            openThinkMarker: DeepSeekV4ChatEncoder.SpecialToken.thinkStart,
            closeThinkMarker: DeepSeekV4ChatEncoder.SpecialToken.thinkEnd,
            rule: { DSMLCommittedTurnRule(tokenizer: $0) },
            markerlessRule: { DSMLCommittedTurnRule(endOfSentenceToken: $0) }),
    ]

    /// One way the previous turn ended, and what the rule decides for the
    /// render of the next turn.
    struct ReasoningEndCase: Sendable, CustomTestStringConvertible {
        /// The name of the case in the test report.
        let name: String
        /// The render of the previous prefill.
        let previousRender: [Int]
        /// The tokens the caches hold: the previous render and the tokens the
        /// model wrote.
        let cached: [Int]
        /// The render of the next turn.
        let prompt: [Int]
        /// Whether the rule splices. When false, the rule declines.
        let splices: Bool

        var testDescription: String { name }
    }

    /// The cases, one row for each way the previous turn ended. The ledger of
    /// each row that splices does not hold the commit.
    static let reasoningEndCases: [ReasoningEndCase] = [
        ReasoningEndCase(
            name: "declines a turn that stopped on the budget inside its reasoning",
            previousRender: thinkingOnRender,
            cached: thinkingOnRender + [generatedOnly],
            prompt: thinkingOnRender + [renderedOnly, closeThink, commit, nextTurn],
            splices: false),
        ReasoningEndCase(
            name: "declines a turn that ended its turn inside its reasoning",
            previousRender: thinkingOnRender,
            cached: thinkingOnRender + [generatedOnly, commit],
            prompt: thinkingOnRender + [renderedOnly, closeThink, commit, nextTurn],
            splices: false),
        ReasoningEndCase(
            name: "splices a turn that closed its reasoning",
            previousRender: thinkingOnRender,
            cached: thinkingOnRender + [generatedOnly, closeThink, generatedOnly],
            prompt: thinkingOnRender + [renderedOnly, closeThink, renderedOnly, commit, nextTurn],
            splices: true),
        ReasoningEndCase(
            name: "splices a turn with thinking off",
            previousRender: thinkingOffRender,
            cached: thinkingOffRender + [generatedOnly],
            prompt: thinkingOffRender + [renderedOnly, commit, nextTurn],
            splices: true),
    ]

    /// What the rule decides after a turn that ends as `row` tells.
    @Test(arguments: ruleCases, reasoningEndCases)
    func `the splice after a turn that ends`(_ rule: RuleCase, _ row: ReasoningEndCase) throws {
        let tokenizer = MarkerTokenizer(identifierOfMarker: [
            rule.endOfTurnMarker: Self.commit,
            rule.openThinkMarker: Self.openThink,
            rule.closeThinkMarker: Self.closeThink,
        ])
        let made = try #require(rule.rule(tokenizer))

        let decision = made.reuse(
            turn: PromptCacheTurn(promptTokens: row.prompt),
            cache: Self.alignedCache(row.cached, previousRender: row.previousRender))

        let expected =
            row.splices ? try Self.spliceAtTheCommit(of: row.prompt, onto: row.cached) : nil
        #expect(decision == expected)
    }

    /// A rule that knows no think markers cannot tell where reasoning ends,
    /// thus it splices as before.
    @Test(arguments: ruleCases)
    func `a rule without think markers keeps the splice inside reasoning`(_ rule: RuleCase) throws {
        let prompt =
            Self.thinkingOnRender + [
                Self.renderedOnly, Self.closeThink, Self.commit, Self.nextTurn,
            ]
        let cached = Self.thinkingOnRender + [Self.generatedOnly]

        let decision = rule.markerlessRule(Self.commit).reuse(
            turn: PromptCacheTurn(promptTokens: prompt),
            cache: Self.alignedCache(cached, previousRender: Self.thinkingOnRender))

        #expect(decision == (try Self.spliceAtTheCommit(of: prompt, onto: cached)))
    }

    /// A cache whose timeline agrees with its ledger.
    private static func alignedCache(_ cached: [Int], previousRender: [Int]) -> PromptCacheState {
        PromptCacheState(
            cachedTokens: cached, previousRenderTokens: previousRender,
            processedTokenCount: cached.count, mainCacheIsAligned: true)
    }

    /// The splice onto a ledger that does not hold the commit: it starts AT
    /// the commit of `prompt`, thus the fed suffix carries the commit.
    private static func spliceAtTheCommit(of prompt: [Int], onto cached: [Int]) throws
        -> PromptCacheReuseDecision
    {
        let commitIndex = try #require(prompt.firstIndex(of: commit))
        return .appendSuffix(
            suffixStart: commitIndex, representedTokens: cached + prompt[commitIndex...])
    }
}
