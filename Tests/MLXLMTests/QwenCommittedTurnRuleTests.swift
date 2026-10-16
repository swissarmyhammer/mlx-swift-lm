// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests for the Qwen 3.5 cache-reuse rule of card `^xx5g893`. The rule
/// is pure, thus the splice contract is examined without a model and without a
/// tokenizer.
///
/// Each fixture below writes the shape the real weights write. The LEDGER holds
/// the tokens the model wrote, ending at the `<|im_end|>` commit that closes
/// its turn. The RENDER holds the same conversation as history, and its version
/// of the generated region differs — the template trims the reasoning, writes
/// the tool-call arguments in its own order, and the tokenizer need not split
/// the text the way the model did. The token 70 below stands for a token that
/// only the model wrote, and 71 for the token the render writes in its place.
struct QwenCommittedTurnRuleTests {

    /// Stands in for the `<|im_end|>` commit token id.
    private static let commit = 1

    /// Stands in for a token that only the model wrote.
    private static let generatedOnly = 70

    /// Stands in for the token the render writes where the model wrote
    /// ``generatedOnly``.
    private static let renderedOnly = 71

    private let rule = QwenCommittedTurnRule(endOfTurnToken: QwenCommittedTurnRuleTests.commit)

    /// A policy wired the way the executor wires it for a Qwen 3.5 model.
    private var policy: PromptCacheReusePolicy {
        PromptCacheReusePolicy(protocolRules: [rule])
    }

    /// The turn that renders the whole conversation again.
    private func turn(prompt: [Int], uncommittedTokens: [Int] = []) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt,
            previousGenerationUncommittedTokens: uncommittedTokens)
    }

    /// A cache whose timeline agrees with its ledger.
    private func alignedCache(
        _ cached: [Int], previousRender: [Int], processed: Int? = nil
    ) -> PromptCacheState {
        let processedTokenCount = processed ?? cached.count
        return PromptCacheState(
            cachedTokens: cached,
            previousRenderTokens: previousRender,
            processedTokenCount: processedTokenCount,
            mainCacheIsAligned: processedTokenCount == cached.count,
            isTrimmable: false)
    }

    // MARK: - Splicing

    @Test func `splices the new tail onto the tokens the model wrote`() {
        // The previous render ended at the generation prompt [10, 11]. The
        // model then wrote [70, commit], which the render writes as [71, commit].
        let decision = rule.reuse(
            turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20, 21]),
            cache: alignedCache(
                [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 4,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20, 21]))
    }

    @Test func `splices at the first commit when the tool response closes with a second one`() {
        // A Qwen tool round writes the assistant's `<|im_end|>` and then the
        // `<|im_end|>` that closes the `<tool_response>` user turn. The
        // assistant's own commit is the first one after the previous render.
        let decision = rule.reuse(
            turn: turn(
                prompt: [10, 11, Self.renderedOnly, Self.commit, 20, 21, Self.commit, 22]),
            cache: alignedCache(
                [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 4,
                    representedTokens: [
                        10, 11, Self.generatedOnly, Self.commit, 20, 21, Self.commit, 22,
                    ]))
    }

    @Test func `splices at the commit this round added, not at an earlier one`() {
        // A later agent round: the render carries the commits of the rounds
        // before it too, and splicing there would feed a completed round again.
        let decision = rule.reuse(
            turn: turn(prompt: [10, Self.commit, 11, Self.renderedOnly, Self.commit, 20]),
            cache: alignedCache(
                [10, Self.commit, 11, Self.generatedOnly, Self.commit],
                previousRender: [10, Self.commit, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 5,
                    representedTokens: [
                        10, Self.commit, 11, Self.generatedOnly, Self.commit, 20,
                    ]))
    }

    @Test func `splices the commit itself when the turn ran out of budget`() {
        // A generation that stopped on the token budget wrote no commit, thus
        // the render closes that turn where the model did not. The suffix
        // carries the commit.
        let decision = rule.reuse(
            turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
            cache: alignedCache([10, 11, Self.generatedOnly], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 3,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20]))
    }

    // MARK: - Declining

    @Test func `declines a render that does not extend the render before it`() {
        // The template rewrote an already-cached rendered region, thus the
        // generation region is no longer the only region that differs.
        #expect(
            rule.reuse(
                turn: turn(prompt: [99, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == nil)
    }

    @Test func `declines when the render adds no commit`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == nil)
    }

    @Test func `declines when nothing follows the commit`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == nil)
    }

    @Test func `declines an uncommitted lookahead the render cannot explain`() {
        #expect(
            rule.reuse(
                turn: turn(
                    prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                    uncommittedTokens: [Self.renderedOnly]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly], previousRender: [10, 11])) == nil)
    }

    @Test func `declines when no render is on record`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [])) == nil)
    }

    @Test func `declines a cache timeline ahead of the ledger`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11],
                    processed: 9)) == nil)
    }

    // MARK: - A turn that ends inside its reasoning

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

    /// The render of a prompt with thinking on: it ends inside an open think
    /// block.
    private static let thinkingOnRender = [conversation, assistant, openThink]

    /// The render of a prompt with thinking off: it ends with a closed, empty
    /// think block.
    private static let thinkingOffRender = [conversation, assistant, openThink, closeThink]

    /// One way the previous turn ended, and what the rule decides for the
    /// render of the next turn (card `^ef2zynk`).
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

    /// The rule as the `qwen35` format makes it from a tokenizer that knows
    /// the end-of-turn marker and the markers of the think block.
    private static let reasoningRule = QwenCommittedTurnRule(
        tokenizer: MarkerTokenizer(identifierOfMarker: [
            QwenCommittedTurnRule.endOfTurnMarker: commit,
            QwenCommittedTurnRule.openThinkMarker: openThink,
            QwenCommittedTurnRule.closeThinkMarker: closeThink,
        ]))

    /// The splice onto a ledger that does not hold the commit: it starts AT
    /// the commit of `prompt`, thus the fed suffix carries the commit.
    private func spliceAtTheCommit(of prompt: [Int], onto cached: [Int]) throws
        -> PromptCacheReuseDecision
    {
        let commitIndex = try #require(prompt.firstIndex(of: Self.commit))
        return .appendSuffix(
            suffixStart: commitIndex, representedTokens: cached + prompt[commitIndex...])
    }

    @Test(arguments: reasoningEndCases)
    func `the splice after a turn that ends`(_ row: ReasoningEndCase) throws {
        let rule = try #require(Self.reasoningRule)

        let decision = rule.reuse(
            turn: turn(prompt: row.prompt),
            cache: alignedCache(row.cached, previousRender: row.previousRender))

        let expected = row.splices ? try spliceAtTheCommit(of: row.prompt, onto: row.cached) : nil
        #expect(decision == expected)
    }

    @Test func `a tokenizer without think markers keeps the splice inside reasoning`() throws {
        // The rule cannot tell where reasoning ends, thus it splices as before.
        let prompt =
            Self.thinkingOnRender + [
                Self.renderedOnly, Self.closeThink, Self.commit, Self.nextTurn,
            ]
        let cached = Self.thinkingOnRender + [Self.generatedOnly]

        let decision = rule.reuse(
            turn: turn(prompt: prompt),
            cache: alignedCache(cached, previousRender: Self.thinkingOnRender))

        #expect(decision == (try spliceAtTheCommit(of: prompt, onto: cached)))
    }

    // MARK: - Composition with the standard rules

    @Test func `a hybrid cache that the rule declines rebuilds`() {
        // The caches of a hybrid model cannot rewind, thus a turn the rule
        // declines has no other way to reuse the prefix.
        #expect(
            policy.decide(
                turn: turn(prompt: [10, 11, Self.renderedOnly, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == .rebuild)
    }

    @Test func `the rule only applies to models that select it`() {
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == .rebuild)
    }

    // MARK: - Selection

    @Test func `the qwen35 format contributes the rule, resolving the end of turn`() {
        let tokenizer = MarkerTokenizer(
            identifierOfMarker: [QwenCommittedTurnRule.endOfTurnMarker: Self.commit])
        let rules = ToolCallFormat.qwen35.promptCacheReuseRules(tokenizer: tokenizer)

        #expect(rules.count == 1)
        #expect((rules.first as? QwenCommittedTurnRule)?.endOfTurnToken == Self.commit)
    }

    @Test func `a tokenizer without the end-of-turn marker contributes no rule`() {
        let tokenizer = MarkerTokenizer(identifierOfMarker: [:])

        #expect(ToolCallFormat.qwen35.promptCacheReuseRules(tokenizer: tokenizer).isEmpty)
        #expect(QwenCommittedTurnRule(tokenizer: tokenizer) == nil)
    }

    @Test func `no other format contributes the Qwen rule`() {
        let tokenizer = MarkerTokenizer(
            identifierOfMarker: [QwenCommittedTurnRule.endOfTurnMarker: Self.commit])
        for format in ToolCallFormat.allCases where format != .qwen35 {
            let rules = format.promptCacheReuseRules(tokenizer: tokenizer)
            #expect(rules.allSatisfy { !($0 is QwenCommittedTurnRule) })
        }
    }
}
