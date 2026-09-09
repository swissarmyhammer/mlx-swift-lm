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
@Suite
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
