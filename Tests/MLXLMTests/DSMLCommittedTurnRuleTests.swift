// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// Unit tests for the DSML cache-reuse rule of card `^v7z7v99`. The rule is
/// pure, thus the splice contract is examined without a model and without a
/// tokenizer.
///
/// Each fixture below writes the shape the real weights write. The LEDGER holds
/// the tokens the model wrote, ending at the commit that closes its turn. The
/// RENDER holds the same conversation as history, and its version of the
/// generated region differs — the measured checkpoint abbreviates its own DSML
/// closing tags and splits ordinary words into other tokens. The token 70 below
/// stands for a token that only the model wrote, and 71 for the token the render
/// writes in its place.
@Suite
struct DSMLCommittedTurnRuleTests {

    /// Stands in for the `<｜end▁of▁sentence｜>` commit token id.
    private static let commit = 1

    /// Stands in for a token that only the model wrote.
    private static let generatedOnly = 70

    /// Stands in for the token the render writes where the model wrote
    /// ``generatedOnly``.
    private static let renderedOnly = 71

    private let rule = DSMLCommittedTurnRule(
        endOfSentenceToken: DSMLCommittedTurnRuleTests.commit)

    /// A policy wired the way `ChatSession` wires it for a DeepSeek-V4 model.
    private var policy: PromptCacheReusePolicy {
        PromptCacheReusePolicy(protocolRules: [rule])
    }

    /// The turn that renders the whole conversation again.
    ///
    /// - Parameters:
    ///   - prompt: the whole new render.
    ///   - toolResultContinuation: whether this turn appends tool results.
    ///   - attentionMask: whether the prepared input carries a mask.
    ///   - newMedia: whether this turn introduces media.
    ///   - uncommittedTokens: tokens the previous generation returned but did
    ///     not commit.
    ///   - usesSpeculativeDecoding: whether a draft model takes part.
    /// - Returns: the turn.
    private func turn(
        prompt: [Int],
        toolResultContinuation: Bool = true,
        attentionMask: Bool = false,
        newMedia: Bool = false,
        uncommittedTokens: [Int] = [],
        usesSpeculativeDecoding: Bool = false
    ) -> PromptCacheTurn {
        PromptCacheTurn(
            promptTokens: prompt,
            carriesNewMedia: newMedia,
            carriesAttentionMask: attentionMask,
            isToolResultContinuation: toolResultContinuation,
            previousGenerationUncommittedTokens: uncommittedTokens,
            usesSpeculativeDecoding: usesSpeculativeDecoding)
    }

    /// A cache whose timeline agrees with its ledger.
    ///
    /// - Parameters:
    ///   - cached: the tokens the cache represents.
    ///   - previousRender: the render of the previous prefill.
    ///   - processed: the timeline of the cache, when it differs from the ledger.
    ///   - hasDraft: whether a live draft cache exists.
    ///   - draftAligned: whether the draft agrees with the ledger.
    /// - Returns: the cache state.
    private func alignedCache(
        _ cached: [Int],
        previousRender: [Int],
        processed: Int? = nil,
        hasDraft: Bool = false,
        draftAligned: Bool = true
    ) -> PromptCacheState {
        let processedTokenCount = processed ?? cached.count
        return PromptCacheState(
            cachedTokens: cached,
            previousRenderTokens: previousRender,
            processedTokenCount: processedTokenCount,
            mainCacheIsAligned: processedTokenCount == cached.count,
            hasDraftCache: hasDraft,
            draftCacheIsAligned: draftAligned,
            isTrimmable: true)
    }

    // MARK: - Splicing

    @Test func `splices the new tail onto the tokens the model wrote`() {
        // The previous render ended at the generation tail [10, 11]. The model
        // then wrote [70, commit], which the render writes as [71, commit].
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

    @Test func `splices an ordinary user turn as well as a tool result`() {
        // The round AFTER a tool round is an ordinary user turn, and its
        // preceding assistant turn is the unrenderable one.
        let decision = rule.reuse(
            turn: turn(
                prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                toolResultContinuation: false),
            cache: alignedCache(
                [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 4,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20]))
    }

    @Test func `splices an uncommitted speculative commit before the new tail`() {
        // A speculative iterator returns its final verifier sample before the
        // cache represents it, thus the commit itself joins the suffix.
        let decision = rule.reuse(
            turn: turn(
                prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                uncommittedTokens: [Self.commit]),
            cache: alignedCache([10, 11, Self.generatedOnly], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 3,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20]))
    }

    @Test func `splices the commit itself when the turn ran out of budget`() {
        // A generation that stopped on the token budget wrote no commit, thus
        // the render closes that turn where the model did not. The suffix
        // carries the commit, and the model reads its own unterminated answer in
        // front of it.
        let decision = rule.reuse(
            turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
            cache: alignedCache([10, 11, Self.generatedOnly], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffix(
                    suffixStart: 3,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20]))
    }

    @Test func `splices at the commit this round added, not at an earlier one`() {
        // A second agent round: the render carries the commit of the round
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

    @Test func `keeps the main cache when the speculative draft trails it`() {
        let decision = rule.reuse(
            turn: turn(
                prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                usesSpeculativeDecoding: true),
            cache: alignedCache(
                [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11],
                hasDraft: true, draftAligned: false))

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: 4,
                    representedTokens: [10, 11, Self.generatedOnly, Self.commit, 20]))
    }

    @Test func `keeps using the main cache when a later round has no draft`() {
        // A previous main-only splice discarded the divergent draft. The next
        // round must stay on the main-cache path rather than rebuild from a
        // render that cannot reproduce the generated region.
        let decision = rule.reuse(
            turn: turn(
                prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                usesSpeculativeDecoding: true),
            cache: alignedCache(
                [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))

        #expect(
            decision
                == .appendSuffixToMain(
                    suffixStart: 4,
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

    @Test func `declines an uncommitted lookahead the render cannot explain`() {
        // The parser saw tokens the cache does not hold, and they are not the
        // commit. The render of the turn holds them somewhere inside its own
        // body, thus no boundary can be proven.
        #expect(
            rule.reuse(
                turn: turn(
                    prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                    uncommittedTokens: [Self.renderedOnly]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly], previousRender: [10, 11])) == nil)
    }

    @Test func `declines when message text adds a second commit`() {
        // A tool result that quotes the end-of-sentence marker must not move the
        // splice.
        #expect(
            rule.reuse(
                turn: turn(
                    prompt: [10, 11, Self.renderedOnly, Self.commit, 20, Self.commit, 21]),
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

    @Test func `declines a commit that the render before it already held`() {
        // The one commit of this render lies inside the render before it, thus
        // it is the boundary of an earlier round and not of this one.
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, Self.commit, 11, Self.renderedOnly, 20]),
                cache: alignedCache(
                    [10, Self.commit, 11, Self.generatedOnly, Self.commit],
                    previousRender: [10, Self.commit, 11])) == nil)
    }

    @Test func `declines when nothing follows the commit`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == nil)
    }

    @Test func `declines when no render is on record`() {
        // A cache restored from a snapshot has a ledger and no render behind it.
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [])) == nil)
    }

    @Test func `declines an invalidated ledger`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: PromptCacheState(
                    cachedTokens: [], previousRenderTokens: [10, 11],
                    processedTokenCount: 8)) == nil)
    }

    @Test func `declines a cache timeline ahead of the ledger`() {
        #expect(
            rule.reuse(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11],
                    processed: 9)) == nil)
    }

    @Test(arguments: [true, false])
    func `declines inputs the splice cannot represent`(mediaRatherThanMask: Bool) {
        #expect(
            rule.reuse(
                turn: turn(
                    prompt: [10, 11, Self.renderedOnly, Self.commit, 20],
                    attentionMask: !mediaRatherThanMask,
                    newMedia: mediaRatherThanMask),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == nil)
    }

    // MARK: - Composition with the standard rules

    @Test func `a declined turn falls through to the standard rules`() {
        // No commit to anchor on, thus the render is reconciled by prefix
        // comparison, which rewinds to where the two streams part.
        #expect(
            policy.decide(
                turn: turn(prompt: [10, 11, Self.renderedOnly, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 2))
    }

    @Test func `the rule only applies to models that select it`() {
        // Without the DSML rule the same turn is resolved by prefix comparison
        // of two streams that are not comparable, which is why the rule exists.
        #expect(
            PromptCacheReusePolicy().decide(
                turn: turn(prompt: [10, 11, Self.renderedOnly, Self.commit, 20]),
                cache: alignedCache(
                    [10, 11, Self.generatedOnly, Self.commit], previousRender: [10, 11]))
                == .trimToCommonPrefix(commonPrefixLength: 2, trimCount: 2))
    }

    // MARK: - Selection

    @Test func `the dsml format contributes the rule, resolving the commit`() {
        let tokenizer = MarkerTokenizer(
            identifierOfMarker: [
                DeepSeekV4ChatEncoder.SpecialToken.endOfSentence: Self.commit
            ])
        let rules = ToolCallFormat.dsml.promptCacheReuseRules(tokenizer: tokenizer)

        #expect(rules.count == 1)
        #expect((rules.first as? DSMLCommittedTurnRule)?.endOfSentenceToken == Self.commit)
    }

    @Test func `a tokenizer without the end-of-sentence marker contributes no rule`() {
        // The model stays on the standard path instead of trapping.
        let tokenizer = MarkerTokenizer(identifierOfMarker: [:])

        #expect(ToolCallFormat.dsml.promptCacheReuseRules(tokenizer: tokenizer).isEmpty)
        #expect(DSMLCommittedTurnRule(tokenizer: tokenizer) == nil)
    }

    @Test func `no other plain-text format contributes the DSML rule`() {
        let tokenizer = MarkerTokenizer(
            identifierOfMarker: [
                DeepSeekV4ChatEncoder.SpecialToken.endOfSentence: Self.commit
            ])
        for format in ToolCallFormat.allCases where format != .dsml {
            let rules = format.promptCacheReuseRules(tokenizer: tokenizer)
            #expect(rules.allSatisfy { !($0 is DSMLCommittedTurnRule) })
        }
    }
}

/// A tokenizer that resolves the marker texts it is given and nothing else,
/// which is all the failable initializer of the rule reads.
private struct MarkerTokenizer: Tokenizer {
    /// The identifier of each marker text this tokenizer knows.
    let identifierOfMarker: [String: Int]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { identifierOfMarker[token] }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
