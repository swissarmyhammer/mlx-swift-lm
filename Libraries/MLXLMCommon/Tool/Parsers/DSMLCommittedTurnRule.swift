// Copyright © 2026 Apple Inc.

/// Cache-reuse rule for a DeepSeek-V4 turn that continues a committed assistant
/// turn.
///
/// The live KV cache holds the tokens the model wrote, and a cold chat-template
/// render cannot reproduce them. Card `^v7z7v99` measured two reasons on
/// `mlx-community/DeepSeek-V4-Flash-4bit`:
///
/// 1. The model abbreviates its own DSML closing tags. It writes
///    `</｜DSML｜inv>` where the syntax states `</｜DSML｜invoke>`, and in chat
///    mode it also writes `</｜DSML｜tool>` for `</｜DSML｜tool_calls>`. The
///    abbreviation belongs to the bytes the model wrote, thus a render that
///    rebuilds the block from a parsed call cannot write it again.
/// 2. The tokens a model writes are not always the canonical tokenization of the
///    text they decode to. The measured run shows the model writing ` pal` +
///    `les` where the tokenizer encodes the same text as ` pall` + `es`, thus
///    even a render of the model's exact text cannot write its exact tokens.
///
/// The two streams therefore stop being prefix-equivalent inside the region the
/// model generated, and the generic prefix rules would compare streams that are
/// not comparable. This rule claims such a turn instead. It keeps the live
/// trajectory, which is the text the model itself read while it wrote, and it
/// feeds only the tokens the new render adds after the
/// ``DeepSeekV4ChatEncoder/SpecialToken/endOfSentence`` marker that closes the
/// last assistant turn.
///
/// It differs from ``HarmonyToolRestartRule`` and ``OnyxToolRestartRule`` in one
/// way. Those two claim a tool-result continuation alone. This one claims any
/// turn that follows a committed assistant turn, because the round AFTER a
/// DeepSeek-V4 tool round is an ordinary user turn whose preceding assistant
/// turn is the unrenderable one.
///
/// It is the only place in the cache pipeline that knows about DSML.
struct DSMLCommittedTurnRule: PromptCacheReuseRule {

    /// The marker that closes an assistant turn in a rendered DeepSeek-V4
    /// conversation, which is also the token the model writes to end its own
    /// turn.
    let endOfSentenceToken: Int

    /// - Parameter endOfSentenceToken: the identifier of the marker that closes
    ///   an assistant turn.
    init(endOfSentenceToken: Int) {
        self.endOfSentenceToken = endOfSentenceToken
    }

    /// Fails when the tokenizer has no DeepSeek-V4 end-of-sentence marker, in
    /// which case the model is not running the DSML protocol and the standard
    /// rules apply unchanged.
    ///
    /// - Parameter tokenizer: the tokenizer that names the marker.
    init?(tokenizer: any Tokenizer) {
        guard
            let token = tokenizer.convertTokenToId(
                DeepSeekV4ChatEncoder.SpecialToken.endOfSentence)
        else {
            return nil
        }
        self.endOfSentenceToken = token
    }

    /// Splices the new tail onto a cache that ends at a committed assistant
    /// turn.
    ///
    /// The rule declines every state it cannot prove:
    ///
    /// - The new render must hold the render of the previous prefill as a WHOLE
    ///   prefix. That proves the template rewrote no already-cached rendered
    ///   region, thus the generation region is the only region left that the two
    ///   can differ in. A render that DOES rewrite an earlier region — a
    ///   conversation that drops the reasoning of an earlier turn, or a changed
    ///   system prompt — falls through to the standard rules.
    /// - The ledger must end where the render's commit stands. A ledger that
    ///   already holds the commit is spliced after it. A ledger that does not —
    ///   a generation that stopped on the token budget, or a speculative
    ///   iterator that returned the commit before the cache represented it — is
    ///   spliced AT the commit, thus the suffix carries the commit and the model
    ///   reads its own unterminated answer in front of it. Any other lookahead
    ///   the render cannot explain is declined.
    /// - The render must add EXACTLY ONE commit after the render before it. The
    ///   cache holds one generation, thus one commit. A second one comes from
    ///   message text — a tool result that quotes the marker — and it must not
    ///   move the splice.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the splice, or `nil` to leave the turn to the next rule.
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard !cache.previousRenderTokens.isEmpty,
            !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            turn.promptTokens.starts(with: cache.previousRenderTokens),
            let commitIndex = soleCommitIndex(
                in: turn.promptTokens, after: cache.previousRenderTokens.count),
            let suffixStart = suffixStart(at: commitIndex, of: turn, cache: cache),
            suffixStart < turn.promptTokens.endIndex
        else {
            return nil
        }

        let representedTokens = cache.cachedTokens + turn.promptTokens[suffixStart...]
        let canContinueWithDraft =
            !turn.usesSpeculativeDecoding
            || (cache.hasDraftCache && cache.draftCacheIsAligned)
        if canContinueWithDraft {
            return .appendSuffix(suffixStart: suffixStart, representedTokens: representedTokens)
        }
        // The draft cache cannot follow the live trajectory. Preserve the
        // authoritative main cache and use it alone for this continuation.
        return .appendSuffixToMain(
            suffixStart: suffixStart, representedTokens: representedTokens)
    }

    /// The index of the one commit the new render adds after the render before
    /// it.
    ///
    /// - Parameters:
    ///   - promptTokens: the whole new render.
    ///   - start: the length of the render of the previous prefill.
    /// - Returns: the index, or `nil` when the render adds no commit, or more
    ///   than one.
    private func soleCommitIndex(in promptTokens: [Int], after start: Int) -> Int? {
        let commits = promptTokens.indices.dropFirst(start)
            .filter { promptTokens[$0] == endOfSentenceToken }
        guard commits.count == 1 else { return nil }
        return commits.first
    }

    /// Where the fed suffix of the new render begins.
    ///
    /// - Parameters:
    ///   - commitIndex: the index of the commit in the new render.
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the index, or `nil` when the previous generation left tokens
    ///   the render cannot explain.
    private func suffixStart(
        at commitIndex: Int, of turn: PromptCacheTurn, cache: PromptCacheState
    ) -> Int? {
        let uncommitted = turn.previousGenerationUncommittedTokens
        if cache.cachedTokens.last == endOfSentenceToken, uncommitted.isEmpty {
            // The model closed its turn and the cache holds that commit, thus
            // the suffix begins after it.
            return commitIndex + 1
        }
        if uncommitted.isEmpty || uncommitted == [endOfSentenceToken] {
            // The cache does not hold the commit: the generation stopped on the
            // token budget, or a speculative iterator returned the commit as its
            // final verifier sample before the cache represented it. The suffix
            // carries the commit either way.
            return commitIndex
        }
        return nil
    }
}
