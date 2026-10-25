// Copyright © 2026 Apple Inc.

/// The splice a committed-turn rule makes onto the tokens the model wrote.
///
/// A committed-turn rule serves a model whose chat-template render cannot write
/// the tokens the model generated: DeepSeek-V4 abbreviates its DSML closing
/// tags, Qwen 3.5 gets its reasoning trimmed and its tool-call arguments
/// reordered, and any BPE model can split a word into tokens the tokenizer does
/// not choose. The two streams then stop being prefix-equivalent inside the
/// generated region, and the generic prefix rules would compare streams that
/// are not comparable.
///
/// This splice keeps the live trajectory, which is the text the model itself
/// read while it wrote, and it feeds only the tokens the new render adds after
/// the commit that closes the last assistant turn. ``DSMLCommittedTurnRule``
/// and ``QwenCommittedTurnRule`` differ only in which commit of the render
/// closes that turn and in the markers they resolve, thus they share this one
/// body.
///
/// The splice declines a turn whose cached tokens end inside an open think
/// block: the generation stopped inside its reasoning, and the model wrote no
/// close of the block. The render of that turn closes the block before the
/// commit, and a splice would make the model read its unclosed reasoning and
/// then the commit. The standard rules then decide: they extend the cached
/// tokens with the render, or a hybrid model goes back to the checkpoint at
/// the end of the previous prompt. Card `^ef2zynk` measured that on Qwen 3.8,
/// and card `^mf1mwaw` on DeepSeek-V4.
struct CommittedTurnSplice: Sendable {

    /// The identifiers of the two markers of a think block.
    struct ThinkBlockTokens: Sendable {
        /// The identifier of the marker that opens a think block.
        let open: Int

        /// The identifier of the marker that closes a think block.
        let close: Int

        /// Resolves the two markers of a think block.
        ///
        /// - Parameters:
        ///   - tokenizer: the tokenizer that names the markers.
        ///   - openMarker: the text of the marker that opens a think block.
        ///   - closeMarker: the text of the marker that closes a think block.
        /// - Returns: `nil` when the tokenizer has no token for one of the
        ///   markers.
        init?(tokenizer: any Tokenizer, open openMarker: String, close closeMarker: String) {
            guard let open = tokenizer.convertTokenToId(openMarker),
                let close = tokenizer.convertTokenToId(closeMarker)
            else {
                return nil
            }
            self.open = open
            self.close = close
        }

        /// Whether `tokens` end inside an open think block: the last marker of
        /// a think block in `tokens` opens one.
        ///
        /// - Parameter tokens: the tokens the caches hold.
        /// - Returns: `true` when no close follows the last open.
        func endsInsideBlock(_ tokens: [Int]) -> Bool {
            tokens.last { $0 == open || $0 == close } == open
        }
    }

    /// Which commit of the new render closes the turn the caches hold.
    enum CommitSelection: Sendable {
        /// The render must add exactly one commit after the render before it.
        /// A second one comes from message text that quotes the marker, and it
        /// must not move the splice. DSML renders one commit for each turn.
        case sole

        /// The first commit after the render before it. A Qwen tool round
        /// writes the assistant's `<|im_end|>` and then the `<|im_end|>` that
        /// closes the tool response, thus the first one is the model's own.
        case first
    }

    /// The marker that closes an assistant turn in a rendered conversation,
    /// which is also the token the model writes to end its own turn.
    let endOfTurnToken: Int

    /// Which commit of the new render closes the turn the caches hold.
    let commitSelection: CommitSelection

    /// The identifiers of the think-block markers, or nil when the tokenizer
    /// has none. Without them the splice cannot see where reasoning ends, thus
    /// it splices every turn.
    let thinkBlock: ThinkBlockTokens?

    /// Splices the new tail onto a cache that ends at a committed assistant
    /// turn.
    ///
    /// The splice declines every state it cannot prove:
    ///
    /// - The caches must not end inside an open think block.
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
    /// - The render must add the commit ``commitSelection`` names after the
    ///   render before it.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the splice, or `nil` to leave the turn to the next rule.
    func decision(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard thinkBlock?.endsInsideBlock(cache.cachedTokens) != true,
            !cache.previousRenderTokens.isEmpty,
            !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            turn.promptTokens.starts(with: cache.previousRenderTokens),
            let commitIndex = commitIndex(
                in: turn.promptTokens, after: cache.previousRenderTokens.count),
            let suffixStart = suffixStart(at: commitIndex, of: turn, cache: cache),
            suffixStart < turn.promptTokens.endIndex
        else {
            return nil
        }

        let representedTokens = cache.cachedTokens + turn.promptTokens[suffixStart...]
        let canContinueWithDraft =
            !turn.usesSpeculativeDecoding
            || cache.speculativeReuseCapability.canContinueWithoutRebuild
        if canContinueWithDraft {
            return .appendSuffix(suffixStart: suffixStart, representedTokens: representedTokens)
        }
        // The draft cache cannot follow the live trajectory. Preserve the
        // authoritative main cache and use it alone for this continuation.
        return .appendSuffixToMain(
            suffixStart: suffixStart, representedTokens: representedTokens)
    }

    /// The index of the commit that closes the turn, among the commits the new
    /// render adds after the render before it.
    ///
    /// - Parameters:
    ///   - promptTokens: the whole new render.
    ///   - start: the length of the render of the previous prefill.
    /// - Returns: the index, or `nil` when the render adds no commit, or when
    ///   ``commitSelection`` is ``CommitSelection/sole`` and it adds more than
    ///   one.
    private func commitIndex(in promptTokens: [Int], after start: Int) -> Int? {
        let commits = promptTokens.indices.dropFirst(start)
            .filter { promptTokens[$0] == endOfTurnToken }
        switch commitSelection {
        case .sole:
            guard commits.count == 1 else { return nil }
            return commits.first
        case .first:
            return commits.first
        }
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
        if cache.cachedTokens.last == endOfTurnToken, uncommitted.isEmpty {
            // The model closed its turn and the cache holds that commit, thus
            // the suffix begins after it.
            return commitIndex + 1
        }
        if uncommitted.isEmpty || uncommitted == [endOfTurnToken] {
            // The cache does not hold the commit: the generation stopped on the
            // token budget, or a speculative iterator returned the commit as its
            // final verifier sample before the cache represented it. The suffix
            // carries the commit either way.
            return commitIndex
        }
        return nil
    }
}
