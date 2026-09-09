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
/// not comparable. This rule claims such a turn instead, through
/// ``CommittedTurnSplice``: it keeps the live trajectory and feeds only the
/// tokens the new render adds after the
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
    /// DSML renders one commit for each turn, thus the render must add EXACTLY
    /// ONE commit after the render before it. A second one comes from message
    /// text — a tool result that quotes the marker — and it must not move the
    /// splice. ``CommittedTurnSplice`` states the other conditions.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the splice, or `nil` to leave the turn to the next rule.
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        CommittedTurnSplice(endOfTurnToken: endOfSentenceToken, commitSelection: .sole)
            .decision(turn: turn, cache: cache)
    }
}
