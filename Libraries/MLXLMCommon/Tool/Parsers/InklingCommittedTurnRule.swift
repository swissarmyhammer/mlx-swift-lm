// Copyright © 2026 Apple Inc.

/// Cache-reuse rule for an Inkling 2 turn that continues a committed assistant turn.
///
/// The live caches hold the frames the model wrote, and a render of the same turn as history
/// cannot always write them again. The published template writes each tool call as compact
/// JSON with sorted keys (`{"name":"f","args":{"a":1}}`), while the model writes its own
/// spacing and key order, and a BPE tokenizer need not split the text the model wrote the way
/// the model did. The two token streams then part inside the turn of the model, and Inkling 2
/// is a hybrid model whose recurrent caches cannot rewind to the point where they part.
///
/// This rule keeps the live trajectory instead, through ``CommittedTurnSplice``, and feeds only
/// the tokens the new render adds after the `<|content_model_end_sampling|>` marker that ends
/// the last turn of the model: the tool results of a tool round, or the next user message.
///
/// Every frame of the model ends at `<|end_message|>`, and a turn of the model ends at one
/// `<|content_model_end_sampling|>` after all its frames, thus the parallel calls of a turn
/// share one commit. The commit of the model is the FIRST one after the render before it,
/// because the model cannot write the marker inside its turn: generation stops on it.
///
/// The rule declines a turn whose written tokens end inside an open thinking frame, as
/// ``CommittedTurnSplice`` states: the render closes that frame before the commit.
///
/// It is the only place in the cache pipeline that knows the Inkling end-of-turn marker and the
/// Inkling thinking markers.
struct InklingCommittedTurnRule: PromptCacheReuseRule {

    /// The text of the marker that ends a turn of the model, which is also the token the
    /// model writes to end its own turn.
    static let endOfTurnMarker = "<|content_model_end_sampling|>"

    /// The text of the marker that opens a thinking frame.
    static let openThinkMarker = "<|content_thinking|>"

    /// The text of the marker that ends every frame, and thus a thinking frame.
    static let closeThinkMarker = "<|end_message|>"

    /// The splice this rule makes, at the first commit after the render before it.
    let splice: CommittedTurnSplice

    /// A rule that knows no thinking markers, thus it splices every turn.
    ///
    /// - Parameter endOfTurnToken: the identifier of the marker that ends a turn of the model.
    init(endOfTurnToken: Int) {
        self.splice = CommittedTurnSplice(
            endOfTurnToken: endOfTurnToken, commitSelection: .first, thinkBlock: nil)
    }

    /// Fails when the tokenizer has no Inkling end-of-turn marker, in which case the model is
    /// not running the Inkling protocol and the standard rules apply unchanged.
    ///
    /// - Parameter tokenizer: the tokenizer that names the markers.
    init?(tokenizer: any Tokenizer) {
        guard let token = tokenizer.convertTokenToId(Self.endOfTurnMarker) else {
            return nil
        }
        self.splice = CommittedTurnSplice(
            endOfTurnToken: token, commitSelection: .first,
            thinkBlock: CommittedTurnSplice.ThinkBlockTokens(
                tokenizer: tokenizer, open: Self.openThinkMarker, close: Self.closeThinkMarker))
    }

    /// Splices the new tail onto a cache that ends at a committed turn of the model.
    ///
    /// ``CommittedTurnSplice`` states the conditions the splice proves.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the splice, or `nil` to leave the turn to the next rule.
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        splice.decision(turn: turn, cache: cache)
    }
}
