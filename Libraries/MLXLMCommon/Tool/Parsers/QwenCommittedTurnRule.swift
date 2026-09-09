// Copyright © 2026 Apple Inc.

/// Cache-reuse rule for a Qwen 3.5 turn that continues a committed assistant
/// turn.
///
/// The live KV cache holds the tokens the model wrote, and a cold chat-template
/// render cannot always reproduce them. Card `^xx5g893` read four reasons in
/// the Qwen 3.5 chat template and in the transcript conversion:
///
/// 1. The template trims `reasoning_content` and `content`, thus whitespace
///    the model wrote at either end of its reasoning or its answer is gone.
/// 2. The template writes the tool-call arguments in the order of a
///    dictionary, thus a call with two or more arguments can come back in
///    another order than the model wrote.
/// 3. A value that is not a string goes through `tojson`, thus it comes back
///    canonical whatever the model wrote.
/// 4. The tokens a model writes are not always the canonical tokenization of
///    the text they decode to, which card `^v7z7v99` measured on DeepSeek-V4.
///
/// None of those can be corrected by a render. This rule keeps the live
/// trajectory instead, through ``CommittedTurnSplice``, and feeds only the
/// tokens the new render adds after the `<|im_end|>` marker that closes the
/// last assistant turn. A hybrid Qwen 3.5 model cannot rewind its recurrent
/// caches, thus this splice is the one path that carries a prompt cache across
/// a tool round whose render differs from the tokens the model wrote.
///
/// A Qwen tool round writes TWO `<|im_end|>` after the previous render: the
/// assistant's own, then the one that closes the `<tool_response>` user turn.
/// The assistant's own commit is the FIRST one, because the model's turn cannot
/// hold the marker — generation stops on it — and every later one belongs to a
/// later message.
///
/// It is the only place in the cache pipeline that knows the Qwen end-of-turn
/// marker.
struct QwenCommittedTurnRule: PromptCacheReuseRule {

    /// The text of the marker that closes an assistant turn in a rendered Qwen
    /// conversation, which is also the token the model writes to end its own
    /// turn.
    static let endOfTurnMarker = "<|im_end|>"

    /// The identifier of ``endOfTurnMarker`` in the model's tokenizer.
    let endOfTurnToken: Int

    /// - Parameter endOfTurnToken: the identifier of the marker that closes an
    ///   assistant turn.
    init(endOfTurnToken: Int) {
        self.endOfTurnToken = endOfTurnToken
    }

    /// Fails when the tokenizer has no Qwen end-of-turn marker, in which case
    /// the model is not running the Qwen protocol and the standard rules apply
    /// unchanged.
    ///
    /// - Parameter tokenizer: the tokenizer that names the marker.
    init?(tokenizer: any Tokenizer) {
        guard let token = tokenizer.convertTokenToId(Self.endOfTurnMarker) else {
            return nil
        }
        self.endOfTurnToken = token
    }

    /// Splices the new tail onto a cache that ends at a committed assistant
    /// turn, at the first commit the render adds after the render before it.
    ///
    /// ``CommittedTurnSplice`` states the conditions the splice proves.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of this turn.
    ///   - cache: what the caches currently hold.
    /// - Returns: the splice, or `nil` to leave the turn to the next rule.
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        CommittedTurnSplice(endOfTurnToken: endOfTurnToken, commitSelection: .first)
            .decision(turn: turn, cache: cache)
    }
}
