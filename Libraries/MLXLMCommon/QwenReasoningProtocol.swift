// Copyright © 2026 Apple Inc.

/// Qwen-family reasoning protocol declarations.
///
/// Model-specific wire behavior lives here rather than in ``ReasoningConfig``.
/// The generic configuration only describes a protocol; this adapter decides
/// which Qwen families have a validated hard-budget transition.
public enum QwenReasoningProtocol {
    /// Qwen-compatible `<think>` tags and tool-call boundary, without claiming
    /// that a hard-budget transition is safe for the model.
    public static let tagged = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true,
        implicitEndDelimiters: ["<tool_call>"])

    /// The Qwen 3.5 protocol: the tags of ``tagged``, with the reasoning of a
    /// past turn replayed into the history render.
    ///
    /// The Qwen 3.5 chat template keeps the `<think>` block of a past assistant
    /// turn unless a caller sets `preserve_thinking` to false, and it writes
    /// the `reasoning_content` of the message inside that block. A history
    /// render that carries the reasoning holds what the model read while it
    /// wrote, thus the render of a later round extends the tokens the model
    /// wrote, and a prompt cache carries across the round. A hybrid Qwen 3.5
    /// model cannot rewind its recurrent caches, thus that extension is its one
    /// path to reuse. Card `^xx5g893` measured this on
    /// `mlx-community/Qwen3.8-27B-mxfp4`.
    public static let qwen35: ReasoningConfig = {
        var config = tagged
        config.replaysReasoningIntoHistory = true
        return config
    }()

    /// The original hybrid Qwen3 protocol and its published budget transition.
    ///
    /// The exact leading space after the two newlines is part of Qwen's
    /// `early_stopping_text`. Keeping this adapter family-specific prevents a
    /// later Qwen-derived model from inheriting the transition merely because it
    /// happens to use the same delimiters.
    public static let qwen3 = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true,
        implicitEndDelimiters: ["<tool_call>"],
        budgetTransition: ReasoningBudgetTransition(
            beforeEndDelimiter:
                "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n",
            afterEndDelimiter: "\n\n"))

}
