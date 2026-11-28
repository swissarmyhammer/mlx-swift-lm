// Copyright © 2026 Apple Inc.

/// Qwen-family reasoning protocol declarations.
///
/// Model-specific wire behavior lives here rather than in ``ReasoningConfig``.
/// The generic configuration only describes a protocol; this adapter decides
/// which Qwen families have a validated hard-budget transition. Each of those
/// is derived from ``tagged`` through
/// ``ReasoningConfig/declaringBudgetTransition(_:)``, so the delimiters have a
/// single definition and a family can only differ in what was measured for it.
public enum QwenReasoningProtocol {
    /// Qwen-compatible `<think>` tags and tool-call boundary, without claiming
    /// that a hard-budget transition is safe for the model.
    public static let tagged = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: true,
        implicitEndDelimiters: ["<tool_call>"])

    /// The Qwen 3.5 protocol: the tags of ``tagged``, with the reasoning of a
    /// past turn replayed into the history render, and thinking off rendered
    /// as a closed, empty think block.
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
    ///
    /// The same template writes its reasoning instructions into the system
    /// block only when `enable_thinking` is undefined or true. A turn with the
    /// flag `false` thus changes the prompt near its start, and the hybrid model
    /// fills its caches again from the start, one time when thinking goes off
    /// and one time when it comes back on. A turn with thinking off therefore
    /// keeps the flag on and ends its prompt with the block that the template
    /// writes for the flag `false` (``ThinkingOffRender/closedBlock(_:)``).
    /// Card `^v3dt28p` measured the two full prefills.
    ///
    /// A named level goes to the template as `reasoning_effort`
    /// (``ReasoningConfig/effortKey``).
    ///
    /// Qwen3.5 answers from a span that was only closed, thus its hard-budget
    /// transition is ``ReasoningBudgetTransition/immediate``. Qwen publishes no
    /// `early_stopping_text` for this generation, and forcing Qwen3's changed
    /// nothing: over 20 forced cuts on `Qwen3.5-9B-8bit` both closing the span
    /// alone and adding Qwen3's text produced a well-formed answer every time,
    /// while unbounded reasoning ran past the generation limit and answered
    /// almost never.
    public static let qwen35: ReasoningConfig = {
        var config = tagged.declaringBudgetTransition(.immediate)
        config.replaysReasoningIntoHistory = true
        config.promptStrategy = .templateFlag(
            key: thinkingFlag, defaultOn: true, thinkingOff: .closedBlock(closedThinkBlock))
        config.effortKey = effortKey
        return config
    }()

    /// The template variable that takes a named reasoning effort level.
    ///
    /// The Qwen3.8 template reads it, for example `low`, and writes "Reasoning effort is set
    /// to low." in its system block; with no value it writes its default `xhigh`. The Qwen
    /// 3.5 template does not read it, thus a level changes nothing in its render.
    private static let effortKey = "reasoning_effort"

    /// The chat template flag that turns Qwen thinking on and off.
    private static let thinkingFlag = "enable_thinking"

    /// The closed, empty think block that the Qwen 3.5 chat template writes
    /// after its generation prompt when `enable_thinking` is false.
    private static let closedThinkBlock = "<think>\n\n</think>\n\n"

    /// The original hybrid Qwen3 protocol and its published budget transition.
    ///
    /// The exact leading space after the two newlines is part of Qwen's
    /// `early_stopping_text`. Keeping this adapter family-specific prevents a
    /// later Qwen-derived model from inheriting the transition merely because it
    /// happens to use the same delimiters.
    public static let qwen3 = tagged.declaringBudgetTransition(
        ReasoningBudgetTransition(
            beforeEndDelimiter:
                "\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n",
            afterEndDelimiter: "\n\n"))
}
