// Copyright © 2026 Apple Inc.

/// Inkling-family reasoning protocol declarations.
///
/// Inkling models reason inside a `<|content_thinking|>` message frame and
/// leave it at `<|end_message|>`, the boundary that ends every frame. As in
/// ``OnyxReasoningProtocol``, the frame syntax is declared once and each family
/// adds only the transition it was measured with.
///
/// The template reads the thinking effort from `reasoning_effort`, a level name
/// or a number and not a Boolean. Thinking off is the level `none`; thinking on
/// keeps the default level of the template, or the level the caller set. The
/// template writes the reasoning of every past assistant message in its own
/// thinking frame, thus the history replays it.
public enum InklingReasoningProtocol {
    /// The template variable that sets the thinking effort.
    public static let effortKey = "reasoning_effort"

    /// The effort level that turns thinking off.
    public static let thinkingOffEffort = "none"

    /// The thinking frame, without claiming that a hard-budget transition is
    /// safe for the model.
    public static let frames = ReasoningConfig(
        startDelimiter: "<|content_thinking|>", endDelimiter: "<|end_message|>",
        promptStrategy: .templateFlag(
            key: effortKey, defaultOn: true, thinkingOff: .templateValue(thinkingOffEffort)),
        isSpecialToken: true,
        replaysReasoningIntoHistory: true)

    /// Inkling 2, which answers from a frame that was only closed.
    ///
    /// `<|end_message|>` ends the frame without ending the turn, so the model
    /// opens its own `<|content_text|>` frame next and closing the frame is the
    /// whole transition. Measured on `Inkling-2-Mini-4bit`: over 20 forced cuts
    /// the model always answered the user and never resumed reasoning.
    public static let inkling2 = frames.declaringBudgetTransition(.immediate)
}
