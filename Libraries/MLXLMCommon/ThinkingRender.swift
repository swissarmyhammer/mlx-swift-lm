// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The thinking state that one render of a prompt carries into the chat
/// template.
///
/// A ``ReasoningPromptStrategy`` tells how a thinking on or off preference
/// reaches the template. A strategy that renders thinking off as a flag passes
/// the flag through the template variables. A strategy that renders thinking
/// off as a closed block (``ThinkingOffRender/closedBlock(_:)``, Qwen 3.5) keeps
/// the flag on, and ``prepare(_:processor:tokenizer:)`` puts the closed block in
/// place of the open block at the end of the render. The start of the prompt is
/// then the same with thinking on and off, and a prompt cache carries across
/// the turns.
///
/// ``ChatSession`` and the executor of `MLXFoundationModels` render each prompt
/// through this type. A caller that prepares a prompt itself can also use it.
public struct ThinkingRender: Sendable {

    /// The template variables that state the thinking flag, or nil when the
    /// render states none.
    public let additionalContext: [String: any Sendable]?

    /// The protocol and the closed block that ends the prompt, or nil when the
    /// render keeps the prompt as the template writes it.
    private let closing: (config: ReasoningConfig, block: String)?

    /// A render that states no thinking flag and closes no block.
    public init() {
        additionalContext = nil
        closing = nil
    }

    /// The render of a "thinking enabled" preference under `config`.
    ///
    /// - Parameters:
    ///   - config: the reasoning protocol of the model.
    ///   - thinkingEnabled: `true` / `false` to force thinking on / off,
    ///     `nil` when the caller expressed no preference. With `nil`, the
    ///     render states the default of the strategy.
    /// - Throws: ``ReasoningError/cannotDisableReasoning`` when `false` is
    ///   requested on a strategy that cannot turn thinking off.
    public init(config: ReasoningConfig, thinkingEnabled: Bool?) throws {
        additionalContext = try config.promptStrategy.additionalContext(
            forThinkingEnabled: thinkingEnabled)
        closing = config.promptStrategy.closedBlock(forThinkingEnabled: thinkingEnabled)
            .map { (config, $0) }
    }

    /// The render of a thinking switch that a caller can leave unset.
    ///
    /// The switch applies only when the model declares a reasoning protocol.
    /// An unset switch, or a model with no reasoning protocol, gives the render
    /// of ``init()``: the template gets the variables of the caller unchanged.
    ///
    /// - Parameters:
    ///   - thinkingSwitch: `true` / `false` to turn thinking on / off, or nil
    ///     to keep the default of the template.
    ///   - config: the reasoning protocol of the model, or nil when the model
    ///     declares none.
    /// - Throws: ``ReasoningError/cannotDisableReasoning`` when `false` is
    ///   requested on a strategy that cannot turn thinking off.
    public init(thinkingSwitch: Bool?, config: ReasoningConfig?) throws {
        guard let thinkingSwitch, let config else {
            self.init()
            return
        }
        try self.init(config: config, thinkingEnabled: thinkingSwitch)
    }

    /// The template variables of a render: `base`, with the variables of this
    /// render in place of the same keys.
    ///
    /// - Parameter base: the template variables of the caller, or nil.
    /// - Returns: the merged variables, or `base` when this render states none.
    public func templateContext(over base: [String: any Sendable]?) -> [String: any Sendable]? {
        guard let additionalContext else { return base }
        return (base ?? [:]).merging(additionalContext) { _, stated in stated }
    }

    /// Renders `input` through `processor` with the template variables of
    /// this render, and closes the reasoning block at the end of the render
    /// when the strategy asks for it.
    ///
    /// - Parameters:
    ///   - input: the input to render. The template variables of this render
    ///     replace the same keys of its `additionalContext`.
    ///   - processor: the processor of the model, which renders the prompt.
    ///   - tokenizer: the tokenizer of the model.
    /// - Returns: the prepared prompt.
    /// - Throws: the error of the processor.
    public func prepare(
        _ input: UserInput, processor: any UserInputProcessor, tokenizer: any Tokenizer
    ) async throws -> LMInput {
        var input = input
        input.additionalContext = templateContext(over: input.additionalContext)
        let prepared = try await processor.prepare(input: input)
        guard let closing else { return prepared }
        return Self.closingReasoning(
            of: prepared, config: closing.config, block: closing.block, tokenizer: tokenizer)
    }

    /// `input` with `block` in place of the open reasoning block at its end,
    /// in the rank and the mask presence of `input`, thus the model sees the
    /// shape its processor makes.
    ///
    /// - Parameters:
    ///   - input: the prepared prompt with thinking on.
    ///   - config: the reasoning protocol whose start delimiter opens the
    ///     block.
    ///   - block: the closed, empty reasoning block that ends the prompt.
    ///   - tokenizer: the tokenizer of the model.
    /// - Returns: the prompt that ends with `block`.
    private static func closingReasoning(
        of input: LMInput, config: ReasoningConfig, block: String, tokenizer: any Tokenizer
    ) -> LMInput {
        let promptTokens = input.text.tokens
        let closed = config.closingReasoning(
            in: promptTokens.asArray(Int.self), with: block, tokenizer: tokenizer)
        var closedTokens = MLXArray(closed.map { Int32($0) }).asType(promptTokens.dtype)
        if promptTokens.ndim == LMInput.Text.batchedTokenRank {
            closedTokens = closedTokens[.newAxis, 0...]
        }
        let closedMask = input.text.mask.map { ones(like: closedTokens).asType($0.dtype) }
        return LMInput(
            text: .init(tokens: closedTokens, mask: closedMask),
            image: input.image, video: input.video, audio: input.audio)
    }
}

extension LMInput.Text {

    /// The rank of a batched token tensor (`[1, N]`) from a VLM processor.
    /// LLM processors make rank-1 tokens (`[N]`).
    package static let batchedTokenRank = 2
}
