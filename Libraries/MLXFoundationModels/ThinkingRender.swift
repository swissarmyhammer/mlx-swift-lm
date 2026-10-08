// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel.Executor {

    /// The thinking state that one render of a prompt carries into the chat
    /// template.
    ///
    /// A strategy that renders thinking off as a flag passes the flag through
    /// the template variables. A strategy that renders thinking off as a closed
    /// block (``ThinkingOffRender/closedBlock(_:)``) keeps the flag on, and
    /// ``prepare(messages:tools:context:)`` puts the closed block in place of
    /// the open block at the end of the render. The start of the prompt is then
    /// the same with thinking on and off.
    struct ThinkingRender {

        /// The template variables that state the thinking flag, or nil when
        /// the render states none.
        let additionalContext: [String: any Sendable]?

        /// The protocol and the closed block that ends the prompt, or nil when
        /// the render keeps the prompt as the template writes it.
        let closing: (config: ReasoningConfig, block: String)?

        /// A render that states no thinking flag and closes no block.
        init() {
            additionalContext = nil
            closing = nil
        }

        /// The render of a "thinking enabled" preference under `config`.
        ///
        /// - Parameters:
        ///   - config: the reasoning protocol of the model.
        ///   - thinkingEnabled: `true` / `false` to force thinking on / off,
        ///     `nil` when the caller expressed no preference.
        /// - Throws: ``ReasoningError/cannotDisableReasoning`` when `false` is
        ///   requested on a strategy that cannot turn thinking off.
        init(config: ReasoningConfig, thinkingEnabled: Bool?) throws {
            additionalContext = try config.promptStrategy.additionalContext(
                forThinkingEnabled: thinkingEnabled)
            closing = config.promptStrategy.closedBlock(forThinkingEnabled: thinkingEnabled)
                .map { (config, $0) }
        }

        /// Renders `messages` through the processor of `context`, and closes
        /// the reasoning block at the end of the render when the strategy asks
        /// for it.
        ///
        /// - Parameters:
        ///   - messages: the chat messages the prompt renders from.
        ///   - tools: the tool specifications the template describes, or nil.
        ///   - context: the loaded model context whose processor renders the
        ///     prompt.
        /// - Returns: the prepared prompt.
        /// - Throws: the error of the processor.
        func prepare(
            messages: [Chat.Message], tools: [ToolSpec]?, context: ModelContext
        ) async throws -> LMInput {
            let input = try await context.processor.prepare(
                input: UserInput(chat: messages, tools: tools, additionalContext: additionalContext)
            )
            guard let closing else { return input }
            return Self.closingReasoning(
                of: input, config: closing.config, block: closing.block,
                tokenizer: context.tokenizer)
        }

        /// The text of the user message that
        /// ``transcriptBoundary(of:messages:tools:context:)`` adds to the
        /// messages. The text does not matter: the render parts from the
        /// prompt before it.
        static let boundaryProbeText = "."

        /// The index of `prompt` where a next render of the same messages
        /// plus one new user message parts from it, for a hybrid model.
        ///
        /// A chat template writes a generation prompt after the last message
        /// (Qwen 3.5: `<|im_start|>assistant\n<think>\n\n</think>\n\n`). A
        /// next turn that drops the turn the model generated, and adds a
        /// user message in its place, renders the same messages and then
        /// that user message, not the generation prompt. The recurrent layers
        /// of a hybrid model cannot rewind into the generation prompt, thus
        /// the executor keeps a checkpoint at the point where the two
        /// renders part.
        ///
        /// This method finds that point: it renders `messages` plus one user
        /// message with the same template variables, tools and closed block
        /// as `prompt`, and takes the prefix the two renders share. With no
        /// generation prompt, the point is the end of `prompt`.
        ///
        /// - Parameters:
        ///   - prompt: the prepared prompt of `messages`, which this render
        ///     made.
        ///   - messages: the chat messages `prompt` renders from.
        ///   - tools: the tool specifications `prompt` describes, or nil.
        ///   - context: the loaded model context whose processor renders the
        ///     prompt.
        /// - Returns: the index, or nil when the model is not hybrid, when
        ///   `prompt` carries media (the prompt cache carries no cache for
        ///   such a prompt), or when the second render fails. A template can
        ///   refuse two user messages after each other, and then the pass
        ///   takes no checkpoint.
        func transcriptBoundary(
            of prompt: LMInput, messages: [Chat.Message], tools: [ToolSpec]?,
            context: ModelContext
        ) async -> Int? {
            guard prompt.image == nil, prompt.video == nil, prompt.audio == nil,
                ExecutorPromptCacheCheckpoint.applies(to: context.model),
                let probe = try? await prepare(
                    messages: messages + [.user(Self.boundaryProbeText)], tools: tools,
                    context: context)
            else {
                return nil
            }
            return commonPrefixLength(
                of: prompt.text.tokens.asArray(Int.self), and: probe.text.tokens.asArray(Int.self))
        }

        /// `input` with `block` in place of the open reasoning block at its
        /// end, in the rank and the mask presence of `input`, thus the model
        /// sees the shape its processor makes.
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
            if promptTokens.ndim == MLXLanguageModel.Executor.batchedPromptRank {
                closedTokens = closedTokens[.newAxis, 0...]
            }
            let closedMask = input.text.mask.map { ones(like: closedTokens).asType($0.dtype) }
            return LMInput(
                text: .init(tokens: closedTokens, mask: closedMask),
                image: input.image,
                video: input.video)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
