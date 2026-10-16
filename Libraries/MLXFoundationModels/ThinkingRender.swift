// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon

/// The executor renders each prompt through ``ThinkingRender`` of
/// `MLXLMCommon`, the same render that `ChatSession` uses.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension ThinkingRender {

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
        try await prepare(
            UserInput(chat: messages, tools: tools), processor: context.processor,
            tokenizer: context.tokenizer)
    }

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
    /// This method finds that point with
    /// ``PromptCacheCheckpoint/lastMessageEnd(of:messages:render:)``,
    /// which renders with the same template variables, tools and closed
    /// block as `prompt`.
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
            PromptCacheCheckpoint.applies(to: context.model)
        else {
            return nil
        }
        return await PromptCacheCheckpoint.lastMessageEnd(of: prompt, messages: messages) {
            try await prepare(messages: $0, tools: tools, context: context)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
