// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator

    /// The message the prompt path throws when the tokenizer has no chat
    /// template and this model forbids the plain-text prompt fallback.
    ///
    /// The default is `nil`: the model permits the fallback. A model family
    /// that ships no chat template, and whose prompts a dedicated encoder
    /// must build, returns a message that names that encoder. The prompt
    /// path then throws
    /// ``PromptPreparationError/plainTextFallbackForbidden(_:)`` with this
    /// message instead of a silent, wrong plain-text prompt.
    var missingChatTemplateRefusal: String? { get }

    /// The tokenizer that the prompt path must use for this model.
    ///
    /// The default returns `tokenizer` unchanged. A model family whose
    /// prompts a dedicated encoder must build wraps the loaded tokenizer
    /// here — DeepSeek-V4 returns a `DeepSeekV4EncodingTokenizer`, because
    /// its checkpoint ships no chat template. `LLMModelFactory._load` calls
    /// this once, thus every consumer of the loaded context speaks through
    /// the returned tokenizer.
    ///
    /// - Parameter tokenizer: the tokenizer loaded from the checkpoint.
    /// - Returns: the tokenizer for the prompt path.
    func promptTokenizer(wrapping tokenizer: any Tokenizer) -> any Tokenizer
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // Prepare the prompt in chunks if larger than the prefill size.
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N.
            var state: LMOutput.State? = state
            while y.tokens.size > prefillStepSize {
                // Cooperative cancellation between prefill windows. On iOS, GPU work
                // submitted after the app moves to the background is rejected by the
                // system ("Insufficient Permission"), and the resulting command-buffer
                // error is thrown from a Metal completion handler where it cannot be
                // caught, aborting the process. Without this check a long prompt's
                // prefill cannot be interrupted, so apps cannot stop GPU submissions
                // in time when entering the background. See ml-explore/mlx-swift-examples#230.
                try Task.checkCancellation()
                let input = y[.newAxis, ..<prefillStepSize]
                let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                state = output.state
                asyncEval(cache)
                y = y[prefillStepSize...]
            }

            // Single sync after the loop to flush any remaining async work.
            eval(cache)
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }

    /// The default refusal: `nil`, which permits the plain-text prompt
    /// fallback for models whose tokenizer has no chat template.
    public var missingChatTemplateRefusal: String? {
        nil
    }

    /// The default prompt tokenizer: the loaded tokenizer itself, unchanged.
    ///
    /// - Parameter tokenizer: the tokenizer loaded from the checkpoint.
    /// - Returns: the same tokenizer.
    public func promptTokenizer(wrapping tokenizer: any Tokenizer) -> any Tokenizer {
        tokenizer
    }
}
