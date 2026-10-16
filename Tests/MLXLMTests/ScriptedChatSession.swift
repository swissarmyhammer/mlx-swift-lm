// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon

extension ChatSession {
    /// A greedy session over a scripted model, with the configuration, the
    /// context and the input processor that a loaded model gets.
    ///
    /// Shared by the weight-free `ChatSession` suites of the history render.
    /// Each suite gives only the settings that are different for its model.
    ///
    /// - Parameters:
    ///   - model: the scripted model. Its tool call format is the response
    ///     protocol of the configuration.
    ///   - modelID: the identifier of the configuration.
    ///   - tokenizer: the tokenizer of the context and of the input processor.
    ///   - reasoningConfig: the reasoning protocol of the model, or nil for
    ///     none.
    ///   - history: the messages of the turns before the first turn.
    ///   - tokenBudget: the tokens each turn may generate.
    ///   - additionalContext: the context that the chat template gets, or nil
    ///     for none.
    ///   - thinkingEnabled: the thinking switch of the session, or nil when the
    ///     switch is not set.
    convenience init(
        scripted model: ScriptedTokenModel, modelID: String, tokenizer: any Tokenizer,
        reasoningConfig: ReasoningConfig?, history: [Chat.Message] = [], tokenBudget: Int,
        additionalContext: [String: any Sendable]? = nil, thinkingEnabled: Bool? = nil
    ) {
        let configuration = ModelConfiguration(
            id: modelID, toolCallFormat: model.toolCallFormat, reasoningConfig: reasoningConfig)
        let context = ModelContext(
            configuration: configuration, model: model,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        self.init(
            context, history: history,
            generateParameters: GenerateParameters(maxTokens: tokenBudget, temperature: 0),
            additionalContext: additionalContext)
        self.thinkingEnabled = thinkingEnabled
    }
}
