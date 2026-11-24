// Copyright © 2026 Apple Inc.

import Foundation

/// Loads the checkpoint in `directory` with no weights, and puts a ``ClusterLanguageModel`` over
/// `transport` in the place of its model.
///
/// The load reads what the app needs around the forward pass: `config.json`, the tokenizer,
/// the processor, the generation configuration and the chat conventions of the model family.
/// It reads no weight file (``WeightFileSelection/skipped``): the weights live in the workers.
/// The suppressed tokens and the reasoning guard of the model family carry over to the cluster
/// model, so generation masks the same tokens a local load would.
///
/// A model factory must be linked, as for ``loadModel(from:using:progress:)``.
///
/// - Parameters:
///   - directory: The local checkpoint directory, the one the workers load.
///   - tokenizerLoader: The loader of the tokenizer.
///   - transport: The transport to the rank-0 worker.
/// - Returns: The context, whose model is the cluster model.
/// - Throws: The error of the worker handshake, or of the load.
public func loadClusterModelContext(
    from directory: URL, using tokenizerLoader: any TokenizerLoader,
    transport: any RemoteModelTransport
) async throws -> ModelContext {
    var configuration = ResolvedModelConfiguration(directory: directory)
    configuration.weightFileSelection = .skipped
    let context = try await load { factory in
        try await factory._load(configuration: configuration, tokenizerLoader: tokenizerLoader)
    }
    let model = try ClusterLanguageModel.connect(over: transport)
    return context.serving(through: model)
}

extension ModelContext {

    /// This context with `model` in the place of its model. `model` takes the suppressed tokens
    /// and the reasoning guard of the model it replaces.
    ///
    /// - Parameter model: The cluster model.
    /// - Returns: The context.
    func serving(through model: ClusterLanguageModel) -> ModelContext {
        model.suppressedTokenIds = suppressedTokenIds(for: self.model)
        registerClosedReasoningEndGuard(
            model: model, reasoning: configuration.reasoningConfig, tokenizer: tokenizer)
        return ModelContext(
            configuration: configuration, model: model, processor: processor,
            tokenizer: tokenizer)
    }
}
