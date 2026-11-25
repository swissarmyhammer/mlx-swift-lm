// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Loads the checkpoint in `directory` as the share of this process of `group`, one layer at a
/// time.
///
/// The load keeps the weights lazy (``WeightEvaluation/deferred``). Then, for each layer, it
/// evaluates the full weights of the layer, splits the layer, evaluates the shares and frees the
/// full weights. A process thus never holds more than one layer's full weights beside its
/// shares, and no single evaluation spans the whole model. The rest of the model, such as the
/// embedding and the LM head, evaluates last, and the model is prepared for inference.
///
/// A group of one process loads the whole model as ``loadModel(from:using:progress:)`` does.
///
/// A model factory must be linked, as for ``loadModel(from:using:progress:)``.
///
/// - Parameters:
///   - directory: The local checkpoint directory. Every process loads the same checkpoint.
///   - tokenizerLoader: The loader of the tokenizer.
///   - group: The group to split across. Every process passes the same group.
///   - willShardLayer: Runs on each layer before its full weights evaluate. This is the hook of
///     the weight broadcast: a process that lacks the weight files loads with
///     ``WeightFileSelection/skipped``, and here each process replaces the layer's parameters
///     with their all-sum across `group`, rank 0 contributing the tensors it read and every
///     other process zeros, as `mlx_lm.share` does. Nil runs nothing.
/// - Returns: The context, whose model holds the share of this process.
/// - Throws: ``TensorParallelShardingError`` when the model cannot split across `group`, the
///   error of `willShardLayer`, or of the load.
public func loadShardedModelContext(
    from directory: URL, using tokenizerLoader: any TokenizerLoader,
    group: MLXDistributed.Group, willShardLayer: TensorParallelLayerHooks.Hook? = nil
) async throws -> ModelContext {
    var configuration = ResolvedModelConfiguration(directory: directory)
    guard try group.size > 1 else {
        return try await load { factory in
            try await factory._load(configuration: configuration, tokenizerLoader: tokenizerLoader)
        }
    }
    configuration.weightEvaluation = .deferred
    // A lazy load reads no tensor, so a factory whose model cannot shard costs little, and the
    // next factory may know a model of the same type that can, as MLXLLM does for Qwen 3.5.
    let (context, shardable) = try await load { factory in
        let context = try await factory._load(
            configuration: configuration, tokenizerLoader: tokenizerLoader)
        guard let shardable = context.model as? TensorParallelShardable else {
            throw TensorParallelShardingError.notShardable(
                modelType: String(describing: type(of: context.model)))
        }
        return (context, shardable)
    }
    let hooks = TensorParallelLayerHooks.evaluating(after: willShardLayer)
    let work = SendableBox((shardable, context.model, group, hooks))
    // Evaluation blocks its thread, and a cooperative thread must never block.
    try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, any Error>) in
        DispatchQueue.global(qos: .userInitiated).async {
            continuation.resume(
                with: Result {
                    try withLoaderThreadError {
                        let (shardable, model, group, hooks) = work.consume()
                        try shardable.shardLayers(group: group, hooks: hooks)
                        materializeModelForInference(model)
                    }
                })
        }
    }
    return context
}

extension TensorParallelLayerHooks {

    /// Hooks that evaluate each layer before it splits, after `willShardLayer`, and evaluate the
    /// shares after it split, then return the freed full weights to the system.
    ///
    /// - Parameter willShardLayer: Runs first on each layer; nil runs nothing.
    /// - Returns: The hooks.
    static func evaluating(after willShardLayer: Hook?) -> Self {
        Self(
            willShard: { index, layer in
                try willShardLayer?(index, layer)
                eval(layer)
            },
            didShard: { _, layer in
                eval(layer)
                Memory.clearCache()
            })
    }
}
