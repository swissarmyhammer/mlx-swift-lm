// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN

/// The share of one process of a module, made in a single process, so a test can sum the partial
/// outputs of every share and compare the sum with the output of the whole module.
public enum TensorParallelShares {

    /// Keeps part `rank` of `groupSize` equal parts of each parameter of `module` that `splits`
    /// names, on the axis of its split. Every other parameter stays whole.
    ///
    /// - Parameters:
    ///   - rank: The process whose part the module keeps.
    ///   - groupSize: The number of processes.
    ///   - module: The module whose parameters split, in place.
    ///   - splits: The split of each key; see `TensorParallel.split(of:in:)`.
    public static func keep(
        rank: Int, of groupSize: Int, in module: Module, _ splits: [String: TensorParallel.Split]
    ) {
        let parameters = module.parameters().flattened().map { path, value in
            guard let split = TensorParallel.split(of: path, in: splits) else {
                return (path, value)
            }
            return (path, contiguous(value.split(parts: groupSize, axis: split.axis)[rank]))
        }
        module.update(parameters: ModuleParameters.unflattened(parameters))
    }
}
