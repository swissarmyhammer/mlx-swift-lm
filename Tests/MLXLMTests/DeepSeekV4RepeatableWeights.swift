// Copyright © 2026 Apple Inc.

import MLX
import MLXNN

/// The repeatable random weights of a synthetic DeepSeek-V4 checkpoint, for
/// the DeepSeek-V4 test suites.
///
/// More than one suite fills a module with random weights that each run draws
/// again with the same values. Each suite fills the module here, thus one
/// change to the filler reaches every suite.
///
/// Each parameter gets its own seed. The first parameter in ``KeyOrder``
/// takes the start seed plus one, and each next parameter takes one more.
/// Each value is uniform between `low` and `high`, in float32.
enum DeepSeekV4RepeatableWeights {

    /// The order in which the parameters get their seeds.
    ///
    /// The two orders are different. `flattened` sorts the keys of each level
    /// on its own and gives the items of an array by index, thus `layers.10`
    /// comes after `layers.2`. `sorted` puts `layers.10` before `layers.2`.
    enum KeyOrder {
        /// The order of `ModuleParameters.flattened()`.
        case flattened

        /// The full parameter paths, in `String` order.
        case sorted
    }

    /// Builds one repeatable random weight for each parameter of a module.
    ///
    /// - Parameters:
    ///   - module: The module whose parameters the checkpoint must fill.
    ///   - seed: The seed the filler starts at.
    ///   - low: The low end of every random value.
    ///   - high: The high end of every random value.
    ///   - order: The order in which the parameters get their seeds.
    ///   - isIncluded: Tells whether the checkpoint holds the parameter at
    ///     one path. A parameter it leaves out takes no seed.
    ///   - replacement: The fixed weight for the parameter at one path, of
    ///     the given shape, or `nil` for a random weight. A parameter that
    ///     gets a fixed weight still takes its seed.
    /// - Returns: The weights, by parameter path.
    static func checkpoint(
        for module: Module, seed: UInt64, low: Float, high: Float,
        order: KeyOrder = .sorted,
        including isIncluded: (String) -> Bool = { _ in true },
        replacing replacement: (String, [Int]) -> MLXArray? = { _, _ in nil }
    ) -> [String: MLXArray] {
        let parameters = orderedParameters(of: module, order: order)
            .filter { isIncluded($0.0) }
        let weights = parameters.enumerated().map { offset, parameter in
            let (path, value) = parameter
            let weight =
                replacement(path, value.shape)
                ?? MLXRandom.uniform(
                    low: low, high: high, value.shape,
                    key: MLXRandom.key(seed + UInt64(offset) + 1))
            return (path, weight)
        }
        return Dictionary(uniqueKeysWithValues: weights)
    }

    /// Loads one repeatable random weight into every parameter a module
    /// declares, through `update(parameters:verify:[.all])`.
    ///
    /// The verification is the one `MLXLMCommon.loadWeights` applies, thus
    /// the load also states that the module declares exactly the parameters
    /// a checkpoint fills.
    ///
    /// - Parameters:
    ///   - module: The module to fill.
    ///   - seed: The seed the filler starts at.
    ///   - low: The low end of every random value.
    ///   - high: The high end of every random value.
    ///   - order: The order in which the parameters get their seeds.
    ///   - replacement: The fixed weight for the parameter at one path, of
    ///     the given shape, or `nil` for a random weight. A parameter that
    ///     gets a fixed weight still takes its seed.
    static func load(
        into module: Module, seed: UInt64, low: Float, high: Float,
        order: KeyOrder = .sorted,
        replacing replacement: (String, [Int]) -> MLXArray? = { _, _ in nil }
    ) throws {
        let weights = checkpoint(
            for: module, seed: seed, low: low, high: high, order: order,
            replacing: replacement)
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(module)
    }

    /// The parameters of a module, by path, in the given order.
    ///
    /// - Parameters:
    ///   - module: The module to read.
    ///   - order: The order of the result.
    /// - Returns: Each parameter path with its array.
    private static func orderedParameters(
        of module: Module, order: KeyOrder
    ) -> [(String, MLXArray)] {
        let parameters = module.parameters().flattened()
        switch order {
        case .flattened:
            return parameters
        case .sorted:
            return parameters.sorted { $0.0 < $1.0 }
        }
    }
}
