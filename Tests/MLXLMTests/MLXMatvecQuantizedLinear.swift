// Copyright © 2026 Apple Inc.
//
// A quantized projection that always runs the matvec of MLX, as the
// reference of the tests that tell where ``MultiRowQuantizedMatvec`` runs.

import MLX
import MLXLMCommon
import MLXNN

/// A projection that always runs the matvec of MLX: ``MultiRowQuantizedMatvec``
/// reads only a stock `QuantizedLinear`, so a model with these projections
/// runs every call as it did before the kernel.
final class MLXMatvecQuantizedLinear: QuantizedLinear {

    /// Quantizes `model` as `quantize(model:groupSize:bits:filter:)` does,
    /// with each linear layer an ``MLXMatvecQuantizedLinear``.
    ///
    /// - Parameters:
    ///   - model: The model to quantize.
    ///   - groupSize: The group size of the quantized weights.
    ///   - bits: The bits of each quantized weight.
    ///   - filter: Tells, from its path and the module, if a layer is
    ///     quantized.
    static func quantize(
        _ model: Module, groupSize: Int, bits: Int,
        filter: (String, Module) -> Bool = { _, _ in true }
    ) {
        MLXNN.quantize(
            model: model, groupSize: groupSize, bits: bits, mode: .affine, filter: filter,
            apply: { layer, groupSize, bits, mode in
                if let linear = layer as? Linear {
                    return MLXMatvecQuantizedLinear(
                        linear, groupSize: groupSize, bits: bits, mode: mode)
                }
                return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
            })
    }
}
