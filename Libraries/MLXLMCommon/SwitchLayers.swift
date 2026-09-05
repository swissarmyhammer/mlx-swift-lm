import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// A compiled, shapeless `silu(gate) * up` product used as the default gate/up combination for ``SwitchGLU`` and ``FusedGateUpSwitchGLU`` when no custom activation is supplied.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

/// Combines expert outputs weighted by routing coefficients, summing across experts per token: `(outputs * weights).sum(axis: -2)`, where `weights` has been expanded to broadcast against `outputs`' trailing feature axis.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

/// Flattens `x` and `indices` and sorts tokens by their assigned expert index so that each expert's tokens are contiguous, enabling an efficient batched gather-matmul in ``SwitchLinear``/``QuantizedSwitchLinear``.
///
/// - Parameters:
///   - x: token hidden states with a trailing `[num_tokens, m, ...]`-shaped
///     leading structure, where `m` is the number of experts routed to per token.
///   - indices: per-token expert indices, shape `[..., m]`.
/// - Returns: a tuple of `(sortedX, sortedIndices, inverseOrder)`, where
///   `inverseOrder` is the permutation that undoes the sort (consumed by
///   ``scatterUnsort``).
public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

/// Inverse of ``gatherSort``: restores `x` to its original token order using the `invOrder` permutation produced by the matching `gatherSort` call.
///
/// - Parameters:
///   - x: expert outputs in sorted-by-expert order.
///   - invOrder: the inverse permutation returned by ``gatherSort``.
///   - shape: if provided, the sorted-order result is unflattened back to this
///     shape (typically the original `indices` shape) after unsorting.
/// - Returns: `x` restored to original token order, reshaped to `shape` when given.
public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

/// Fused inverse-permutation and weighted reduction for sorted MoE rows.
///
/// `SwitchGLU` sorts expert assignments before its gathered matrix
/// multiplications. The established path restores a full
/// `[tokens, topK, hidden]` tensor and then reduces `topK`. This kernel reads
/// the sorted rows through the inverse permutation and writes
/// `[tokens, hidden]` directly, avoiding that intermediate allocation.
private let weightedExpertUnsortKernel = MLXFast.metalKernel(
    name: "weighted_expert_unsort",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
            uint feature = thread_position_in_grid.x;
            uint token = thread_position_in_grid.y;

            T accumulator = (T)0;
            const uint assignment_base = token * (uint)K;
            for (uint slot = 0; slot < (uint)K; ++slot) {
                const uint assignment = assignment_base + slot;
                const uint sorted_row = (uint)inverse_order[assignment];
                // Match the legacy bfloat16 multiply-then-reduce rounding.
                const T weighted = (T)(
                    (float)sorted_outputs[sorted_row * threads_per_grid.x + feature]
                    * (float)weights[assignment]);
                accumulator = accumulator + weighted;
            }
            output[token * threads_per_grid.x + feature] = accumulator;
        """,
    ensureRowContiguous: true)

/// Reduce sorted top-8 bfloat16 expert rows without materializing their
/// unsorted assignment tensor. Callers must retain the established path for
/// every unsupported dtype, shape, or training topology.
package func weightedExpertUnsort(
    sortedOutputs: MLXArray,
    inverseOrder: MLXArray,
    weights: MLXArray
) -> MLXArray {
    let hidden = sortedOutputs.dim(1)
    precondition(
        sortedOutputs.ndim == 2 && hidden.isMultiple(of: 64)
            && sortedOutputs.dtype == .bfloat16,
        "weightedExpertUnsort requires bfloat16 [assignments, hidden], hidden % 64 == 0")
    precondition(
        inverseOrder.ndim == 1 && inverseOrder.dtype == .uint32,
        "weightedExpertUnsort requires flat uint32 inverse order")
    precondition(
        weights.ndim == 2 && weights.dim(1) == 8 && weights.size >= 64
            && weights.dtype == .bfloat16,
        "weightedExpertUnsort requires sorted-prefill bfloat16 [tokens, 8]")
    precondition(
        sortedOutputs.dim(0) == weights.size && inverseOrder.size == weights.size,
        "weightedExpertUnsort assignment counts must match")

    let tokens = weights.dim(0)
    return weightedExpertUnsortKernel(
        [sortedOutputs, inverseOrder, weights],
        template: [("T", sortedOutputs.dtype), ("K", 8)],
        grid: (hidden, tokens, 1),
        threadGroup: (64, 4, 1),
        outputShapes: [[tokens, hidden]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// The projected expert outputs of ``projectSwitchExperts(_:_:transformInput:combine:transformHidden:downProj:)``.
///
/// `output` holds the `downProj` result. When the tokens were sorted by expert
/// for the gather-matmul, `output` is still in that sorted order and
/// `inverseOrder` is the permutation that restores the caller's token order
/// (see ``unsortSwitchOutput(_:indices:)``); otherwise `inverseOrder` is `nil`.
private typealias ProjectedSwitchOutput = (output: MLXArray, inverseOrder: MLXArray?)

/// Shared projection path for the switch-layer MoE modules: expands `x` for per-expert routing, applies `transformInput`, sorts tokens by expert when the batch is large enough to benefit, invokes `combine` to compute the (already gate/up-activated) per-expert hidden state, applies `transformHidden`, and projects down via `downProj`.
///
/// The result stays in sorted order when sorting was applied; pass it to
/// ``unsortSwitchOutput(_:indices:)`` to restore the caller's token order, or
/// to ``weightedExpertUnsort(sortedOutputs:inverseOrder:weights:)`` to reduce
/// it directly.
///
/// `combine` receives the (possibly sorted) hidden states, the matching
/// expert indices, and whether sorting was applied, and must return the
/// activated hidden state ready for `downProj` -- this is the one piece that
/// differs between ``SwitchGLU`` (separate `gate_proj`/`up_proj`) and
/// ``FusedGateUpSwitchGLU`` (single fused `gate_up_proj` split in two).
private func projectSwitchExperts(
    _ x: MLXArray,
    _ indices: MLXArray,
    transformInput: (MLXArray) -> MLXArray = { $0 },
    combine: (_ x: MLXArray, _ indices: MLXArray, _ sorted: Bool) -> MLXArray,
    transformHidden: (MLXArray) -> MLXArray = { $0 },
    downProj: SwitchLinear
) -> ProjectedSwitchOutput {
    var x = MLX.expandedDimensions(x, axes: [-2, -3])
    x = transformInput(x)

    let doSort = indices.size >= 64

    var expertIndices = indices
    var inverseOrder = MLXArray()

    if doSort {
        (x, expertIndices, inverseOrder) = gatherSort(x: x, indices: indices)
    }

    let activated = transformHidden(combine(x, expertIndices, doSort))
    let out = downProj(activated, expertIndices, sortedIndices: doSort)

    return (out, doSort ? inverseOrder : nil)
}

/// Restores the projected expert outputs to the caller's token order (when ``projectSwitchExperts(_:_:transformInput:combine:transformHidden:downProj:)`` sorted them) and squeezes the routing axis back out.
///
/// - Returns: per-token, per-selected-expert outputs, shape `[..., k, inputDims]`.
private func unsortSwitchOutput(
    _ projected: ProjectedSwitchOutput, indices: MLXArray
) -> MLXArray {
    var out = projected.output
    if let inverseOrder = projected.inverseOrder {
        out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
    }
    return MLX.squeezed(out, axis: -2)
}

// MARK: - SwitchGLU

/// A mixture-of-experts gated linear unit that routes each token through the `gate_proj` / `up_proj` / `down_proj` rows selected by its expert indices, combining `activation(gate) * up` before the down projection.
///
/// Used by sparse MoE decoder blocks throughout the repo (e.g. Mixtral,
/// DeepSeek-V3, Qwen3-MoE) wherever experts ship as separate gate/up weights
/// rather than a single fused tensor (see ``FusedGateUpSwitchGLU`` for that
/// layout).
open class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Creates a `SwitchGLU` with `numExperts` independent gate/up/down expert weight matrices.
    ///
    /// - Parameters:
    ///   - inputDims: hidden size of the token embeddings entering/leaving the block.
    ///   - hiddenDims: per-expert intermediate (MLP) size.
    ///   - numExperts: number of experts (rows in each `SwitchLinear`).
    ///   - activation: optional single-argument activation applied to the gate
    ///     projection before multiplying elementwise by the up projection. When
    ///     `nil` (the default), the compiled `silu(gate) * up` product
    ///     (``compiledSiluProduct``) is used instead.
    ///   - bias: whether the per-expert projections carry a bias term.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: ((MLXArray) -> MLXArray)? = nil,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        if let activation {
            self.activation = activation
            self.activationProduct = nil
        } else {
            self.activation = MLXNN.silu
            self.activationProduct = compiledSiluProduct
        }

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Transforms the expanded input ahead of the expert gather/sort.
    ///
    /// This pair of hooks exists so subclasses can wrap the expert
    /// projections without copying `projectExperts`' dataflow -- and
    /// silently detaching from future changes to it, e.g. the gather/sort
    /// threshold or the compiled activation product. `RotateSwitchGLU`
    /// rotates activations here; the identity defaults add no graph nodes.
    func transformInput(_ x: MLXArray) -> MLXArray { x }

    /// Transforms the activated hidden state ahead of `downProj`.
    /// Identity by default -- see `transformInput`.
    func transformHidden(_ x: MLXArray) -> MLXArray { x }

    /// Runs the shared projection path with this module's gate/up/down experts and transform hooks, leaving the output in sorted order when the tokens were sorted.
    private func projectExperts(_ x: MLXArray, _ indices: MLXArray) -> ProjectedSwitchOutput {
        projectSwitchExperts(
            x, indices,
            transformInput: transformInput,
            combine: { x, expertIndices, doSort in
                let xUp = self.upProj(x, expertIndices, sortedIndices: doSort)
                let xGate = self.gateProj(x, expertIndices, sortedIndices: doSort)
                if let activationProduct = self.activationProduct {
                    return activationProduct(xGate, xUp)
                } else {
                    return self.activation(xGate) * xUp
                }
            },
            transformHidden: transformHidden,
            downProj: downProj)
    }

    private func legacyWeightedReduction(
        _ projected: ProjectedSwitchOutput,
        indices: MLXArray,
        weights: MLXArray
    ) -> MLXArray {
        weightedExpertSum(unsortSwitchOutput(projected, indices: indices), weights)
    }

    /// Whether this call has the exact frozen, quantized inference topology
    /// supported by ``weightedExpertUnsort(sortedOutputs:inverseOrder:weights:)``.
    package func supportsDirectWeightedReduction(
        _ x: MLXArray, _ indices: MLXArray, weights: MLXArray
    ) -> Bool {
        let projections = [gateProj, upProj, downProj]
        return inputDims.isMultiple(of: 64)
            && x.ndim == 2
            && x.dim(1) == inputDims
            && x.dtype == .bfloat16
            && indices.ndim == 2
            && indices.dim(0) == x.dim(0)
            && indices.dim(1) == 8
            && indices.dtype == .uint32
            && weights.shape == indices.shape
            && weights.dtype == .bfloat16
            && indices.size >= 64
            && projections.allSatisfy {
                ObjectIdentifier(type(of: $0)) == ObjectIdentifier(QuantizedSwitchLinear.self)
                    && $0.bias == nil
            }
            && trainableParameters().flattened().isEmpty
    }

    /// Routes `x` through the experts selected by `indices`, combining the gate/up projections with the configured activation and projecting the result back down to `inputDims`.
    ///
    /// - Parameters:
    ///   - x: token hidden states, shape `[..., inputDims]`.
    ///   - indices: per-token expert indices, shape `[..., k]` for top-`k` routing.
    /// - Returns: per-token, per-selected-expert outputs, shape `[..., k, inputDims]`.
    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        unsortSwitchOutput(projectExperts(x, indices), indices: indices)
    }

    /// Projects and combines the selected experts, directly reducing sorted
    /// production prefill rows when requested and eligible.
    ///
    /// Disabled, decode-sized, non-bfloat16, custom, and trainable calls use
    /// the established scatter + ``weightedExpertSum`` path unchanged.
    package func callAndWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool
    ) -> MLXArray {
        guard fuseSortedReduction,
            supportsDirectWeightedReduction(x, indices, weights: weights)
        else {
            return weightedExpertSum(callAsFunction(x, indices), weights)
        }

        let projected = projectExperts(x, indices)
        guard let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == inputDims,
            projected.output.dtype == .bfloat16
        else {
            return legacyWeightedReduction(projected, indices: indices, weights: weights)
        }

        return weightedExpertUnsort(
            sortedOutputs: MLX.squeezed(projected.output, axis: -2),
            inverseOrder: inverseOrder,
            weights: weights)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate `gate_proj` / `up_proj`.
///
/// Used by Gemma 4 26B MoE and MiniMax-M3.
open class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?
    let twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)?

    /// Number of components fused into `gate_up_proj` (gate + up = 2). Used to
    /// size the fused projection's output dimension and to split it back into
    /// its two halves; kept as a single constant so the two sites cannot drift
    /// out of sync.
    private static let gateUpFusionFactor = 2

    /// Creates a `FusedGateUpSwitchGLU` with `numExperts` fused gate/up expert weight matrices.
    ///
    /// - Parameters:
    ///   - inputDims: hidden size of the token embeddings entering/leaving the block.
    ///   - hiddenDims: per-expert intermediate (MLP) size (before the `2x` gate/up fusion).
    ///   - numExperts: number of experts (rows in each `SwitchLinear`).
    ///   - activation: optional single-argument activation applied to the gate
    ///     half of the fused projection before multiplying elementwise by the
    ///     up half. Mutually exclusive with `twoArgActivation`; when both are
    ///     `nil` (the default), the compiled `silu(gate) * up` product
    ///     (``compiledSiluProduct``) is used instead.
    ///   - twoArgActivation: optional activation that combines the split
    ///     gate/up halves directly rather than via the `activation(gate) * up`
    ///     seam above -- e.g. MiniMax-M3's swigluoai, which clips the gate and
    ///     up halves asymmetrically before combining them (see
    ///     `MiniMaxM3SwiGLUOAI`) rather than computing a plain elementwise
    ///     product. The closure receives the split halves as `(gate, up)`,
    ///     matching the `activationProduct` convention above. Takes precedence
    ///     over `activation` if both are supplied.
    ///   - bias: whether the per-expert projections carry a bias term.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: ((MLXArray) -> MLXArray)? = nil,
        twoArgActivation: ((MLXArray, MLXArray) -> MLXArray)? = nil,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        if let twoArgActivation {
            self.activation = { $0 }
            self.activationProduct = nil
            self.twoArgActivation = twoArgActivation
        } else if let activation {
            self.activation = activation
            self.activationProduct = nil
            self.twoArgActivation = nil
        } else {
            self.activation = MLXNN.silu
            self.activationProduct = compiledSiluProduct
            self.twoArgActivation = nil
        }

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: Self.gateUpFusionFactor * hiddenDims,
            numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Routes `x` through the experts selected by `indices`, splitting the fused gate/up projection in half, combining the halves with the configured activation (`twoArgActivation`, `activationProduct`, or the single-argument `activation` fallback, in that order of precedence), and projecting the result back down to `inputDims`.
    ///
    /// - Parameters:
    ///   - x: token hidden states, shape `[..., inputDims]`.
    ///   - indices: per-token expert indices, shape `[..., k]` for top-`k` routing.
    /// - Returns: per-token, per-selected-expert outputs, shape `[..., k, inputDims]`.
    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let projected = projectSwitchExperts(
            x, indices,
            combine: { x, expertIndices, doSort in
                let gateUp = self.gateUpProj(x, expertIndices, sortedIndices: doSort)
                let parts = MLX.split(gateUp, parts: Self.gateUpFusionFactor, axis: -1)
                if let twoArgActivation = self.twoArgActivation {
                    return twoArgActivation(parts[0], parts[1])
                } else if let activationProduct = self.activationProduct {
                    return activationProduct(parts[0], parts[1])
                } else {
                    return self.activation(parts[0]) * parts[1]
                }
            },
            downProj: downProj)
        return unsortSwitchOutput(projected, indices: indices)
    }
}

/// Adds the per-expert bias (if present) selected by `indices` to `result`, broadcasting the gathered bias row against the trailing token axis.
///
/// Shared by ``SwitchLinear`` and ``QuantizedSwitchLinear``'s `callAsFunction`.
///
/// - Parameters:
///   - result: the gather-matmul output to add bias to, shape `[..., outputDims]`.
///   - bias: per-expert bias, shape `[numExperts, outputDims]`, or `nil` if unused.
///   - indices: per-token expert indices selecting which bias row to use.
/// - Returns: `result` with the gathered bias added, or `result` unchanged if `bias` is `nil`.
private func applyBias(_ result: MLXArray, bias: MLXArray?, indices: MLXArray) -> MLXArray {
    guard let bias else { return result }
    return result + MLX.expandedDimensions(bias[indices], axis: -2)
}

/// Default number of elements sharing a single quantization scale/bias group.
/// Shared by ``SwitchLinear/toQuantized(groupSize:bits:mode:)`` and
/// ``QuantizedSwitchLinear/init(_:groupSize:bits:mode:)`` so their defaults
/// cannot drift out of sync.
public let defaultQuantizationGroupSize = 64

/// Default number of bits per quantized weight value. Shared by
/// ``SwitchLinear/toQuantized(groupSize:bits:mode:)`` and
/// ``QuantizedSwitchLinear/init(_:groupSize:bits:mode:)`` so their defaults
/// cannot drift out of sync.
public let defaultQuantizationBits = 4

// MARK: - SwitchLinear

/// A per-expert linear layer: holds `numExperts` independent `[outputDims, inputDims]` weight matrices (plus optional per-expert bias) and applies the one selected by each token's expert index via a gather-matmul, rather than materializing a dense batched matmul over all experts.
open class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    /// Creates a `SwitchLinear` with randomly-initialized weights (uniform in `[-1/sqrt(inputDims), 1/sqrt(inputDims)]`), matching `MLXNN.Linear`'s default initialization scaled per expert.
    ///
    /// - Parameters:
    ///   - inputDims: input feature dimension.
    ///   - outputDims: output feature dimension.
    ///   - numExperts: number of independent per-expert weight matrices.
    ///   - bias: whether to allocate a zero-initialized per-expert bias.
    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    /// Applies the per-expert weight (and bias, if present) selected by `indices` to `x` via a gather-matmul.
    ///
    /// - Parameters:
    ///   - x: input activations, shape `[..., inputDims]`.
    ///   - indices: per-token expert indices selecting which weight row to use.
    ///   - sortedIndices: pass `true` when `indices` is already sorted by
    ///     expert (as produced by ``gatherSort``), enabling a faster gather-matmul.
    /// - Returns: per-token outputs, shape `[..., outputDims]`.
    open func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        let result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        return applyBias(result, bias: self.bias, indices: indices)
    }

    /// Returns a quantized copy of this layer as a ``QuantizedSwitchLinear``.
    ///
    /// - Parameters:
    ///   - groupSize: number of elements per quantization group.
    ///   - bits: number of bits per quantized value.
    ///   - mode: the quantization scheme to apply.
    /// - Returns: a new ``QuantizedSwitchLinear`` wrapping quantized weights.
    public func toQuantized(
        groupSize: Int = defaultQuantizationGroupSize, bits: Int = defaultQuantizationBits,
        mode: QuantizationMode
    ) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Quantized counterpart of ``SwitchLinear``: stores the per-expert weight matrix in packed/quantized form (plus per-group `scales` and `biases` for dequantization) instead of full-precision floats, and applies it via a quantized gather-matmul.
open class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    /// Number of elements sharing a single quantization scale/bias group.
    public let groupSize: Int
    /// Number of bits used per quantized weight value.
    public let bits: Int
    /// The quantization scheme (e.g. affine vs. symmetric) used for this layer.
    public let mode: QuantizationMode

    /// Quantizes an existing ``SwitchLinear``'s weights into this layer, freezing the result (quantized parameters are not trainable).
    ///
    /// - Parameters:
    ///   - other: the full-precision `SwitchLinear` to quantize.
    ///   - groupSize: number of elements per quantization group.
    ///   - bits: number of bits per quantized value.
    ///   - mode: the quantization scheme to apply.
    public init(
        _ other: SwitchLinear, groupSize: Int = defaultQuantizationGroupSize,
        bits: Int = defaultQuantizationBits, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    /// Applies the per-expert quantized weight (and bias, if present) selected by `indices` to `x` via a quantized gather-matmul, dequantizing on the fly.
    ///
    /// - Parameters:
    ///   - x: input activations, shape `[..., inputDims]`.
    ///   - indices: per-token expert indices selecting which weight row to use.
    ///   - sortedIndices: pass `true` when `indices` is already sorted by
    ///     expert (as produced by ``gatherSort``), enabling a faster gather-matmul.
    /// - Returns: per-token outputs, shape `[..., outputDims]`.
    override open func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        return applyBias(result, bias: self.bias, indices: indices)
    }
}
