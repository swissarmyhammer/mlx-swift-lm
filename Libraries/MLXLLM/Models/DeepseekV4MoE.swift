// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Three details do not come from that file:
//
//  1. The routed experts. The file above calls a `SwitchGLU` of its own that
//     takes a two-argument activation, and it carries fused Metal kernels for
//     the mxfp4 and mxfp8 expert weights. This repository keeps its own
//     `Libraries/MLXLMCommon/SwitchLayers.swift`, whose `SwitchGLU` takes a
//     one-argument activation and thus cannot state the asymmetric DeepSeek-V4
//     clamp, and whose `FusedGateUpSwitchGLU` reads one fused `gate_up_proj`
//     tensor that this checkpoint does not ship. ``DeepseekV4SwitchGLU`` below
//     therefore builds on `SwitchLinear`, `gatherSort` and `scatterUnsort`.
//     The fused kernels stay out of scope.
//  2. The float32 reduction. The file above sums the weighted expert outputs
//     with `(y * scores[..., .newAxis]).sum(axis: -2)`. This file calls
//     `DeepseekV4Math.reduceRoutedExpertsFP32`, which states the float32
//     contract in one place, and casts back one time at the end.
//  3. The token identifiers. The file above threads them into the gate
//     through a mutable `currentInputIds` property, so that the layer can
//     conform to `UnaryLayer`. Every DeepSeek-V4 layer holds a mixture of
//     experts, thus a decoder layer can hold this type itself and hand the
//     identifiers over as an argument. This file takes them as an argument
//     and holds no state.
//
// One divergence between the references is open. Both Swift copies give
// `swiglu_limit` to the shared expert. The DeepSeek-V4 Python reference
// (Thump604/mlx-lm @ deepseek-v4-support-fixes, mlx_lm/models/deepseek_v4.py,
// `DeepseekV4MoE.__init__`) gives `swiglu_limit=0.0` there, which turns the
// clamp off. This file follows the Swift copies, because they are the copies
// it ports. Task `kp1pnj4` decides the point against the real checkpoint.

import MLX
import MLXLMCommon
import MLXNN

// MARK: - Dense feed-forward block

/// One dense DeepSeek-V4 feed-forward block.
///
/// DeepSeek-V4 reads this block as the shared expert of a mixture-of-experts
/// layer, which every token passes through beside its routed experts.
///
/// The activation is the DeepSeek-V4 clamped SwiGLU, thus it is not the plain
/// `silu(gate) * up` of the earlier DeepSeek families.
class DeepseekV4MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    /// The `swiglu_limit` the activation clamps with. A limit of zero or less
    /// turns the clamp off.
    let swigluLimit: Float

    /// Builds one dense feed-forward block.
    ///
    /// - Parameters:
    ///   - hiddenSize: The width of the residual stream.
    ///   - intermediateSize: The width of the block.
    ///   - swigluLimit: The `swiglu_limit` of the checkpoint.
    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.swigluLimit = swigluLimit
    }

    /// Reads one block of tokens.
    ///
    /// - Parameter x: The block input, shape `(..., hiddenSize)`.
    /// - Returns: The block output, of the shape of `x`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(
            DeepseekV4Math.clampedSwiGLU(gate: gateProj(x), up: upProj(x), limit: swigluLimit))
    }
}

// MARK: - Routed experts

/// The routed experts of one DeepSeek-V4 mixture-of-experts layer.
///
/// The checkpoint names this module `switch_mlp` and holds one
/// `gate_proj`, one `up_proj` and one `down_proj` for each routed expert. A
/// token reads only the experts its gate selected, thus the projections are
/// `SwitchLinear` layers, which gather the rows they need instead of
/// multiplying by every expert.
///
/// This is not ``SwitchGLU``. That layer takes a one-argument activation and
/// computes `activation(gate) * up`, and the DeepSeek-V4 clamp holds `up`
/// inside the limit as well, thus it cannot be written as a function of the
/// gate alone. ``FusedGateUpSwitchGLU`` does take a two-argument activation,
/// and it reads one fused `gate_up_proj` tensor, which this checkpoint does
/// not ship. The sort this layer runs, and the point it starts to sort at,
/// are the ones ``SwitchGLU`` runs.
class DeepseekV4SwitchGLU: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    /// The `swiglu_limit` each expert clamps its activation with.
    let swigluLimit: Float

    /// The number of routes a block must hold before a sort by expert pays
    /// for itself. `SwitchGLU` reads the same number, thus the two routed
    /// paths of this repository start to sort at the same point.
    private static let sortThreshold = 64

    /// The two axes a token gains before it reaches the experts: one axis for
    /// the routes of that token, and one axis the gathered matrix multiply
    /// reads as a row.
    private static let routeAxes = [-2, -3]

    /// The axis that holds one expert of a routed stack.
    private static let expertAxis = -2

    /// Builds the routed experts of one layer.
    ///
    /// - Parameters:
    ///   - inputDims: The width of the residual stream.
    ///   - hiddenDims: The width of one expert.
    ///   - expertCount: The number of routed experts.
    ///   - swigluLimit: The `swiglu_limit` of the checkpoint.
    init(inputDims: Int, hiddenDims: Int, expertCount: Int, swigluLimit: Float) {
        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: expertCount, bias: false)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: expertCount, bias: false)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: expertCount, bias: false)
        self.swigluLimit = swigluLimit
    }

    /// Sends each token through the experts its gate selected.
    ///
    /// - Parameters:
    ///   - x: The token values, shape `(..., inputDims)`.
    ///   - indices: The selected experts, shape `(..., routes)`.
    /// - Returns: One output for each route, shape `(..., routes, inputDims)`.
    ///   The caller weighs them and adds them up.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let routed = MLX.expandedDimensions(x, axes: Self.routeAxes)
        guard indices.size >= Self.sortThreshold else {
            let outputs = expertOutputs(routed, indices, sortedByExpert: false)
            return MLX.squeezed(outputs, axis: Self.expertAxis)
        }
        let (sortedInput, sortedIndices, inverseOrder) = gatherSort(x: routed, indices: indices)
        let sortedOutputs = expertOutputs(sortedInput, sortedIndices, sortedByExpert: true)
        let outputs = scatterUnsort(
            x: sortedOutputs, invOrder: inverseOrder, shape: indices.shape)
        return MLX.squeezed(outputs, axis: Self.expertAxis)
    }

    /// Reads the three projections of the selected experts.
    ///
    /// - Parameters:
    ///   - x: The token values, already carrying the route axes.
    ///   - indices: The selected experts, one for each row of `x`.
    ///   - sortedByExpert: True when `gatherSort` already grouped the rows by
    ///     expert, which lets the gathered matrix multiply take a faster path.
    /// - Returns: The output of each route.
    private func expertOutputs(
        _ x: MLXArray, _ indices: MLXArray, sortedByExpert: Bool
    ) -> MLXArray {
        let gate = gateProj(x, indices, sortedIndices: sortedByExpert)
        let up = upProj(x, indices, sortedIndices: sortedByExpert)
        let activated = DeepseekV4Math.clampedSwiGLU(gate: gate, up: up, limit: swigluLimit)
        return downProj(activated, indices, sortedIndices: sortedByExpert)
    }
}

// MARK: - Routing gate

/// The routing gate of one DeepSeek-V4 mixture-of-experts layer.
///
/// The gate answers which routed experts each token reads, and how much of
/// each expert output that token keeps.
///
/// Two paths reach that answer, and the layer index decides which one:
///
/// - A **hash layer**, one of the first `num_hash_layers` layers, reads the
///   `tid2eid` table of the checkpoint. The token identifier alone names the
///   experts, thus the hidden state cannot move the selection.
/// - Every **later layer** scores the experts as `sqrt(softplus(logits))`,
///   adds the routing bias, and takes the highest `num_experts_per_tok`
///   scores.
///
/// The bias belongs to the SELECTION and not to the weights. The gate adds it
/// before it picks the experts and gathers the UNBIASED score of each expert
/// it picked. A gate that gathered the biased score would run, and would
/// weigh a lifted expert far too heavily.
class DeepseekV4MoEGate: Module {

    /// The number of routed experts one token reads.
    let expertsPerToken: Int

    /// The number of routed experts in this layer.
    let routedExpertCount: Int

    /// The factor the selected weights take.
    let routedScalingFactor: Float

    /// True when the selected weights divide by their own sum.
    let normalizesTopkProbabilities: Bool

    /// True when this layer routes through the hash table.
    let isHashLayer: Bool

    /// The gate projection, shape `(routedExpertCount, hiddenSize)`. It is a
    /// raw parameter and not a `Linear`, because the projection runs in
    /// float32 whatever the activation dtype is.
    @ParameterInfo(key: "weight") var weight: MLXArray

    /// The routing bias, one value for each routed expert. It joins the
    /// scores for the selection only.
    @ParameterInfo(key: "bias") var bias: MLXArray

    /// The hash table, from token identifier to expert identifier, shape
    /// `(vocabSize, expertsPerToken)`.
    ///
    /// A layer that does not route through the hash table still holds this
    /// parameter, at the placeholder shape below, because the reference does
    /// and because a checkpoint that carries the tensor then loads.
    @ParameterInfo(key: "tid2eid") var tokenToExpert: MLXArray

    /// The shape the hash table takes on a layer that never reads it.
    private static let unusedHashTableShape = [1, 1]

    /// The epsilon the normalization adds, so that a row of scores that adds
    /// up to zero cannot divide by zero.
    private static let normalizeEpsilon: Float = 1e-20

    /// Builds the gate of one layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this gate belongs to.
    init(configuration: DeepseekV4Configuration, layer: Int) {
        let hashLayer = configuration.isHashLayer(layer)
        self.expertsPerToken = configuration.numExpertsPerTok
        self.routedExpertCount = configuration.nRoutedExperts
        self.routedScalingFactor = configuration.routedScalingFactor
        self.normalizesTopkProbabilities = configuration.normalizeTopkProb
        self.isHashLayer = hashLayer
        self._weight.wrappedValue = zeros([
            configuration.nRoutedExperts, configuration.hiddenSize,
        ])
        self._bias.wrappedValue = zeros([configuration.nRoutedExperts])
        self._tokenToExpert.wrappedValue = zeros(
            hashLayer
                ? [configuration.vocabSize, configuration.numExpertsPerTok]
                : Self.unusedHashTableShape)
    }

    /// Routes one block of tokens.
    ///
    /// - Parameters:
    ///   - x: The token values, shape `(batch, tokens, hiddenSize)`.
    ///   - inputIds: The token identifiers, shape `(batch, tokens)`. A hash
    ///     layer reads them and every later layer passes over them.
    /// - Returns: The selected experts and their weights, each of shape
    ///   `(batch, tokens, expertsPerToken)`.
    func callAsFunction(
        _ x: MLXArray, inputIds: MLXArray
    ) -> (indices: MLXArray, weights: MLXArray) {
        let logits = x.asType(.float32).matmul(weight.asType(.float32).transposed())
        let scores = DeepseekV4Math.sqrtSoftplus(logits)
        let indices = selectedExperts(scores: scores, inputIds: inputIds)
        return (indices.asType(.uint32), routedWeights(gatheredFrom: scores, at: indices))
    }

    /// Names the experts one block of tokens reads.
    ///
    /// This is the only place the routing bias reaches. The scores this
    /// function reads leave it unchanged, thus the caller cannot gather a
    /// biased weight by mistake.
    private func selectedExperts(scores: MLXArray, inputIds: MLXArray) -> MLXArray {
        guard isHashLayer else {
            let biased = scores + bias.asType(.float32)
            return argPartition(-biased, kth: expertsPerToken - 1, axis: -1)[
                .ellipsis, ..<expertsPerToken]
        }
        return tokenToExpert[inputIds].asType(.int32)
    }

    /// Reads the weight of each selected expert out of the UNBIASED scores.
    private func routedWeights(gatheredFrom scores: MLXArray, at indices: MLXArray) -> MLXArray {
        var weights = takeAlong(scores, indices, axis: -1)
        if normalizesTopkProbabilities {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + Self.normalizeEpsilon)
        }
        return weights * routedScalingFactor
    }
}

// MARK: - Mixture of experts

/// One DeepSeek-V4 mixture-of-experts layer.
///
/// The layer sends each token through the routed experts its gate selected,
/// adds those outputs up in float32, and adds the shared expert, which every
/// token reads.
class DeepseekV4MoE: Module {

    @ModuleInfo(key: "switch_mlp") var switchMLP: DeepseekV4SwitchGLU
    @ModuleInfo(key: "gate") var gate: DeepseekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4MLP?

    /// Builds one mixture-of-experts layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this mixture belongs to.
    init(configuration: DeepseekV4Configuration, layer: Int) {
        self._switchMLP.wrappedValue = DeepseekV4SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.moeIntermediateSize,
            expertCount: configuration.nRoutedExperts,
            swigluLimit: configuration.swigluLimit)
        self._gate.wrappedValue = DeepseekV4MoEGate(configuration: configuration, layer: layer)
        if configuration.numSharedExperts > 0 {
            self._sharedExperts.wrappedValue = DeepseekV4MLP(
                hiddenSize: configuration.hiddenSize,
                intermediateSize: configuration.moeIntermediateSize
                    * configuration.numSharedExperts,
                swigluLimit: configuration.swigluLimit)
        }
    }

    /// Reads one block of tokens.
    ///
    /// - Parameters:
    ///   - x: The block input, shape `(batch, tokens, hiddenSize)`.
    ///   - inputIds: The token identifiers, shape `(batch, tokens)`. A hash
    ///     layer routes by them.
    /// - Returns: The block output, of the shape and the dtype of `x`.
    func callAsFunction(_ x: MLXArray, inputIds: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x, inputIds: inputIds)
        let weighted =
            switchMLP(x, indices).asType(.float32) * MLX.expandedDimensions(weights, axis: -1)
        var y = DeepseekV4Math.reduceRoutedExpertsFP32(weighted)
        if let sharedExperts {
            y = y + sharedExperts(x).asType(.float32)
        }
        return y.asType(x.dtype)
    }
}
