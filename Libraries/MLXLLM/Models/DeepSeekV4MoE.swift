// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Four details do not come from that file:
//
//  1. The routed experts. The file above calls a `SwitchGLU` of its own that
//     takes a two-argument activation, and it carries fused Metal kernels for
//     the mxfp4 and mxfp8 expert weights. This repository keeps its own
//     `Libraries/MLXLMCommon/SwitchLayers.swift`, whose `SwitchGLU` takes a
//     one-argument activation and thus cannot state the asymmetric DeepSeek-V4
//     clamp, and whose `FusedGateUpSwitchGLU` reads one fused `gate_up_proj`
//     tensor that this checkpoint does not ship. ``DeepSeekV4SwitchGLU`` below
//     therefore builds on `SwitchLinear`, `gatherSort` and `scatterUnsort`.
//     The fused kernels stay out of scope.
//  2. The float32 reduction. The file above sums the weighted expert outputs
//     with `(y * scores[..., .newAxis]).sum(axis: -2)`. This file calls
//     `DeepSeekV4Math.reduceRoutedExpertsFP32`, which states the float32
//     contract in one place, and casts back one time at the end.
//  3. The token identifiers. The file above threads them into the gate
//     through a mutable `currentInputIds` property, so that the layer can
//     conform to `UnaryLayer`. Every DeepSeek-V4 layer holds a mixture of
//     experts, thus a decoder layer can hold this type itself and hand the
//     identifiers over as an argument. This file takes them as an argument
//     and holds no state.
//  4. The routing parameters of the gate. The file above declares `tid2eid`
//     AND `bias` on every layer, and this file did the same until task
//     `^3zest44`. That decision is REVERSED here, and the doc comments below
//     record the reversal. The DeepSeek-V4 Python reference (Thump604/mlx-lm
//     @ deepseek-v4-support-fixes, mlx_lm/models/deepseek_v4.py,
//     `MoEGate.__init__`) declares one of the two on each layer and never
//     both, and the published checkpoint carries that set: `tid2eid` on the
//     hash layers alone, and `bias` on every layer after them.
//     `MLXLMCommon.loadWeights` verifies with `.allModelKeysSet`, which
//     throws for a module parameter no weight fills, thus a gate that
//     declared both could load no real checkpoint. Each parameter below is
//     therefore an `Optional`, and a layer builds the one it holds.
//
// The shared expert READS the SwiGLU clamp. Task `^kp1pnj4` decided this
// against the training-time reference of the checkpoint. The checkpoint
// `deepseek-ai/DeepSeek-V4-Flash` publishes no modeling file of its own, and
// its `config.json` (`"model_type": "deepseek_v4"`, no `auto_map`) binds to
// the native `deepseek_v4` model of Hugging Face `transformers`. There,
// `DeepseekV4SparseMoeBlock.__init__` builds the shared expert as
// `DeepseekV4MLP(config)`, and that MLP clamps its gate and its up
// projection with `config.swiglu_limit`. The `swiglu_limit=0.0` that
// Thump604/mlx-lm @ deepseek-v4-support-fixes (`DeepseekV4MoE.__init__`)
// gives the shared expert diverges from that reference. The test
// `DeepSeekV4MoETests/theSharedExpertReadsTheClampOfTheCheckpoint()` pins
// this decision.

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
final class DeepSeekV4MLP: Module, UnaryLayer {

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
            DeepSeekV4Math.clampedSwiGLU(gate: gateProj(x), up: upProj(x), limit: swigluLimit))
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
final class DeepSeekV4SwitchGLU: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    /// The `swiglu_limit` each expert clamps its activation with.
    let swigluLimit: Float

    /// The number of routes a block must hold before a sort by expert pays
    /// for itself. `SwitchGLU` reads the same number, thus the two routed
    /// paths of this repository start to sort at the same point.
    private static let sortThreshold = 64

    /// The axis positions of a `(batch, tokens, routes, rows, width)` tensor,
    /// counted from the end.
    ///
    /// A token carries the width axis alone when it reaches this layer. It
    /// gains two more axes before the experts read it, and it loses the row
    /// axis again when they give their answer back.
    private enum RoutedAxis {
        /// The last axis, which holds the numbers of one token.
        static let width = -1
        /// The axis before ``width``, which the gathered matrix multiply
        /// reads as the one row of a matrix.
        static let row = width - 1
        /// The axis before ``row``, which holds one route of a token.
        static let route = row - 1
    }

    /// The two axes a token gains before it reaches the experts.
    private static let routeAxes = [RoutedAxis.row, RoutedAxis.route]

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
            return MLX.squeezed(outputs, axis: RoutedAxis.row)
        }
        let (sortedInput, sortedIndices, inverseOrder) = gatherSort(x: routed, indices: indices)
        let sortedOutputs = expertOutputs(sortedInput, sortedIndices, sortedByExpert: true)
        let outputs = scatterUnsort(
            x: sortedOutputs, invOrder: inverseOrder, shape: indices.shape)
        return MLX.squeezed(outputs, axis: RoutedAxis.row)
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
        let activated = DeepSeekV4Math.clampedSwiGLU(gate: gate, up: up, limit: swigluLimit)
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
final class DeepSeekV4MoEGate: Module {

    /// The number of routed experts one token reads.
    let expertsPerToken: Int

    /// The factor the selected weights take.
    let routedScalingFactor: Float

    /// True when the selected weights divide by their own sum.
    let normalizesTopkProbabilities: Bool

    /// True when this layer routes through the hash table.
    ///
    /// The table is the one parameter a hash layer holds and a later layer
    /// does not, thus its presence answers the question on its own and the
    /// gate keeps no second copy of the answer.
    var isHashLayer: Bool { tokenToExpert != nil }

    /// The gate projection, shape `(routed experts, hiddenSize)`. It is a
    /// raw parameter and not a `Linear`, because the projection runs in
    /// float32 whatever the activation dtype is.
    @ParameterInfo(key: "weight") var weight: MLXArray

    /// The routing bias, one value for each routed expert. It joins the
    /// scores for the selection only.
    ///
    /// A hash layer selects its experts from the table and scores none of
    /// them, thus it holds no bias and this parameter is `nil` there. The
    /// checkpoint carries the tensor on the later layers alone.
    @ParameterInfo(key: "bias") var bias: MLXArray?

    /// The hash table, from token identifier to expert identifier, shape
    /// `(vocabSize, expertsPerToken)`.
    ///
    /// A layer that does not route through the hash table holds no table, and
    /// this parameter is `nil` there. An earlier version of this file gave
    /// such a layer a placeholder table, because the Swift reference declares
    /// one on every layer. Task `^3zest44` REVERSED that decision: the Swift
    /// reference builds a load path of its own, the DeepSeek-V4 Python
    /// reference the checkpoint was written for declares the table on a hash
    /// layer alone, and `MLXLMCommon.loadWeights` verifies with
    /// `.allModelKeysSet`, which throws for a parameter no checkpoint tensor
    /// fills.
    @ParameterInfo(key: "tid2eid") var tokenToExpert: MLXArray?

    /// The epsilon the normalization adds, so that a row of scores that adds
    /// up to zero cannot divide by zero.
    private static let normalizeEpsilon: Float = 1e-20

    /// The axis of a `(batch, tokens, experts)` score tensor that holds one
    /// score for each routed expert. It is the last axis.
    private static let expertScoreAxis = -1

    /// Builds the gate of one layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this gate belongs to.
    init(configuration: DeepSeekV4Configuration, layer: Int) {
        let hashLayer = configuration.isHashLayer(layer)
        self.expertsPerToken = configuration.numExpertsPerTok
        self.routedScalingFactor = configuration.routedScalingFactor
        self.normalizesTopkProbabilities = configuration.normalizeTopkProb
        self._weight.wrappedValue = zeros([
            configuration.nRoutedExperts, configuration.hiddenSize,
        ])
        if hashLayer {
            self._tokenToExpert.wrappedValue = zeros([
                configuration.vocabSize, configuration.numExpertsPerTok,
            ])
        } else {
            self._bias.wrappedValue = zeros([configuration.nRoutedExperts])
        }
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
        let scores = DeepSeekV4Math.sqrtSoftplus(logits)
        let indices = selectedExperts(scores: scores, inputIds: inputIds)
        return (indices.asType(.uint32), routedWeights(gatheredFrom: scores, at: indices))
    }

    /// Names the experts one block of tokens reads.
    ///
    /// Each layer holds one routing parameter alone, thus this function reads
    /// the one the layer has: a hash layer reads its table, and every later
    /// layer scores the experts and adds its bias.
    ///
    /// This is the only place the routing bias reaches. The scores this
    /// function reads leave it unchanged, thus the caller cannot gather a
    /// biased weight by mistake.
    private func selectedExperts(scores: MLXArray, inputIds: MLXArray) -> MLXArray {
        if let tokenToExpert {
            return tokenToExpert[inputIds].asType(.int32)
        }
        let biased = bias.map { scores + $0.asType(.float32) } ?? scores
        let lastSelectedPosition = expertsPerToken - 1
        return argPartition(-biased, kth: lastSelectedPosition, axis: Self.expertScoreAxis)[
            .ellipsis, ..<expertsPerToken]
    }

    /// Reads the weight of each selected expert out of the UNBIASED scores.
    private func routedWeights(gatheredFrom scores: MLXArray, at indices: MLXArray) -> MLXArray {
        var weights = takeAlong(scores, indices, axis: Self.expertScoreAxis)
        if normalizesTopkProbabilities {
            weights =
                weights
                / (weights.sum(axis: Self.expertScoreAxis, keepDims: true) + Self.normalizeEpsilon)
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
final class DeepSeekV4MoE: Module {

    @ModuleInfo(key: "switch_mlp") var switchMLP: DeepSeekV4SwitchGLU
    @ModuleInfo(key: "gate") var gate: DeepSeekV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepSeekV4MLP?

    /// The axis a routed weight gains, so that one weight broadcasts against
    /// every number of the expert output it weighs. It is the last axis.
    private static let widthAxis = -1

    /// Builds one mixture-of-experts layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of the decoder layer this mixture belongs to.
    init(configuration: DeepSeekV4Configuration, layer: Int) {
        self._switchMLP.wrappedValue = DeepSeekV4SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.moeIntermediateSize,
            expertCount: configuration.nRoutedExperts,
            swigluLimit: configuration.swigluLimit)
        self._gate.wrappedValue = DeepSeekV4MoEGate(configuration: configuration, layer: layer)
        if configuration.numSharedExperts > 0 {
            self._sharedExperts.wrappedValue = DeepSeekV4MLP(
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
            switchMLP(x, indices).asType(.float32)
            * MLX.expandedDimensions(weights, axis: Self.widthAxis)
        var y = DeepSeekV4Math.reduceRoutedExpertsFP32(weighted)
        if let sharedExperts {
            y = y + sharedExperts(x).asType(.float32)
        }
        return y.asType(x.dtype)
    }
}
