// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Building-block tests for MiniMax-M3's swigluoai activation and sparse MoE
/// block (kanban ^mv9aq7w). No decoder/model yet -- ^xgvth41 builds the full
/// `MiniMaxM3TextConfiguration` + dense-attention decoder on top of these.
///
/// Reference: mlx-vlm `mlx_vlm/models/minimax_m3_vl/language.py`
/// (`MiniMaxSwiGLUOAI`, `MiniMaxPackedSwitchGLU`, `MiniMaxSparseMoeBlock`,
/// `_minimax_moe_select`), cross-checked against upstream mlx-lm PRs
/// #1398/#1401 and the `mlx-community/MiniMax-M3-4bit` checkpoint's
/// `model.safetensors.index.json` / safetensors shard headers.
struct MiniMaxM3Tests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - MiniMaxM3SwiGLUOAI

    /// mlx-vlm's `_swiglu_oai` reference formula:
    ///   x_glu = clip(x_glu, max: limit)                  // upper bound only
    ///   x_linear = clip(x_linear, min: -limit, max: limit)
    ///   return x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)
    private func referenceSwiGLUOAI(
        xLinear: Float, xGlu: Float, alpha: Float, limit: Float, beta: Float
    ) -> Float {
        let glu = min(xGlu, limit)
        let linear = max(min(xLinear, limit), -limit)
        let sig = 1 / (1 + expf(-alpha * glu))
        return glu * sig * (linear + beta)
    }

    @Test("interior values match the reference formula")
    func interiorMatchesReferenceFormula() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.5
        let xGlu: Float = -0.25
        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("gate branch clips only the upper bound, leaving very negative gates untouched")
    func gateClipsUpperBoundOnly() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.0
        let xGlu: Float = -8.0  // below -limit

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        // Correct (asymmetric) formula: gate stays -8.
        let expectedUnclamped = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        // A wrong symmetric clamp would instead use gate == -7.
        let expectedIfWronglySymmetric = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: -7.0, alpha: 1.702, limit: 7.0, beta: 1.0)

        #expect(abs(out.item(Float.self) - expectedUnclamped) < 1e-8)
        #expect(abs(out.item(Float.self) - expectedIfWronglySymmetric) > 1e-6)
    }

    @Test("gate branch clips at the upper limit boundary")
    func gateClipsAtUpperBoundary() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xLinear: Float = 0.25
        let xGlu: Float = 12.0  // above limit

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: 7.0, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("linear branch clips symmetrically at both bounds", arguments: [12.0 as Float, -12.0 as Float])
    func linearClipsSymmetrically(xLinear: Float) {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.702, limit: 7.0, beta: 1.0)
        let xGlu: Float = 1.0

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let clampedLinear: Float = xLinear > 0 ? 7.0 : -7.0
        let expected = referenceSwiGLUOAI(
            xLinear: clampedLinear, xGlu: xGlu, alpha: 1.702, limit: 7.0, beta: 1.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    @Test("alpha, limit, and beta are configuration-driven, not hardcoded")
    func configDrivenParameters() {
        let activation = MiniMaxM3SwiGLUOAI(alpha: 1.0, limit: 3.0, beta: 0.0)
        let xLinear: Float = 1.0
        let xGlu: Float = 5.0  // above the custom limit of 3.0

        let out = activation(MLXArray(xLinear), MLXArray(xGlu))
        eval(out)

        let expected = referenceSwiGLUOAI(
            xLinear: xLinear, xGlu: 3.0, alpha: 1.0, limit: 3.0, beta: 0.0)
        #expect(abs(out.item(Float.self) - expected) < 1e-5)
    }

    // MARK: - MiniMaxM3SparseMoeBlock

    private func tinyConfig(
        numLocalExperts: Int = 8, numExpertsPerTok: Int = 2, hiddenSize: Int = 16,
        intermediateSize: Int = 32
    ) -> MiniMaxM3MoEConfiguration {
        MiniMaxM3MoEConfiguration(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numLocalExperts: numLocalExperts,
            numExpertsPerTok: numExpertsPerTok
        )
    }

    @Test("output shape is (B, L, hiddenSize)")
    func outputShapeMatchesInput() {
        MLXRandom.seed(0)
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        let x = MLXRandom.normal([2, 3, config.hiddenSize])
        let y = block(x)
        eval(y)

        #expect(y.shape == [2, 3, config.hiddenSize])
    }

    @Test("switch_mlp packs the shared expert as one extra expert row")
    func switchMLPHasOneMoreExpertThanRouted() {
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        #expect(block.switchMLP.numExperts == config.numLocalExperts + 1)
    }

    @Test("zeroing the packed shared-expert weights changes the output")
    func sharedExpertContributionIsObservable() throws {
        MLXRandom.seed(1)
        let config = tinyConfig()
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        let x = MLXRandom.normal([1, 4, config.hiddenSize])
        let baseline = block(x)
        eval(baseline)

        // The packed shared expert is the last (index numLocalExperts) row of
        // the fused switch_mlp weight tensors -- zero it out.
        let gateUp = block.switchMLP.gateUpProj.weight
        gateUp[config.numLocalExperts...] = MLXArray.zeros(like: gateUp[config.numLocalExperts...])
        let down = block.switchMLP.downProj.weight
        down[config.numLocalExperts...] = MLXArray.zeros(like: down[config.numLocalExperts...])

        try block.update(
            parameters: ModuleParameters.unflattened([
                "switch_mlp.gate_up_proj.weight": gateUp,
                "switch_mlp.down_proj.weight": down,
            ]), verify: [])

        let zeroedShared = block(x)
        eval(zeroedShared)

        let maxDiff = MLX.abs(baseline - zeroedShared).max().item(Float.self)
        #expect(maxDiff > 1e-6)
    }

    @Test("routing selects via biased scores but weights via unbiased, normalized, scaled scores")
    func routingUsesBiasedSelectionAndUnbiasedScaledWeights() throws {
        // 4 routed experts, top-2, hidden == numLocalExperts so the router
        // can be an identity matrix (logit_i == x_i), mirroring
        // LFM2MoeRoutingTests's style.
        let config = MiniMaxM3MoEConfiguration(
            hiddenSize: 4, intermediateSize: 8, numLocalExperts: 4, numExpertsPerTok: 2,
            routedScalingFactor: 2.0)
        let block = MiniMaxM3SparseMoeBlock(config)
        try block.update(
            parameters: ModuleParameters.unflattened([
                "gate.weight": MLX.identity(4),
                "e_score_correction_bias": MLXArray([Float(0), 0, 1, 0]),
            ]), verify: [])
        eval(block)

        let logits: [Float] = [2, 1, 0, -1]
        let x = MLXArray(logits).reshaped(1, 1, 4)

        let (indices, weights) = block.route(x)
        eval(indices, weights)

        func sig(_ v: Float) -> Float { 1 / (1 + expf(-v)) }

        let idx = indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = weights.reshaped(-1).asArray(Float.self)
        let routed = Dictionary(uniqueKeysWithValues: zip(idx, w))

        // Selection uses the *biased* scores: expert 2's correction bias of
        // +1 promotes it ahead of expert 1 into the top-2, even though
        // sigmoid(0) < sigmoid(1) on the raw (unbiased) scores.
        #expect(Set(routed.keys) == [0, 2])

        // Weighting uses the *unbiased* scores, renormalized to sum to 1,
        // then scaled by routed_scaling_factor (2.0).
        let denom = sig(2) + sig(0)
        #expect(abs((routed[0] ?? .nan) - (sig(2) / denom) * 2.0) < 1e-4)
        #expect(abs((routed[2] ?? .nan) - (sig(0) / denom) * 2.0) < 1e-4)
    }

    @Test("the packed shared expert contributes with unscaled weight 1.0, not routed_scaling_factor")
    func sharedExpertWeightIsUnscaled() throws {
        // A distinctive routed_scaling_factor far from 1.0: if the shared
        // expert's weight were (incorrectly) also multiplied by it, the
        // assertion below would fail.
        let config = MiniMaxM3MoEConfiguration(
            hiddenSize: 4, intermediateSize: 8, numLocalExperts: 4, numExpertsPerTok: 2,
            routedScalingFactor: 5.0)
        let block = MiniMaxM3SparseMoeBlock(config)
        eval(block)

        // Zero every *routed* expert's weights (rows 0..<numLocalExperts).
        // Since swigluoai(0, 0) == 0, their contribution is exactly zero
        // regardless of which are selected or how they're weighted, so
        // `block(x)` reduces to exactly `1.0 * sharedExpert(x)`.
        let gateUp = block.switchMLP.gateUpProj.weight
        gateUp[..<config.numLocalExperts] = MLXArray.zeros(
            like: gateUp[..<config.numLocalExperts])
        let down = block.switchMLP.downProj.weight
        down[..<config.numLocalExperts] = MLXArray.zeros(like: down[..<config.numLocalExperts])

        try block.update(
            parameters: ModuleParameters.unflattened([
                "switch_mlp.gate_up_proj.weight": gateUp,
                "switch_mlp.down_proj.weight": down,
            ]), verify: [])

        let x = MLXRandom.normal([1, 2, config.hiddenSize])
        let output = block(x)
        eval(output)

        // Manually compute the shared expert's output with an explicit
        // weight of 1.0, independent of `routed_scaling_factor`.
        let sharedIndices = MLXArray.full(
            [1, 2, 1], values: MLXArray(Int32(config.numLocalExperts)), dtype: .int32)
        let sharedOnly = block.switchMLP(x, sharedIndices).squeezed(axis: -2)
        eval(sharedOnly)

        let maxDiff = MLX.abs(output - sharedOnly).max().item(Float.self)
        #expect(maxDiff < 1e-4)
    }
}
