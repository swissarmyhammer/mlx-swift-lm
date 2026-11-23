// Copyright © 2026 Apple Inc.
//
// The sparse mixture of experts of Inkling 2: the router, the quantized experts and their decode
// paths.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Mixture of experts

/// The epsilon that keeps the sum of the routed expert weights above zero.
private let routeWeightEpsilon: Float = 1e-20

/// `softcap(gate) * sigmoid(gate) * softcap(up)` in float32. The gate and up columns of
/// the fused projection alternate.
private func softcappedSwiGLU(gateSoftcap: Float, upSoftcap: Float)
    -> @Sendable (MLXArray, MLXArray) -> MLXArray
{
    compile(shapeless: true) { gate, up in
        let dtype = gate.dtype
        let gate = gate.asType(.float32)
        let up = up.asType(.float32)
        let gateCapped = tanh(gate / gateSoftcap) * gateSoftcap
        let upCapped = tanh(up / upSoftcap) * upSoftcap
        return (gateCapped * sigmoid(gate) * upCapped).asType(dtype)
    }
}

/// The routed experts' outputs `[..., K, D]` weighted by `[..., K, 1]` and summed in float32.
private let weightedExperts: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs.asType(.float32) * weights).sum(axis: matrixRowAxis).asType(outputs.dtype)
}

/// `residual + x` in float32, rounded to `x`'s dtype.
private let residualAdd: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    residual, x in
    (residual + x.asType(.float32)).asType(x.dtype)
}

/// One projection of all experts, quantized in the checkpoint's own format.
final class Inkling2ExpertProjection: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var scales: MLXArray
    @ParameterInfo var biases: MLXArray?
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?

    let format: Inkling2ExpertFormat

    init(experts: Int, inputDims: Int, outputDims: Int, format: Inkling2ExpertFormat) {
        self.format = format
        let packed = inputDims * format.bits / UInt32.bitWidth
        let groups = inputDims / format.groupSize
        self._weight.wrappedValue = MLXArray.zeros([experts, outputDims, packed], dtype: .uint32)
        let scaleType: DType = format.mode == .affine ? .bfloat16 : .uint8
        self._scales.wrappedValue = MLXArray.zeros(
            [experts, outputDims, groups], dtype: scaleType)
        if format.mode == .affine {
            self._biases.wrappedValue = MLXArray.zeros(
                [experts, outputDims, groups], dtype: .bfloat16)
        }
        if format.hasGlobalScale {
            self._globalScale.wrappedValue = MLXArray.ones([experts], dtype: .float32)
        }
        super.init()
        freeze()
    }

    var tensors: Inkling2ExpertTensors {
        .init(
            weight: weight, scales: scales, biases: biases, globalScale: globalScale,
            format: format)
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices, transpose: true,
            groupSize: format.groupSize, bits: format.bits, mode: format.mode,
            globalScale: globalScale, sortedIndices: sorted)
    }
}

/// The quantization of an expert projection, read from the checkpoint's tensor shapes.
struct Inkling2ExpertFormat: Equatable, Sendable {
    let mode: QuantizationMode
    let bits: Int
    let groupSize: Int
    let hasGlobalScale: Bool

    /// The bits of the 4-bit and the 8-bit formats.
    private static let fourBits = 4
    private static let eightBits = 8

    /// The group sizes of the NVIDIA and the OCP microscaled formats.
    private static let nvfp4GroupSize = 16
    private static let microscaledGroupSize = 32

    static let nvfp4 = Inkling2ExpertFormat(
        mode: .nvfp4, bits: fourBits, groupSize: nvfp4GroupSize, hasGlobalScale: true)

    init(mode: QuantizationMode, bits: Int, groupSize: Int, hasGlobalScale: Bool) {
        self.mode = mode
        self.bits = bits
        self.groupSize = groupSize
        self.hasGlobalScale = hasGlobalScale
    }

    /// The format of the tensors at `path`, or `nil` when they are not quantized.
    init?(path: String, weights: [String: MLXArray], inputDims: Int) {
        guard let weight = weights["\(path).weight"], let scales = weights["\(path).scales"],
            weight.dtype == .uint32
        else { return nil }
        let bits = weight.dim(-1) * UInt32.bitWidth / inputDims
        let groupSize = inputDims / scales.dim(-1)
        if weights["\(path).biases"] != nil {
            self.init(mode: .affine, bits: bits, groupSize: groupSize, hasGlobalScale: false)
            return
        }
        let mode: QuantizationMode
        switch (bits, groupSize) {
        case (Self.fourBits, Self.nvfp4GroupSize): mode = .nvfp4
        case (Self.fourBits, Self.microscaledGroupSize): mode = .mxfp4
        case (Self.eightBits, Self.microscaledGroupSize): mode = .mxfp8
        default: return nil
        }
        self.init(
            mode: mode, bits: bits, groupSize: groupSize,
            hasGlobalScale: weights["\(path).global_scale"] != nil)
    }
}

final class Inkling2Experts: Module {
    /// The rows alternate between gate and up, as in the checkpoint.
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Inkling2ExpertProjection
    @ModuleInfo(key: "down_proj") var downProj: Inkling2ExpertProjection

    private let activation: @Sendable (MLXArray, MLXArray) -> MLXArray
    private let softcaps: Inkling2ConstantBox

    init(
        _ config: Inkling2TextConfiguration, gateUp: Inkling2ExpertFormat,
        down: Inkling2ExpertFormat
    ) {
        let (dims, hidden) = (config.hiddenSize, config.intermediateSize)
        _gateUpProj.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: dims, outputDims: gateUpRowsPerChannel * hidden,
            format: gateUp)
        _downProj.wrappedValue = Inkling2ExpertProjection(
            experts: config.experts, inputDims: hidden, outputDims: dims, format: down)
        activation = softcappedSwiGLU(gateSoftcap: config.gateSoftcap, upSoftcap: config.upSoftcap)
        softcaps = Inkling2ConstantBox(MLXArray([config.gateSoftcap, config.upSoftcap]))
        super.init()
    }

    /// The routed rows from which the experts gather their rows sorted by expert.
    private static let sortedRowsThreshold = 64

    /// The most experts per token, and the multiples of the hidden and the input widths, that the
    /// fused decode kernels take.
    private static let fusedDecodeMaximumExperts = metalSimdWidth
    private static let fusedDecodeHiddenMultiple = 4
    private static let fusedDecodeWidthMultiple = 2

    /// `[..., K, D]` outputs of the `[..., K]` experts of each row of `x`.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let (out, order) = outputs(x, indices)
        guard let order else { return out.reshaped(indices.shape + [out.dim(-1)]) }
        return scatterUnsort(x: out, invOrder: order, shape: indices.shape)
    }

    /// The expert outputs `[N * K, D]` of the rows of `x` and, when they are sorted by expert,
    /// the row of each output: output `i` of the flattened `indices` is at row `order[i]`.
    func outputs(_ x: MLXArray, _ indices: MLXArray) -> (MLXArray, order: MLXArray?) {
        var x = MLX.expandedDimensions(x, axes: [matrixRowAxis, matrixRowAxis - 1])
        let sort = indices.size >= Self.sortedRowsThreshold
        var idx = indices
        var inverseOrder: MLXArray?
        if sort {
            let sorted = gatherSort(x: x, indices: indices)
            (x, idx, inverseOrder) = (sorted.0, sorted.1, sorted.2)
        }
        let gateUp = gateUpProj(x, idx, sorted: sort)
        let activated = activation(
            gateUp[.ellipsis, .stride(from: 0, by: gateUpRowsPerChannel)],
            gateUp[.ellipsis, .stride(from: 1, by: gateUpRowsPerChannel)])
        let out = downProj(activated, idx, sorted: sort)
        return (out.reshaped(-1, out.dim(-1)), inverseOrder)
    }

    /// The layer output `residual + conv(sum(weights * experts(x)))` of one token per row and
    /// the next convolution state. Up to ``Inkling2/speculativeVerifyPositions`` rows run
    /// through the fused expert kernels.
    func decode(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray, residual: MLXArray, state: MLXArray,
        convolutionWeight: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (gateUp, down) = (gateUpProj.tensors, downProj.tensors)
        let (dims, hidden) = (down.weight.dim(1), gateUp.weight.dim(1) / gateUpRowsPerChannel)
        // Up to the rows of a verify pass, so that each row of a verify computes what a decode
        // step of that row computes.
        if indices.dim(0) <= Inkling2.speculativeVerifyPositions,
            indices.dim(-1) <= Self.fusedDecodeMaximumExperts,
            hidden.isMultiple(of: Self.fusedDecodeHiddenMultiple),
            dims.isMultiple(of: Self.fusedDecodeWidthMultiple),
            gateUp.supportsFusedDecode(inputDims: dims),
            down.supportsFusedDecode(inputDims: hidden)
        {
            let activations = inkling2ExpertGateUp(
                x, experts: gateUp, indices: indices, softcaps: softcaps.value)
            return inkling2ExpertDown(
                activations, experts: down, indices: indices, weights: weights,
                residual: residual, state: state, convolutionWeight: convolutionWeight)
        }
        return inkling2ExpertEpilogue(
            outputs: self(x, indices), weights: weights, residual: residual, state: state,
            convolutionWeight: convolutionWeight)
    }

    /// ``decode(_:indices:weights:residual:state:convolutionWeight:)`` that routes inside the
    /// gate and up kernel, or `nil` when the fused kernels do not take these shapes.
    func decode(
        _ x: MLXArray, logits: MLXArray, bias: MLXArray, topK: Int, scoreEps: Float,
        routeScale: Float, residual: MLXArray, state: MLXArray, convolutionWeight: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let (gateUp, down) = (gateUpProj.tensors, downProj.tensors)
        let (dims, hidden) = (down.weight.dim(1), gateUp.weight.dim(1) / gateUpRowsPerChannel)
        guard logits.dim(0) <= Inkling2.speculativeVerifyPositions,
            topK <= Self.fusedDecodeMaximumExperts,
            hidden.isMultiple(of: Self.fusedDecodeHiddenMultiple),
            dims.isMultiple(of: Self.fusedDecodeWidthMultiple),
            gateUp.supportsFusedDecode(inputDims: dims),
            down.supportsFusedDecode(inputDims: hidden)
        else { return nil }
        let routed = inkling2RoutedExpertGateUp(
            x, experts: gateUp, logits: logits, bias: bias, topK: topK, scoreEps: scoreEps,
            routeScale: routeScale, softcaps: softcaps.value)
        return inkling2ExpertDown(
            routed.activations, experts: down, indices: routed.indices, weights: routed.weights,
            residual: residual, state: state, convolutionWeight: convolutionWeight)
    }
}

private final class Inkling2Router: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray

    let topK: Int
    let routeScale: Float

    init(_ config: Inkling2TextConfiguration) {
        topK = config.expertsPerToken
        routeScale = config.routeScale
        _weight.wrappedValue = MLXArray.zeros([config.experts, config.hiddenSize], dtype: .float32)
        _bias.wrappedValue = MLXArray.zeros([config.experts], dtype: .float32)
        super.init()
    }

    /// The float32 router logits `[N, E]` of the rows of `x` from the logits kernel, or `nil`
    /// when the rows do not take it.
    func kernelLogits(_ x: MLXArray) -> MLXArray? {
        let width = x.dim(-1)
        let rows = x.size / width
        guard rows <= Inkling2.speculativeVerifyPositions,
            inkling2RouterKernelReads(width: width, experts: weight.dim(0)),
            Device.defaultDevice().deviceType == .gpu
        else { return nil }
        return inkling2RouterLogits(x.reshaped(rows, width).asType(.float32), weight: weight)
    }

    /// The top experts of each row of `x` and their weights.
    ///
    /// A few rows on the GPU take the router logits kernel, whose every row has the same bits
    /// however many rows the call holds, so that a verify pass routes as decode steps do.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let width = x.dim(-1)
        let rows = x.size / width
        let logits =
            rows <= Inkling2.speculativeVerifyPositions
                && inkling2RouterKernelReads(width: width, experts: weight.dim(0))
                && Device.defaultDevice().deviceType == .gpu
            ? inkling2RouterLogits(x.reshaped(rows, width).asType(.float32), weight: weight)
                .reshaped(Array(x.shape.dropLast()) + [weight.dim(0)])
            : matmul(x.asType(.float32), weight.transposed())
        if Device.defaultDevice().deviceType == .gpu {
            return inkling2ExpertRouting(
                logits: logits, bias: bias, topK: topK, scoreEps: routeWeightEpsilon,
                routeScale: routeScale)
        }
        let scores = sigmoid(logits)
        let biased = scores + bias
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + routeWeightEpsilon)
        return (indices.asType(.uint32), weights * routeScale)
    }
}

final class Inkling2SparseMoE: Module {
    @ModuleInfo(key: "gate") fileprivate var router: Inkling2Router
    @ModuleInfo var experts: Inkling2Experts

    init(_ config: Inkling2TextConfiguration) {
        _router.wrappedValue = Inkling2Router(config)
        _experts.wrappedValue = Inkling2Experts(config, gateUp: .nvfp4, down: .nvfp4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = router(x)
        return weightedExperts(experts(x, indices), weights[.ellipsis, .newAxis])
    }

    /// The layer output `residual + convolution(self(x))` of a prefill chunk and the window of
    /// the convolution. Without a mask, the expert outputs go straight into the convolution,
    /// unless the caller keeps the whole window, which needs the expert outputs themselves.
    func prefill(
        _ x: MLXArray, residual: MLXArray, convolution: Inkling2ShortConvolution,
        state: MLXArray?, mask: MLXArray?, keepsWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        guard convolution.usesKernel(mask: mask), !keepsWindow else {
            let (output, window) = convolution(
                self(x), state: state, mask: mask, keepsWindow: keepsWindow)
            return (residualAdd(residual, output), window)
        }
        let (indices, weights) = router(x)
        let (outputs, order) = experts.outputs(x, indices)
        return inkling2RowConvolution(
            outputs, state: state, weight: convolution.conv1d.weight, base: residual,
            outputType: x.dtype, experts: (weights, order))
    }

    /// ``Inkling2Experts/decode(_:indices:weights:residual:state:convolutionWeight:)``, routed
    /// from the float32 copy of the input.
    func decode(
        _ x: MLXArray, routerInput: MLXArray, residual: MLXArray, state: MLXArray,
        convolutionWeight: MLXArray
    ) -> (MLXArray, MLXArray) {
        let rows = routerInput.dim(0)
        if let logits = router.kernelLogits(routerInput),
            let output = experts.decode(
                x, logits: logits, bias: router.bias, topK: router.topK,
                scoreEps: routeWeightEpsilon, routeScale: router.routeScale, residual: residual,
                state: state, convolutionWeight: convolutionWeight)
        {
            return output
        }
        var (indices, weights) = router(routerInput)
        (indices, weights) = (indices.reshaped(rows, -1), weights.reshaped(rows, -1))
        return experts.decode(
            x, indices: indices, weights: weights, residual: residual, state: state,
            convolutionWeight: convolutionWeight)
    }
}
