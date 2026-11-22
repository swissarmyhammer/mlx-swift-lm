// Copyright © 2026 Apple Inc.
//
// The constants, fused element-wise helpers, convolutions and depth attention that the Inkling 2
// attention, mixture-of-experts and decoder layers share.

import Foundation
import MLX
import MLXNN

// MARK: - Shared constants

/// The epsilon of the RMS norms and of the L2 clips.
let normEpsilon: Float = 1e-6

/// The axis of the positions of `[B, H, L, D]` heads. It is the axis of the heads of
/// `[B, L, H, D]` rows, so swapping it with axis 1 turns rows into heads and heads into rows.
let headPositionAxis = 2

/// The axis of the rows of a batch of matrices `[..., R, C]`.
let matrixRowAxis = -2

/// The messages of each loop core layer, in the order of the message bank: the query, key and
/// value messages of the attention, then the message of the MLP.
enum LayerMessage {
    static let query = 0
    static let key = 1
    static let value = 2
    static let mlp = 3
    /// The messages of the attention.
    static let attentionCount = 3
    /// The messages of one layer.
    static let count = 4
}

/// The slots of the convolution cache of each layer visit.
enum ConvolutionSlot {
    static let key = 0
    static let value = 1
    static let attention = 2
    static let mlp = 3
    /// The convolutions of one layer visit.
    static let count = 4
}

/// The three arrays of `arrays`, in order.
func unpacked(_ arrays: [MLXArray]) -> (MLXArray, MLXArray, MLXArray) {
    var rest = arrays[...]
    let values = (rest.removeFirst(), rest.removeFirst(), rest.removeFirst())
    precondition(rest.isEmpty, "more arrays than the tuple holds")
    return values
}

/// The four arrays of `arrays`, in order.
func unpacked(_ arrays: [MLXArray]) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    var rest = arrays[...]
    let values = (rest.removeFirst(), rest.removeFirst(), rest.removeFirst(), rest.removeFirst())
    precondition(rest.isEmpty, "more arrays than the tuple holds")
    return values
}

/// The five arrays of `arrays`, in order.
func unpacked(_ arrays: [MLXArray]) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
    var rest = arrays[...]
    let values = (
        rest.removeFirst(), rest.removeFirst(), rest.removeFirst(), rest.removeFirst(),
        rest.removeFirst()
    )
    precondition(rest.isEmpty, "more arrays than the tuple holds")
    return values
}

/// The six arrays of `arrays`, in order.
func unpacked(_ arrays: [MLXArray]) -> (
    MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray
) {
    var rest = arrays[...]
    let values = (
        rest.removeFirst(), rest.removeFirst(), rest.removeFirst(), rest.removeFirst(),
        rest.removeFirst(), rest.removeFirst()
    )
    precondition(rest.isEmpty, "more arrays than the tuple holds")
    return values
}

// MARK: - Fused element-wise helpers

func unweightedRMSNorm(_ x: MLXArray, eps: Float = normEpsilon) -> MLXArray {
    MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
}

/// Hides a constant `MLXArray` from `Module` parameter reflection.
final class Inkling2ConstantBox: @unchecked Sendable {
    let value: MLXArray
    init(_ value: MLXArray) { self.value = value }
}

// MARK: - Small modules

/// The weight of a depthwise convolution, `[channels, kernel, 1]`.
final class Inkling2DepthwiseWeight: Module {
    @ParameterInfo var weight: MLXArray

    init(channels: Int, kernelSize: Int) {
        self._weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
        super.init()
    }
}

/// The inputs of a causal convolution: the previous state, then `x`, with masked rows zeroed.
private func convolutionWindow(
    _ x: MLXArray, state: MLXArray?, mask: MLXArray?, kernelSize: Int
) -> MLXArray {
    var x = x
    if let mask {
        x = which(mask[.ellipsis, .newAxis], x, MLXArray(0, dtype: x.dtype))
    }
    let state =
        state ?? MLXArray.zeros([x.dim(0), kernelSize - 1, x.dim(-1)], dtype: x.dtype)
    return concatenated([state, x], axis: 1)
}

/// The window length, in states, past which ``convolutionState(_:lengths:count:)`` copies the
/// state out of the window rather than keeping a view of it.
private let convolutionStateCopyThreshold = 2

/// The last `kernelSize - 1` rows of each row's window, at its length when the batch is ragged.
///
/// A state cut from a window more than ``convolutionStateCopyThreshold`` states long is a copy:
/// a slice of the window would keep the whole window alive, and so would `contiguous` of the
/// slice, which shares the window's buffer when the batch holds one row.
func convolutionState(_ window: MLXArray, lengths: MLXArray?, count: Int) -> MLXArray {
    let length = window.dim(1)
    if let lengths {
        let ends = clip(lengths, min: 0, max: length - count)
        let positions = (ends[0..., .newAxis] + MLXArray(0 ..< count))[.ellipsis, .newAxis]
        return takeAlong(window, positions.asType(.int32), axis: 1)
    }
    guard length > convolutionStateCopyThreshold * count else {
        return window[0..., (length - count)..., 0...]
    }
    return take(window, MLXArray(Int32(length - count) ..< Int32(length)), axis: 1)
}

/// One decode step of the static convolution, from the `[C, K, 1]` weight: the bfloat16
/// products accumulate in float32.
private func staticConvolutionStep(kernelSize: Int) -> @Sendable ([MLXArray]) -> [MLXArray] {
    let divisor = Float(kernelSize + 1).squareRoot()
    return compile { args in
        let (window, weight, x) = unpacked(args)
        let taps = weight.squeezed(axis: -1).transposed().asType(.float32)
        let out = (window.asType(.float32) * taps).sum(axis: 1, keepDims: true).asType(x.dtype)
        return [((out.asType(.float32) + x.asType(.float32)) / divisor).asType(x.dtype)]
    }
}

/// `x` as `[L, 1, D]` rows that are not contiguous with each other, so that a matmul over it
/// runs one matvec per row, each with the bits of a decode step, rather than one matmul over all
/// rows, whose sums run in another order.
///
/// - Parameter x: the rows, any shape whose last axis is the input width.
/// - Returns: a view of the rows, `[L, 1, D]`.
func rowwiseMatvecInput(_ x: MLXArray) -> MLXArray {
    let rows = x.reshaped(-1, 1, x.dim(-1))
    return concatenated([rows, rows], axis: 1)[0..., ..<1, 0...]
}

/// The convolution state before each token of a sequence of decode rows, as one decode step
/// after another would leave it, and the window of the sequence.
///
/// - Parameters:
///   - state: `[1, K - 1, C]`, the state before the first token.
///   - inputs: `[L, 1, C]`, the inputs of the convolution, one row per token.
/// - Returns: the states `[L, K - 1, C]` and the window `[1, K - 1 + L, C]`: the state, then the
///   inputs.
func chainedConvolutionStates(_ state: MLXArray, inputs: MLXArray)
    -> (states: MLXArray, window: MLXArray)
{
    let (rows, length) = (inputs.dim(0), state.dim(1))
    let window = concatenated(
        [state, inputs.reshaped(1, rows, -1).asType(state.dtype)], axis: 1)
    let channels = window.dim(-1)
    // Row i is rows i ..< i + length of the window: overlapping views, then one copy.
    let states = contiguous(
        asStrided(window, [rows, length, channels], strides: [channels, channels, 1]))
    return (states, window)
}

/// The window a chunk through ``inkling2RowConvolution`` returns: the whole window when the
/// caller keeps it, which a speculative verify pass rewinds from, else the next state the kernel
/// wrote.
///
/// - Parameters:
///   - x: the inputs of the chunk.
///   - state: the state before the chunk, or `nil` for zeros.
///   - next: the next state the kernel wrote.
///   - keepsWindow: whether the caller keeps the whole window.
private func kernelWindow(_ x: MLXArray, state: MLXArray?, next: MLXArray, keepsWindow: Bool)
    -> MLXArray
{
    guard keepsWindow else { return next }
    return convolutionWindow(x, state: state, mask: nil, kernelSize: next.dim(1) + 1)
}

/// A depthwise causal convolution with a residual, `(conv(x) + x) / sqrt(K + 1)`.
final class Inkling2ShortConvolution: Module {
    @ModuleInfo(key: "conv1d") var conv1d: Inkling2DepthwiseWeight

    let kernelSize: Int
    private let step: @Sendable ([MLXArray]) -> [MLXArray]

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self._conv1d.wrappedValue = Inkling2DepthwiseWeight(
            channels: channels, kernelSize: kernelSize)
        self.step = staticConvolutionStep(kernelSize: kernelSize)
        super.init()
    }

    /// Returns the output and the window, whose last rows are the next state.
    ///
    /// - Parameter keepsWindow: whether a chunk through the kernel returns its whole window, the
    ///   state before it and its inputs, rather than the next state alone.
    func callAsFunction(
        _ x: MLXArray, state: MLXArray?, mask: MLXArray?, keepsWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        let length = x.dim(1)
        if length > 1, usesKernel(mask: mask) {
            let (output, next) = inkling2RowConvolution(
                x, state: state, weight: conv1d.weight, outputType: x.dtype)
            return (output, kernelWindow(x, state: state, next: next, keepsWindow: keepsWindow))
        }
        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: kernelSize)
        if length == 1 {
            return (step([window, conv1d.weight, x])[0], window)
        }
        // [K, C]: tap k multiplies the k-th oldest input.
        let taps = conv1d.weight.squeezed(axis: -1).transposed().asType(.float32)
        var acc = window[0..., 0 ..< length, 0...].asType(.float32) * taps[0]
        for k in 1 ..< kernelSize {
            acc = acc + window[0..., k ..< (k + length), 0...].asType(.float32) * taps[k]
        }
        let out = acc.asType(x.dtype).asType(.float32)
        let divisor = Float(kernelSize + 1).squareRoot()
        return (((out + x.asType(.float32)) / divisor).asType(x.dtype), window)
    }

    /// `base + self(x)` in float32, `self(x)` without a base, and the window.
    ///
    /// - Parameter keepsWindow: as in ``callAsFunction(_:state:mask:keepsWindow:)``.
    func residual(
        _ x: MLXArray, base: MLXArray?, state: MLXArray?, mask: MLXArray?,
        keepsWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        if x.dim(1) > 1, usesKernel(mask: mask) {
            let (output, next) = inkling2RowConvolution(
                x, state: state, weight: conv1d.weight, base: base, outputType: .float32)
            return (output, kernelWindow(x, state: state, next: next, keepsWindow: keepsWindow))
        }
        let (out, window) = self(x, state: state, mask: mask)
        let residual = out.asType(.float32)
        return (base.map { $0.asType(.float32) + residual } ?? residual, window)
    }

    /// Whether a prefill chunk runs through ``inkling2RowConvolution``, which needs every row of
    /// the batch to be valid and returns the next state as the window.
    func usesKernel(mask: MLXArray?) -> Bool {
        kernelSize == staticConvolutionTaps && mask == nil
            && Device.defaultDevice().deviceType == .gpu
    }
}

/// A convolution whose taps per head are a static part plus a projection of the layer input,
/// `(conv(x) + x) / sqrt(2K + 1)`.
final class Inkling2DynamicShortConvolution: Module {
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProj: Linear

    let kernelSize: Int
    let heads: Int

    init(channels: Int, inputDims: Int, heads: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        self.heads = heads
        self._staticWeight.wrappedValue = MLXArray.zeros([kernelSize, channels])
        self._weightProj.wrappedValue = Linear(inputDims, heads * kernelSize, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - dynamicWeights: `weightProj` of the layer input, `[B, L, heads * K]`.
    ///   - keepsWindow: whether a chunk through the kernel returns its whole window rather than
    ///     the next state alone.
    func callAsFunction(
        _ x: MLXArray, dynamicWeights: MLXArray, state: MLXArray?, mask: MLXArray?,
        keepsWindow: Bool = false
    ) -> (MLXArray, MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(-1))
        let k = kernelSize
        if l > 1, k == staticConvolutionTaps, mask == nil, Device.defaultDevice().deviceType == .gpu
        {
            let (output, next) = inkling2RowConvolution(
                x, state: state, weight: staticWeight, outputType: x.dtype,
                dynamicTaps: dynamicWeights)
            return (output, kernelWindow(x, state: state, next: next, keepsWindow: keepsWindow))
        }
        let dynamicWeights = dynamicWeights.asType(.float32).reshaped(b, l, heads, 1, k)
        let staticWeights = staticWeight.asType(.float32).transposed().reshaped(heads, -1, k)
        let weights = staticWeights + dynamicWeights

        let window = convolutionWindow(x, state: state, mask: mask, kernelSize: k)
        // Lag 0 is the current input.
        let windows = stacked(
            (0 ..< k).map { lag in window[0..., (k - 1 - lag) ..< (k - 1 - lag + l), 0...] },
            axis: -1
        ).reshaped(b, l, heads, -1, k)
        let out = (windows.asType(.float32) * weights).sum(axis: -1).reshaped(b, l, c)
        // The static and the dynamic taps, and the residual.
        let divisor = Float(k + k + 1).squareRoot()
        return (((out + x.asType(.float32)) / divisor).asType(x.dtype), window)
    }
}

/// Softmax attention of learned queries over the depth sources: the completed blocks and
/// the partial block, all normalized.
final class Inkling2BlockAttentionResidual: Module {
    @ParameterInfo var query: MLXArray

    init(dimensions: Int, queries: Int = 1) {
        self._query.wrappedValue =
            queries == 1
            ? MLXArray.zeros([dimensions], dtype: .float32)
            : MLXArray.zeros([queries, dimensions], dtype: .float32)
        super.init()
    }

    /// float32 `[..., Q, D]` from blocks `[K, ..., D]` and the partial block `[..., D]`.
    func callAsFunction(_ blocks: MLXArray, _ partial: MLXArray) -> MLXArray {
        let d = partial.dim(-1)
        return inkling2DepthReadout(
            blocks: blocks, partial: partial, queries: query.reshaped(-1, d))
    }
}

/// The depth attention on the GPU, otherwise the reference computation.
func inkling2DepthReadout(
    blocks: MLXArray, partial: MLXArray, queries: MLXArray, queriesPerGroup: Int? = nil
) -> MLXArray {
    let scale = 1 / Float(partial.dim(-1)).squareRoot()
    if Device.defaultDevice().deviceType == .gpu {
        return inkling2DepthAttention(
            blocks: blocks, partial: partial, queries: queries, scale: scale,
            queriesPerGroup: queriesPerGroup)
    }
    let sources = movedAxis(
        concatenated([blocks, partial.expandedDimensions(axis: 0)]), source: 0,
        destination: matrixRowAxis)
    let weights = softmax(
        matmul(sources, queries.transposed()) * scale, axis: matrixRowAxis, precise: true)
    return matmul(weights.swappedAxes(-1, matrixRowAxis), sources)
}
