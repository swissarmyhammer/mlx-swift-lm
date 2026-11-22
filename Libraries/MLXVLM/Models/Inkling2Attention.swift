// Copyright © 2026 Apple Inc.
//
// The attention of Inkling 2: the sliding and full attention layers, their decode paths and the
// key distances their masks share.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Attention

/// The head vectors of `x * scale` with each L2 norm clipped to `threshold`, in float32 and
/// before any rounding to the input dtype.
private func clippedStates(_ x: MLXArray, scale: MLXArray, threshold: MLXArray) -> MLXArray {
    let states = x.asType(.float32) * scale
    let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
    return states * minimum(threshold / (norm + normEpsilon), 1)
}

/// Clips the L2 norm of each head vector of `x * scale` to `threshold`, in float32.
private let l2Clip: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { args in
    let (x, scale, threshold) = unpacked(args)
    return [clippedStates(x, scale: scale, threshold: threshold).asType(x.dtype)]
}

/// `l2Clip` followed by the log-length scaling of the full-attention layers. Each step
/// rounds to the input dtype, as the reference does.
private let l2ClipScaled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) {
    args in
    let (x, scale, threshold, tau) = unpacked(args)
    let clipped = clippedStates(x, scale: scale, threshold: threshold).asType(x.dtype)
    return [(clipped.asType(.float32) * tau).asType(x.dtype)]
}

/// `output * sigmoid(gate)` in float32, rounded to the gate's dtype.
private let sigmoidGate: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    output, gate in
    (output.asType(.float32) * sigmoid(gate.asType(.float32))).asType(gate.dtype)
}

/// The unweighted RMS norm of the values, in float32.
private let valueNorm: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { v in
    unweightedRMSNorm(v.asType(.float32)).asType(v.dtype)
}

/// Hides an inference-only projection from `Module` reflection.
private final class QuantizedLinearBox {
    var value: QuantizedLinear?
}

private final class Inkling2ScaledL2Clip: Module {
    @ParameterInfo var scale: MLXArray
    private let threshold: Inkling2ConstantBox

    init(heads: Int, headDim: Int, threshold: Float) {
        self._scale.wrappedValue = MLXArray.ones([heads, headDim])
        self.threshold = Inkling2ConstantBox(MLXArray(threshold))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tau: MLXArray? = nil) -> MLXArray {
        if let tau {
            return l2ClipScaled([x, scale, threshold.value, tau])[0]
        }
        return l2Clip([x, scale, threshold.value])[0]
    }
}

private final class Inkling2RelativeLogits: Module {
    @ParameterInfo var proj: MLXArray

    init(relativeDim: Int, extent: Int) {
        self._proj.wrappedValue = MLXArray.zeros([relativeDim, extent])
        super.init()
    }

    func callAsFunction(_ relative: MLXArray) -> MLXArray {
        matmul(relative, proj.asType(relative.dtype))
    }
}

final class Inkling2Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "r_proj") var relativeProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") fileprivate var relativeLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") fileprivate var queryClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") fileprivate var relativeClip: Inkling2ScaledL2Clip

    /// Exactly one of the static and the dynamic convolution pairs is set.
    @ModuleInfo(key: "k_sconv") var keyConvolution: Module
    @ModuleInfo(key: "v_sconv") var valueConvolution: Module

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let isSliding: Bool

    let logScalingFloor: Float
    let logScalingAlpha: Float
    let qkThreshold: Float
    let relativeThreshold: Float
    private let keyClipScale: Inkling2ConstantBox
    private let keyClipThreshold: Inkling2ConstantBox

    /// Inference-only projections that run several projections of one input as one matmul:
    /// q, r, gate and sink, then k and v in sliding layers. The registered projections become
    /// views of the fused storage.
    private let fusedQuery = FusedQuantizedLinearProjectionCache()
    /// The q, r, gate and sink rows of a sliding layer's fused projection, for loop passes
    /// whose keys and values have their own inputs.
    private let queryRows = QuantizedLinearBox()
    /// k_proj and v_proj of a full layer, each with the convolution taps of its input.
    private let fusedKey = FusedQuantizedLinearProjectionCache()
    private let fusedValue = FusedQuantizedLinearProjectionCache()
    /// Set while ``prepareFusedProjections()`` installs the views, which must not invalidate the
    /// projections prepared before them.
    private var installingViews = false

    private static let projectionKeys: Set<String> = [
        "q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj", "k_sconv", "v_sconv",
    ]

    init(_ config: Inkling2TextConfiguration, sliding: Bool) {
        self.isSliding = sliding
        if sliding {
            heads = config.slidingAttentionHeads
            kvHeads = config.slidingKVHeads
            headDim = config.slidingHeadDim
            window = config.slidingWindow
        } else {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            window = 0
        }
        scale = 1 / Float(headDim).squareRoot()
        logScalingFloor = Float(config.logScalingFloor)
        logScalingAlpha = config.logScalingAlpha
        qkThreshold = config.qkClipThreshold
        relativeThreshold = config.relativeClipThreshold
        let extent = sliding ? config.slidingWindow : config.relativeExtent

        let dims = config.hiddenSize
        let queryDim = heads * headDim
        let kvDim = kvHeads * headDim
        _queryProj.wrappedValue = Linear(dims, queryDim, bias: false)
        _keyProj.wrappedValue = Linear(dims, kvDim, bias: false)
        _valueProj.wrappedValue = Linear(dims, kvDim, bias: false)
        _relativeProj.wrappedValue = Linear(dims, heads * config.relativeDim, bias: false)
        _outputProj.wrappedValue = Linear(queryDim, dims, bias: false)
        _gateProj.wrappedValue = Linear(dims, queryDim, bias: false)
        _sinkProj.wrappedValue = Linear(dims, heads, bias: false)
        _relativeLogits.wrappedValue = Inkling2RelativeLogits(
            relativeDim: config.relativeDim, extent: extent)
        _queryClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, headDim: headDim, threshold: config.qkClipThreshold)
        _relativeClip.wrappedValue = Inkling2ScaledL2Clip(
            heads: heads, headDim: config.relativeDim, threshold: config.relativeClipThreshold)
        keyClipScale = Inkling2ConstantBox(MLXArray(Float(1)))
        keyClipThreshold = Inkling2ConstantBox(MLXArray(config.qkClipThreshold))

        let k = config.convolutionKernelSize
        if sliding {
            _keyConvolution.wrappedValue = Inkling2ShortConvolution(channels: kvDim, kernelSize: k)
            _valueConvolution.wrappedValue = Inkling2ShortConvolution(
                channels: kvDim, kernelSize: k)
        } else {
            _keyConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvDim, inputDims: dims, heads: kvHeads, kernelSize: k)
            _valueConvolution.wrappedValue = Inkling2DynamicShortConvolution(
                channels: kvDim, inputDims: dims, heads: kvHeads, kernelSize: k)
        }
        super.init()
    }

    private var dynamicKeyConvolution: Inkling2DynamicShortConvolution? {
        keyConvolution as? Inkling2DynamicShortConvolution
    }

    private var dynamicValueConvolution: Inkling2DynamicShortConvolution? {
        valueConvolution as? Inkling2DynamicShortConvolution
    }

    // MARK: Fused projections

    /// The parameter names of the projections a fused query projection holds, in its order.
    private static let fusedProjectionNames = [
        "q_proj", "r_proj", "gate_proj", "sink_proj", "k_proj", "v_proj",
    ]

    /// The projections of `views`, the views of a fused projection, under their names.
    private static func fusedViews(_ views: [Linear]) -> ModuleChildren {
        ModuleChildren(
            values: Dictionary(
                uniqueKeysWithValues: zip(fusedProjectionNames, views.map { .value($0) })))
    }

    func prepareFusedProjections() throws {
        installingViews = true
        defer { installingViews = false }
        if isSliding {
            let prepared = try fusedQuery.prepare(
                enabled: true,
                linears: [queryProj, relativeProj, gateProj, sinkProj, keyProj, valueProj]
            ) { views in
                try update(modules: Self.fusedViews(views), verify: [])
            }
            if prepared, let fused = fusedQuery.fused {
                let rows =
                    ..<(fused.weight.dim(0) - keyProj.weight.dim(0) - valueProj.weight.dim(0))
                queryRows.value = QuantizedLinear(
                    weight: fused.weight[rows], bias: nil, scales: fused.scales[rows],
                    biases: fused.biases?[rows], groupSize: fused.groupSize, bits: fused.bits,
                    mode: fused.mode)
            }
            return
        }
        try fusedQuery.prepare(
            enabled: true, linears: [queryProj, relativeProj, gateProj, sinkProj]
        ) { views in
            try update(modules: Self.fusedViews(views), verify: [])
        }
        try prepareConvolutionProjection(
            dynamicKeyConvolution, linear: keyProj, cache: fusedKey, key: "k_proj")
        try prepareConvolutionProjection(
            dynamicValueConvolution, linear: valueProj, cache: fusedValue, key: "v_proj")
    }

    /// Fuses `linear` with the weight projection of `convolution` into `cache`, and installs the
    /// views of the fused weight as `key` and as the convolution's `weight_proj`.
    private func prepareConvolutionProjection(
        _ convolution: Inkling2DynamicShortConvolution?, linear: Linear,
        cache: FusedQuantizedLinearProjectionCache, key: String
    ) throws {
        guard let convolution else { return }
        try cache.prepare(enabled: true, linears: [linear, convolution.weightProj]) { views in
            try update(modules: ModuleChildren(values: [key: .value(views[0])]), verify: [])
            try convolution.update(
                modules: ModuleChildren(values: ["weight_proj": .value(views[1])]), verify: [])
        }
    }

    private func invalidateFusedProjections() {
        guard !installingViews else { return }
        fusedQuery.invalidate()
        fusedKey.invalidate()
        fusedValue.invalidate()
        queryRows.value = nil
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let replacesProjection = parameters.flattened().contains { key, _ in
            Self.projectionKeys.contains(String(key.prefix { $0 != "." }))
        }
        defer { if replacesProjection { invalidateFusedProjections() } }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { if Self.projectionKeys.contains(key) { invalidateFusedProjections() } }
        try super.updateModule(key: key, value)
    }

    private struct Projections {
        var queries: MLXArray
        var relative: MLXArray
        var gate: MLXArray
        var sinks: MLXArray
        var keys: MLXArray
        var values: MLXArray
        var keyTaps: MLXArray?
        var valueTaps: MLXArray?
    }

    /// The projections of the attention inputs, fused into as few matmuls as the inputs allow.
    ///
    /// - Parameters:
    ///   - queryInput: the input of the query, relative, gate and sink projections.
    ///   - keyInput: the input of the key projection.
    ///   - valueInput: the input of the value projection.
    ///   - apply: runs one projection over its input.
    /// - Returns: the projections.
    private func project(
        queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray,
        apply: (Linear, MLXArray) -> MLXArray = { $0($1) }
    ) -> Projections {
        let queryDim = heads * headDim
        let relativeDim = relativeProj.weight.dim(0)
        let kvDim = kvHeads * headDim
        func split(_ x: MLXArray, _ sizes: [Int]) -> [MLXArray] {
            var start = 0
            return sizes.map { size in
                defer { start += size }
                return x[.ellipsis, start ..< (start + size)]
            }
        }
        let sharedInput = queryInput === keyInput && keyInput === valueInput

        if isSliding, sharedInput, let fused = fusedQuery.fused {
            let (queries, relative, gate, sinks, keys, values) = unpacked(
                split(
                    apply(fused, queryInput),
                    [queryDim, relativeDim, queryDim, heads, kvDim, kvDim]))
            return Projections(
                queries: queries, relative: relative, gate: gate, sinks: sinks, keys: keys,
                values: values)
        }
        let queries: MLXArray
        let relative: MLXArray
        let gate: MLXArray
        let sinks: MLXArray
        if let fused = isSliding ? queryRows.value : fusedQuery.fused {
            (queries, relative, gate, sinks) = unpacked(
                split(apply(fused, queryInput), [queryDim, relativeDim, queryDim, heads]))
        } else {
            (queries, relative, gate, sinks) = (
                apply(queryProj, queryInput), apply(relativeProj, queryInput),
                apply(gateProj, queryInput), apply(sinkProj, queryInput)
            )
        }

        /// The projection of `input` by `linear`, and the taps of `convolution` when it is
        /// dynamic: one matmul when `cache` holds the two fused.
        func convolvedProjection(
            _ convolution: Inkling2DynamicShortConvolution?, linear: Linear,
            cache: FusedQuantizedLinearProjectionCache, input: MLXArray
        ) -> (MLXArray, MLXArray?) {
            guard let convolution else { return (apply(linear, input), nil) }
            guard let fused = cache.fused else {
                return (apply(linear, input), apply(convolution.weightProj, input))
            }
            let parts = split(apply(fused, input), [kvDim, convolution.weightProj.weight.dim(0)])
            return (parts[0], parts[1])
        }
        let (keys, keyTaps) = convolvedProjection(
            dynamicKeyConvolution, linear: keyProj, cache: fusedKey, input: keyInput)
        let (values, valueTaps) = convolvedProjection(
            dynamicValueConvolution, linear: valueProj, cache: fusedValue, input: valueInput)
        return Projections(
            queries: queries, relative: relative, gate: gate, sinks: sinks, keys: keys,
            values: values, keyTaps: keyTaps, valueTaps: valueTaps)
    }

    private func convolve(
        _ convolution: Module, _ x: MLXArray, taps: MLXArray?, state: MLXArray?, mask: MLXArray?,
        keepsWindow: Bool
    ) -> (MLXArray, MLXArray) {
        if let convolution = convolution as? Inkling2ShortConvolution {
            return convolution(x, state: state, mask: mask, keepsWindow: keepsWindow)
        }
        guard let convolution = convolution as? Inkling2DynamicShortConvolution, let taps else {
            preconditionFailure("A key or value convolution is static, or dynamic with taps")
        }
        return convolution(
            x, dynamicWeights: taps, state: state, mask: mask, keepsWindow: keepsWindow)
    }

    /// - Parameters:
    ///   - queryInput: the normalized input of the queries, relative states, gate and sinks.
    ///   - convolutions: the key and value convolution states, updated in place.
    ///   - keepsWindows: whether the key and value convolutions return their whole windows.
    func callAsFunction(
        queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray,
        cache: KVCache?, convolutions: [MLXArray?], convolutionMask: MLXArray?,
        masks: Inkling2DecodeMasks, keepsWindows: Bool = false
    ) -> (MLXArray, keyWindow: MLXArray, valueWindow: MLXArray) {
        let (b, l) = (queryInput.dim(0), queryInput.dim(1))
        let offset = cache?.offset ?? 0

        let projections = project(
            queryInput: queryInput, keyInput: keyInput, valueInput: valueInput)
        let (keys, keyWindow) = convolve(
            keyConvolution, projections.keys, taps: projections.keyTaps, state: convolutions[0],
            mask: convolutionMask, keepsWindow: keepsWindows)
        let (values, valueWindow) = convolve(
            valueConvolution, projections.values, taps: projections.valueTaps,
            state: convolutions[1], mask: convolutionMask, keepsWindow: keepsWindows)

        var tau: MLXArray?
        if !isSliding {
            let positions = MLXArray(Int32(offset) ..< Int32(offset + l)).asType(.float32)
            let t = 1 + logScalingAlpha * log(maximum((positions + 1) / logScalingFloor, 1))
            tau = t.reshaped(1, l, 1, 1)
        }

        var queries: MLXArray
        var k: MLXArray
        var v: MLXArray
        let relative: MLXArray
        // The prologue kernel gives each lane of a simdgroup a slice of a head and one relative
        // dimension.
        if Device.defaultDevice().deviceType == .gpu, headDim.isMultiple(of: metalSimdWidth),
            relativeLogits.proj.dim(0) <= metalSimdWidth
        {
            let out = inkling2PrefillPrologue(
                q: projections.queries, r: projections.relative, k: keys, v: values,
                qScale: queryClip.scale, rScale: relativeClip.scale, tau: tau?.reshaped(-1),
                heads: heads, kvHeads: kvHeads, headDim: headDim,
                relativeDim: relativeLogits.proj.dim(0), qkThreshold: qkThreshold,
                relativeThreshold: relativeThreshold)
            (queries, relative, k, v) = unpacked(out)
        } else {
            queries = queryClip(projections.queries.reshaped(b, l, heads, headDim), tau: tau)
            k =
                l2Clip([
                    keys.reshaped(b, l, kvHeads, headDim), keyClipScale.value,
                    keyClipThreshold.value,
                ])[0]
            v = valueNorm(values.reshaped(b, l, kvHeads, headDim))
            relative = relativeClip(projections.relative.reshaped(b, l, heads, -1), tau: tau)
        }
        let positionBias = relativeLogits(relative)
        let gate = projections.gate
        let sinks = projections.sinks

        queries = queries.swappedAxes(1, headPositionAxis)
        k = k.swappedAxes(1, headPositionAxis)
        v = v.swappedAxes(1, headPositionAxis)
        let output: MLXArray
        if let quantized = cache as? QuantizedKVCacheProtocol {
            let (quantizedKeys, quantizedValues) = quantized.updateQuantized(keys: k, values: v)
            let s = quantizedKeys.0.dim(headPositionAxis)
            // The query groups of each key head share its keys and values.
            func grouped(_ x: MLXArray) -> MLXArray {
                x.expandedDimensions(axis: headPositionAxis)
            }
            output = referenceAttention(
                queries: queries, positionBias: positionBias, sinks: sinks,
                distance: masks.distance(keys: s, queries: l, ringIndex: nil, keep: 0, evicted: 0),
                scores: { q in
                    quantizedMM(
                        q, grouped(quantizedKeys.0), scales: grouped(quantizedKeys.1),
                        biases: quantizedKeys.2.map(grouped), transpose: true,
                        groupSize: quantized.groupSize, bits: quantized.bits,
                        mode: quantized.mode
                    ).asType(.float32)
                },
                values: { weights in
                    quantizedMM(
                        weights, grouped(quantizedValues.0), scales: grouped(quantizedValues.1),
                        biases: quantizedValues.2.map(grouped), transpose: false,
                        groupSize: quantized.groupSize, bits: quantized.bits,
                        mode: quantized.mode)
                })
        } else {
            precondition(
                !(cache is KVCacheAttentionProtocol),
                "Inkling 2 attention needs full-precision or affine-quantized KV caches")
            if let cache {
                (k, v) = cache.update(keys: k, values: v)
            }
            output = attend(
                queries: queries, keys: k, values: v, positionBias: positionBias, sinks: sinks,
                cache: cache, end: offset + l, masks: masks)
        }
        let gated = sigmoidGate(output.reshaped(b, l, -1), gate)
        return (outputProj(gated), keyWindow, valueWindow)
    }

    /// Whether ``decode(queryInput:keyInput:valueInput:cache:states:)`` handles this cache.
    static func supportsDecode(_ cache: KVCache) -> Bool {
        !(cache is QuantizedKVCacheProtocol) && !(cache is KVCacheAttentionProtocol)
    }

    /// One token per row through the decode kernels.
    ///
    /// The rows are the sequences of a batch, or with `sequential` the consecutive tokens of one
    /// sequence: each token then convolves with the inputs of the tokens before it and attends
    /// their keys, as one decode step after another would compute it.
    ///
    /// - Parameters:
    ///   - states: the key and value convolution states, `[B, 3, kvHeads * headDim]`, or
    ///     `[1, 3, kvHeads * headDim]` with `sequential`.
    ///   - sequential: whether the rows are the consecutive tokens of one sequence.
    /// - Returns: the output `[B, 1, D]` and the next key and value convolution states, or with
    ///   `sequential` the key and value convolution windows `[1, 3 + B, kvHeads * headDim]`.
    func decode(
        queryInput: MLXArray, keyInput: MLXArray, valueInput: MLXArray, cache: KVCache,
        states: (MLXArray, MLXArray), sequential: Bool = false
    ) -> (MLXArray, MLXArray, MLXArray) {
        let rows = queryInput.dim(0)
        let offset = cache.offset
        // The rows of one sequence take the bits of decode steps.
        let p =
            sequential
            ? project(
                queryInput: queryInput, keyInput: keyInput, valueInput: valueInput,
                apply: { rowwiseProjection($1, by: $0) })
            : project(queryInput: queryInput, keyInput: keyInput, valueInput: valueInput)
        var states = states
        var windows: (MLXArray, MLXArray)?
        if sequential {
            let keys = chainedConvolutionStates(states.0, inputs: p.keys)
            let values = chainedConvolutionStates(states.1, inputs: p.values)
            states = (keys.states, values.states)
            windows = (keys.window, values.window)
        }

        var taps: (k: MLXArray, v: MLXArray)?
        let weights: (k: MLXArray, v: MLXArray)
        var tau: MLXArray?
        if let keyConvolution = dynamicKeyConvolution,
            let valueConvolution = dynamicValueConvolution
        {
            guard let keyTaps = p.keyTaps, let valueTaps = p.valueTaps else {
                preconditionFailure("The projection of a dynamic convolution returns its taps")
            }
            taps = (keyTaps, valueTaps)
            weights = (keyConvolution.staticWeight, valueConvolution.staticWeight)
            // One scaling per token: a sequence's tokens stand at consecutive positions.
            let positions = sequential ? rows : 1
            tau = MLXArray(
                (1 ... positions).map { step in
                    1 + logScalingAlpha * log(max(Float(offset + step) / logScalingFloor, 1))
                })
        } else {
            guard let keyConvolution = keyConvolution as? Inkling2ShortConvolution,
                let valueConvolution = valueConvolution as? Inkling2ShortConvolution
            else { preconditionFailure("Without dynamic convolutions both are static") }
            weights = (keyConvolution.conv1d.weight, valueConvolution.conv1d.weight)
        }
        let out = inkling2AttentionPrologue(
            q: p.queries, r: p.relative, k: p.keys, v: p.values, taps: taps, weights: weights,
            states: states, qScale: queryClip.scale, rScale: relativeClip.scale, tau: tau,
            heads: heads, kvHeads: kvHeads, headDim: headDim,
            relativeDim: relativeLogits.proj.dim(0), qkThreshold: qkThreshold,
            relativeThreshold: relativeThreshold)
        let rotating = cache as? RotatingKVCache
        // The tokens of one sequence write their keys in order and attend causally.
        let (newKeys, newValues) =
            windows == nil
            ? (out[Inkling2PrologueOutput.keys], out[Inkling2PrologueOutput.values])
            : (
                // The `[rows, Hkv, 1, DH]` keys of the rows as the `[1, Hkv, rows, DH]` keys of
                // one sequence.
                out[Inkling2PrologueOutput.keys].swappedAxes(0, headPositionAxis),
                out[Inkling2PrologueOutput.values].swappedAxes(0, headPositionAxis)
            )
        let (keys, values) = cache.update(keys: newKeys, values: newValues)
        let written = windows == nil ? 1 : rows
        let attention = inkling2DecodeAttention(
            q: out[Inkling2PrologueOutput.queries], k: keys, v: values,
            relative: out[Inkling2PrologueOutput.relative], table: relativeLogits.proj,
            sinks: p.sinks,
            gate: p.gate, scale: scale, ringIndex: windows == nil ? rotating?.ringWriteIndex : nil,
            window: window, keep: rotating?.preservedPrefixTokens ?? 0,
            evicted: offset + written - keys.dim(headPositionAxis),
            causalRows: windows == nil ? 0 : rows)
        let attended = attention.reshaped(rows, 1, -1)
        if let windows {
            return (rowwiseProjection(attended, by: outputProj), windows.0, windows.1)
        }
        return (
            outputProj(attended), out[Inkling2PrologueOutput.nextKeyState],
            out[Inkling2PrologueOutput.nextValueState]
        )
    }

    /// Attention with a relative position bias and a sink, `[B, L, H, D]`.
    private func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, positionBias: MLXArray,
        sinks: MLXArray, cache: KVCache?, end: Int, masks: Inkling2DecodeMasks
    ) -> MLXArray {
        let (b, l) = (queries.dim(0), queries.dim(headPositionAxis))
        let s = keys.dim(headPositionAxis)
        let rotating = cache as? RotatingKVCache
        let keep = rotating?.preservedPrefixTokens ?? 0
        let onGPU = Device.defaultDevice().deviceType == .gpu
        let bf16 = queries.dtype == .bfloat16 && keys.dtype == .bfloat16
        if l > 1 && onGPU && bf16 {
            return inkling2RelativeAttention(
                q: queries, k: keys, v: values, rel: positionBias, sinks: sinks, scale: scale,
                window: window, keyStart: MLXArray.full([b], values: MLXArray(Int32(s - end))),
                keep: keep, evicted: end - s)
        }

        let distance = masks.distance(
            keys: s, queries: l, ringIndex: rotating?.ringWriteIndex, keep: keep, evicted: end - s)
        return referenceAttention(
            queries: queries, positionBias: positionBias, sinks: sinks, distance: distance,
            scores: { q in
                matmul(
                    q.asType(.float32),
                    keys.asType(.float32).expandedDimensions(axis: headPositionAxis)
                        .swappedAxes(-1, matrixRowAxis))
            },
            values: { weights in
                matmul(weights, values.expandedDimensions(axis: headPositionAxis))
            })
    }

    /// The materialized attention, for prefill off the GPU, batched decode and quantized caches.
    ///
    /// - Parameters:
    ///   - scores: float32 logits `[B, Hkv, G, L, S]` of the grouped queries `[B, Hkv, G, L, D]`.
    ///   - values: the attention weights `[B, Hkv, G, L, S]` applied to the values.
    private func referenceAttention(
        queries: MLXArray, positionBias: MLXArray, sinks: MLXArray, distance: MLXArray,
        scores: (MLXArray) -> MLXArray, values: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let (b, l, d) = (queries.dim(0), queries.dim(headPositionAxis), queries.dim(-1))
        let s = distance.dim(-1)
        let extent = positionBias.dim(-1)
        var visible = distance .>= 0
        if window > 0 {
            visible = logicalAnd(visible, distance .< window)
        }
        // [B, H, L, S] gather of the bias at each distance.
        let index = broadcast(
            clip(distance, min: 0, max: extent - 1), to: [b, heads, l, s])
        var bias = takeAlong(positionBias.swappedAxes(1, headPositionAxis), index, axis: -1)
        bias = which(distance .< extent, bias.asType(.float32), MLXArray(Float(0)))
        let mask = which(visible, bias, MLXArray(-Float.infinity)).reshaped(
            b, kvHeads, -1, l, s)

        let logits = scores(queries.reshaped(b, kvHeads, -1, l, d)) * scale + mask
        let sinkLogits = sinks.asType(.float32).swappedAxes(1, -1)
            .reshaped(b, kvHeads, -1, l, 1)
        let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[
            .ellipsis, ..<s]
        let output = values(weights.asType(queries.dtype))
        return output.reshaped(b, heads, l, d).swappedAxes(1, headPositionAxis)
    }
}

/// The key distances of one forward pass, shared by the layers whose caches agree.
final class Inkling2DecodeMasks {
    private var distances: [String: MLXArray] = [:]

    /// int32 `[L, S]` (`[S]` for one query): each query's distance to each key.
    ///
    /// The first `keep` keys are pinned and `evicted` positions older than their slots.
    func distance(keys: Int, queries: Int, ringIndex: Int?, keep: Int, evicted: Int) -> MLXArray {
        let key = "\(keys):\(queries):\(ringIndex ?? -1):\(keep):\(evicted)"
        if let cached = distances[key] { return cached }
        let k = MLXArray.arange(keys, dtype: .int32)
        let slots: MLXArray
        if queries == 1 {
            slots = MLXArray(Int32(keys - 1)) - k
        } else {
            let q = MLXArray.arange(queries, dtype: .int32) + Int32(keys - queries)
            slots = q[0..., .newAxis] - k[.newAxis, 0...]
        }
        var result = slots
        if let ringIndex {
            // A decode step wrote the newest key at ringIndex - 1 of the ring after the pinned keys.
            let ring = keys - keep
            result = (MLXArray(Int32(ringIndex - 1 + ring)) - k) % Int32(ring)
        }
        if keep > 0 {
            result = which(k .< Int32(keep), slots + Int32(evicted), result)
        }
        distances[key] = result
        return result
    }
}
