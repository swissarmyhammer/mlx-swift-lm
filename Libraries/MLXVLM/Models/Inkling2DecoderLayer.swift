// Copyright © 2026 Apple Inc.
//
// The decoder layer of Inkling 2: the loop message gates, the prefill pass and the decode paths.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Decoder layer

/// Merges a loop message into a sublayer input:
/// `(1 - tanh(forget(h))) * input + tanh(gain(h)) * message`, with `h` the rank-`r`
/// projection of the RMS-normalized input and message.
private final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetProj: Linear
    @ModuleInfo(key: "message_gate_proj") var gainProj: Linear

    init(dimensions: Int, rank: Int) {
        self._downProj.wrappedValue = Linear(messagePairParts * dimensions, rank, bias: false)
        self._forgetProj.wrappedValue = Linear(rank, dimensions, bias: false)
        self._gainProj.wrappedValue = Linear(rank, dimensions, bias: false)
        super.init()
    }

    /// float32 `input` and `message` to float32.
    func callAsFunction(_ input: MLXArray, _ message: MLXArray) -> MLXArray {
        inkling2MessageMerge(
            [input, message, downProj.weight, forgetProj.weight, gainProj.weight])[0]
    }
}

private let inkling2MessageMerge: @Sendable ([MLXArray]) -> [MLXArray] = compile {
    args in
    let (input, message, down, forget, gain) = unpacked(args)
    let states = concatenated(
        [unweightedRMSNorm(input), unweightedRMSNorm(message)], axis: -1)
    let hidden = matmul(states.asType(down.dtype), down.transposed()).asType(.float32)
    let forgetGate = tanh(matmul(hidden, forget.transposed().asType(.float32)))
    let messageGate = tanh(matmul(hidden, gain.transposed().asType(.float32)))
    return [(1 - forgetGate) * input + messageGate * message]
}

/// The query, key, value and MLP messages of core layer `index` of a message bank
/// `[..., 4 * layers, D]`.
func layerMessages(_ bank: MLXArray, index: Int) -> MLXArray {
    let first = index * LayerMessage.count
    return bank[.ellipsis, first ..< (first + LayerMessage.count), 0...]
}

final class Inkling2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Inkling2Attention
    @ModuleInfo var mlp: Inkling2SparseMoE
    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockAttentionResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockAttentionResidual
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConvolution:
        Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMLPConvolution: Inkling2ShortConvolution

    // Sliding layers normalize one attention input, full layers three.
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var queryInputNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var keyInputNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var valueInputNorm: RMSNorm?

    // Loop core layers merge a message into each sublayer input in later passes.
    @ModuleInfo(key: "q_message_gate") fileprivate var queryMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") fileprivate var keyMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") fileprivate var valueMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") fileprivate var mlpMessageGate: Inkling2MessageGate?

    let isSliding: Bool
    let isBlockBoundary: Bool
    let convolutionStateLength: Int

    init(_ config: Inkling2TextConfiguration, layer: Int) {
        let dims = config.hiddenSize
        let eps = config.rmsNormEps
        isSliding = config.isSliding(layer)
        isBlockBoundary = layer % config.layersPerBlock == 0
        convolutionStateLength = config.convolutionKernelSize - 1

        _attention.wrappedValue = Inkling2Attention(config, sliding: isSliding)
        _mlp.wrappedValue = Inkling2SparseMoE(config)
        if layer > 0 {
            _attentionInput.wrappedValue = Inkling2BlockAttentionResidual(
                dimensions: dims, queries: isSliding ? 1 : LayerMessage.attentionCount)
        }
        _mlpInput.wrappedValue = Inkling2BlockAttentionResidual(dimensions: dims)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        _postAttentionConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: dims, kernelSize: config.convolutionKernelSize)
        _postMLPConvolution.wrappedValue = Inkling2ShortConvolution(
            channels: dims, kernelSize: config.convolutionKernelSize)
        if isSliding {
            _inputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        } else {
            _queryInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
            _keyInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
            _valueInputNorm.wrappedValue = RMSNorm(dimensions: dims, eps: eps)
        }
        if let loop = config.loop, loop.core.contains(layer) {
            _queryMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _keyMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _valueMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
            _mlpMessageGate.wrappedValue = Inkling2MessageGate(
                dimensions: dims, rank: loop.gateRank)
        }
        super.init()
    }

    /// The normalized query, key and value inputs of the attention.
    private func attentionInputs(
        _ hidden: MLXArray, readout: MLXArray?, messages: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let dtype = hidden.dtype
        var sources: (MLXArray, MLXArray, MLXArray)
        if let readout {
            sources =
                isSliding
                ? (
                    readout[.ellipsis, 0, 0...], readout[.ellipsis, 0, 0...],
                    readout[.ellipsis, 0, 0...]
                )
                : (
                    readout[.ellipsis, LayerMessage.query, 0...],
                    readout[.ellipsis, LayerMessage.key, 0...],
                    readout[.ellipsis, LayerMessage.value, 0...]
                )
        } else {
            sources = (hidden, hidden, hidden)
        }
        if let messages, let queryMessageGate, let keyMessageGate, let valueMessageGate {
            sources = (
                queryMessageGate(sources.0, messages[.ellipsis, LayerMessage.query, 0...]),
                keyMessageGate(sources.1, messages[.ellipsis, LayerMessage.key, 0...]),
                valueMessageGate(sources.2, messages[.ellipsis, LayerMessage.value, 0...])
            )
        } else if isSliding, let inputNorm {
            let normed = inputNorm(sources.0).asType(dtype)
            return (normed, normed, normed)
        }
        if let inputNorm {
            return (
                inputNorm(sources.0).asType(dtype), inputNorm(sources.1).asType(dtype),
                inputNorm(sources.2).asType(dtype)
            )
        }
        let norms = attentionInputNorms
        return (
            norms.query(sources.0).asType(dtype), norms.key(sources.1).asType(dtype),
            norms.value(sources.2).asType(dtype)
        )
    }

    /// - Parameters:
    ///   - hidden: the partial sum of the current block.
    ///   - blocks: float32 `[K, B, L, D]`, the normalized completed blocks.
    ///   - messages: float32 `[B, L, 4 * coreLayers, D]` and the core index of this layer, whose
    ///     query, key, value and MLP messages start at `4 * index`.
    ///   - sequential: whether the rows of a `[B, 1, D]` input are the consecutive tokens of one
    ///     sequence rather than the sequences of a batch (see ``Inkling2TextModel``).
    func callAsFunction(
        _ hidden: MLXArray, blocks: MLXArray?, cache: CacheList?, convolutionMask: MLXArray?,
        messages bank: (MLXArray, Int)?, masks: Inkling2DecodeMasks, sequential: Bool = false
    ) -> (MLXArray, MLXArray) {
        if let cache, let convolutionCache = cache[1] as? ArraysCache, hidden.dim(1) == 1,
            Self.decodes(hidden, cache: cache, convolutionMask: convolutionMask)
        {
            return decode(
                hidden, blocks: blocks, cache: cache, convolutionCache: convolutionCache,
                bank: bank, sequential: sequential)
        }
        let messages = bank.map(layerMessages)
        let dtype = hidden.dtype
        let convolutionCache = cache?[1] as? ArraysCache
        let verifyCache = keepsVerifyWindows(
            convolutionCache, length: hidden.dim(1), convolutionMask: convolutionMask)
        let keepsWindows = verifyCache != nil
        let states: [MLXArray?] = (0 ..< ConvolutionSlot.count).map { convolutionCache?[$0] }
        let normedResidual = unweightedRMSNorm(hidden.asType(.float32))

        var readout: MLXArray?
        if let attentionInput, let blocks {
            readout = attentionInput(blocks, normedResidual)
        }
        let (b, l, dims) = (hidden.dim(0), hidden.dim(1), hidden.dim(-1))
        let eps = postAttentionNorm.eps
        var gates: (controllers: MessageControllers, messages: MLXArray, first: Int)?
        if let bank, Device.defaultDevice().deviceType == .gpu,
            let controllers = messageControllers(), controllers.down.dtype == dtype
        {
            gates = (controllers, bank.0.reshaped(b * l, -1, dims), bank.1 * LayerMessage.count)
        }
        let (q, k, v): (MLXArray, MLXArray, MLXArray)
        if let gates, let readout {
            let (controllers, messages, first) = gates
            let norms = attentionNormWeights
            let out = inkling2MessageGates(
                inputs: readout.reshaped(b * l, -1, dims), messages: messages, first: first,
                down: controllers.attention.down, forget: controllers.attention.forget,
                gain: controllers.attention.gain, normWeight: norms, eps: eps)
            (q, k, v) = (
                out[LayerMessage.query].reshaped(b, l, dims),
                out[LayerMessage.key].reshaped(b, l, dims),
                out[LayerMessage.value].reshaped(b, l, dims)
            )
        } else {
            (q, k, v) = attentionInputs(hidden, readout: readout, messages: messages)
        }
        let (attentionOutput, keyWindow, valueWindow) = attention(
            queryInput: q, keyInput: k, valueInput: v, cache: cache?[0], convolutions: states,
            convolutionMask: convolutionMask, masks: masks, keepsWindows: keepsWindows)
        let (residual, attentionWindow) = postAttentionConvolution.residual(
            attentionOutput, base: isBlockBoundary ? nil : hidden,
            state: states[ConvolutionSlot.attention],
            mask: convolutionMask, keepsWindow: keepsWindows)

        let blocks = completedBlocks(blocks, normedResidual: normedResidual)

        var mlpInputs = mlpInput(blocks, unweightedRMSNorm(residual))[.ellipsis, 0, 0...]
        let expertInput: MLXArray
        if let gates {
            let (controllers, messages, first) = gates
            expertInput = inkling2MessageGates(
                inputs: mlpInputs.reshaped(b * l, 1, dims), messages: messages,
                first: first + LayerMessage.mlp, down: controllers.mlp.down,
                forget: controllers.mlp.forget, gain: controllers.mlp.gain,
                normWeight: postAttentionNorm.weight.reshaped(1, dims),
                eps: eps
            ).reshaped(b, l, dims)
        } else {
            if let messages, let mlpMessageGate {
                mlpInputs = mlpMessageGate(mlpInputs, messages[.ellipsis, LayerMessage.mlp, 0...])
            }
            expertInput = postAttentionNorm(mlpInputs).asType(dtype)
        }
        let (output, mlpWindow) = mlp.prefill(
            expertInput, residual: residual,
            convolution: postMLPConvolution, state: states[ConvolutionSlot.mlp],
            mask: convolutionMask,
            keepsWindow: keepsWindows)

        if let convolutionCache {
            let lengths = convolutionCache.currentLengths
            storeWindows(
                [keyWindow, valueWindow, attentionWindow, mlpWindow], into: convolutionCache,
                advancing: hidden.dim(1), keeping: verifyCache
            ) { convolutionState($0, lengths: lengths, count: convolutionStateLength) }
        }
        return (output, blocks)
    }

    /// The completed blocks the MoE input reads: `blocks`, and at a block boundary the
    /// normalized residual of the layer as one block more.
    ///
    /// - Parameters:
    ///   - blocks: the blocks before this layer, `[K, ...]`, or `nil` before the first block.
    ///   - normedResidual: the normalized residual, the shape of one block.
    private func completedBlocks(_ blocks: MLXArray?, normedResidual: MLXArray) -> MLXArray {
        guard isBlockBoundary else {
            guard let blocks else { preconditionFailure("The first layer opens the first block") }
            return blocks
        }
        let block = normedResidual.expandedDimensions(axis: 0)
        return blocks.map { concatenated([$0, block]) } ?? block
    }

    /// The query, key and value input norms of a full-attention layer.
    private var attentionInputNorms: (query: RMSNorm, key: RMSNorm, value: RMSNorm) {
        guard let queryInputNorm, let keyInputNorm, let valueInputNorm else {
            preconditionFailure("A full-attention layer has query, key and value input norms")
        }
        return (queryInputNorm, keyInputNorm, valueInputNorm)
    }

    /// The weights of the attention input norms for the kernels: `[1, D]` for the one norm of a
    /// sliding layer, `[3, D]` for the query, key and value norms of a full-attention layer.
    private var attentionNormWeights: MLXArray {
        if let inputNorm {
            return inputNorm.weight.reshaped(1, -1)
        }
        let norms = attentionInputNorms
        return stacked([norms.query.weight, norms.key.weight, norms.value.weight])
    }

    /// Stores the state each convolution window of a forward pass ends with, moves the cache past
    /// the pass, and hands `verifyCache` the windows, so that it rewinds the pass in place.
    ///
    /// - Parameters:
    ///   - windows: the key, value, attention and MLP convolution windows, in slot order.
    ///   - cache: the convolution cache of this layer visit.
    ///   - count: the tokens of the pass.
    ///   - verifyCache: the cache that keeps the windows, or `nil`.
    ///   - state: the state a window ends with.
    private func storeWindows(
        _ windows: [MLXArray], into cache: ArraysCache, advancing count: Int,
        keeping verifyCache: SpeculativeConvolutionCache?, state: (MLXArray) -> MLXArray
    ) {
        for (slot, window) in windows.enumerated() {
            cache[slot] = state(window)
        }
        cache.advance(count)
        verifyCache?.keepVerifyWindows(windows)
    }

    /// The convolution cache that keeps the windows of this forward pass, so that a speculative
    /// verify pass rewinds in place, or `nil` when the pass is not such a pass: a chunk of more
    /// tokens than the cache rewinds, one token, or a masked or ragged batch.
    private func keepsVerifyWindows(
        _ cache: ArraysCache?, length: Int, convolutionMask: MLXArray?
    ) -> SpeculativeConvolutionCache? {
        guard let cache = cache as? SpeculativeConvolutionCache, length > 1,
            length <= cache.maximumVerifyPositions, convolutionMask == nil,
            cache.currentLengths == nil
        else { return nil }
        return cache
    }

    /// Whether the decode kernels run a layer visit of `hidden` through `cache`: bfloat16 on the
    /// GPU, unmasked, with an attention cache the kernels read.
    static func decodes(_ hidden: MLXArray, cache: CacheList, convolutionMask: MLXArray?) -> Bool {
        cache[1] is ArraysCache && hidden.dtype == .bfloat16 && convolutionMask == nil
            && Inkling2Attention.supportsDecode(cache[0])
            && Device.defaultDevice().deviceType == .gpu
    }

    /// One token per row through the decode kernels; with `sequential` the rows are the
    /// consecutive tokens of one sequence.
    private func decode(
        _ hidden: MLXArray, blocks: MLXArray?, cache: CacheList, convolutionCache: ArraysCache,
        bank: (MLXArray, Int)?, sequential: Bool
    ) -> (MLXArray, MLXArray) {
        let (rows, dims) = (hidden.dim(0), hidden.dim(-1))
        let kvDims = attention.kvHeads * attention.headDim
        let stateRows = sequential ? 1 : rows
        func state(_ slot: Int, _ channels: Int) -> MLXArray {
            convolutionCache[slot]
                ?? MLXArray.zeros(
                    [stateRows, convolutionStateLength, channels], dtype: hidden.dtype)
        }
        let x = hidden.reshaped(rows, dims)
        let eps = postAttentionNorm.eps
        let messages = bank.map { ($0.0.reshaped(rows, -1, dims), $0.1 * LayerMessage.count) }
        let controllers = messages == nil ? nil : messageControllers()
        let attentionNorms = attentionNormWeights

        // The normalized attention inputs.
        let normedResidual: MLXArray
        var inputs: [MLXArray]
        if let attentionInput, let blocks {
            let mix = inkling2DepthMix(
                blocks: blocks.reshaped(-1, rows, dims), base: x, convolution: nil,
                queries: attentionInput.query.reshaped(-1, dims),
                normWeight: controllers == nil ? attentionNorms : nil, eps: eps)
            normedResidual = mix.normed
            if let messages, let controllers {
                let (out, _) = inkling2MessageControllers(
                    inputs: mix.readout.swappedAxes(0, 1), messages: messages.0,
                    first: messages.1, down: controllers.attention.down,
                    forget: controllers.attention.forget, gain: controllers.attention.gain,
                    normWeight: attentionNorms, eps: eps, f32Copy: false)
                inputs = (0 ..< LayerMessage.attentionCount).map {
                    out[$0 ..< ($0 + 1)].reshaped(rows, 1, dims)
                }
            } else {
                let readout = mix.readout.reshaped(rows, 1, -1, dims)
                inputs = (0 ..< readout.dim(matrixRowAxis)).map { readout[0..., 0..., $0, 0...] }
            }
        } else {
            normedResidual = unweightedRMSNorm(x.asType(.float32))
            let (q, k, v) = attentionInputs(
                hidden, readout: nil,
                messages: bank.map(layerMessages))
            inputs = [q, k, v]
        }
        if inputs.count == 1 {
            inputs = [inputs[0], inputs[0], inputs[0]]
        }

        let (attentionOutput, keyState, valueState) = attention.decode(
            queryInput: inputs[LayerMessage.query], keyInput: inputs[LayerMessage.key],
            valueInput: inputs[LayerMessage.value], cache: cache[0],
            states: (state(ConvolutionSlot.key, kvDims), state(ConvolutionSlot.value, kvDims)),
            sequential: sequential)
        var attentionState = state(ConvolutionSlot.attention, dims)
        var attentionWindow: MLXArray?
        if sequential {
            (attentionState, attentionWindow) = chainedConvolutionStates(
                attentionState, inputs: attentionOutput)
        }

        // The MoE input from the blocks and the residual after the attention.
        let blocks = completedBlocks(
            blocks?.reshaped(-1, rows, dims), normedResidual: normedResidual)
        let mlpNorm = postAttentionNorm.weight.reshaped(1, dims)
        let mix = inkling2DepthMix(
            blocks: blocks, base: isBlockBoundary ? nil : x,
            convolution: (
                attentionOutput.reshaped(rows, dims), attentionState,
                postAttentionConvolution.conv1d.weight
            ),
            queries: mlpInput.query.reshaped(1, dims),
            normWeight: controllers == nil ? mlpNorm : nil, f32Copy: controllers == nil,
            eps: eps)
        var expertInput = mix.readout.reshaped(rows, dims)
        var routerFloat32 = mix.readoutF32
        if let messages, let controllers {
            let (out, f32) = inkling2MessageControllers(
                inputs: mix.readout.reshaped(1, rows, dims), messages: messages.0,
                first: messages.1 + LayerMessage.mlp, down: controllers.mlp.down,
                forget: controllers.mlp.forget, gain: controllers.mlp.gain,
                normWeight: mlpNorm, eps: eps, f32Copy: true)
            expertInput = out.reshaped(rows, dims)
            routerFloat32 = f32
        }
        guard let routerInput = routerFloat32?.reshaped(rows, dims) else {
            preconditionFailure("The MoE input has a float32 copy for the router")
        }
        let convolutionWeight = postMLPConvolution.conv1d.weight
        if let attentionWindow {
            // Each token's expert sum convolves with the sums of the tokens before it, so the
            // sums come first: the next state of every row ends with the row's own sum.
            let sums = mlp.decode(
                expertInput, routerInput: routerInput, residual: mix.residual,
                state: MLXArray.zeros([rows, convolutionStateLength, dims], dtype: hidden.dtype),
                convolutionWeight: convolutionWeight
            ).1[0..., convolutionStateLength - 1, 0...]
            let (mlpStates, mlpWindow) = chainedConvolutionStates(
                state(ConvolutionSlot.mlp, dims), inputs: sums.expandedDimensions(axis: 1))
            let output = inkling2ExpertConvolution(
                sums: sums, residual: mix.residual, state: mlpStates,
                convolutionWeight: convolutionWeight
            ).hidden
            let verifyCache = keepsVerifyWindows(
                convolutionCache, length: rows, convolutionMask: nil)
            storeWindows(
                [keyState, valueState, attentionWindow, mlpWindow], into: convolutionCache,
                advancing: rows, keeping: verifyCache
            ) { window in
                // A window the cache keeps lives on anyway, so its state stays a view of it.
                verifyCache == nil
                    ? convolutionState(window, lengths: nil, count: convolutionStateLength)
                    : window[0..., (-convolutionStateLength)..., 0...]
            }
            return (output.reshaped(rows, 1, dims), blocks.reshaped(-1, rows, 1, dims))
        }
        let (output, mlpState) = mlp.decode(
            expertInput, routerInput: routerInput, residual: mix.residual,
            state: state(ConvolutionSlot.mlp, dims),
            convolutionWeight: convolutionWeight)

        convolutionCache[ConvolutionSlot.key] = keyState
        convolutionCache[ConvolutionSlot.value] = valueState
        convolutionCache[ConvolutionSlot.attention] = mix.state
        convolutionCache[ConvolutionSlot.mlp] = mlpState
        convolutionCache.advance(1)
        return (output.reshaped(rows, 1, dims), blocks.reshaped(-1, rows, 1, dims))
    }

    // MARK: Stacked message controllers

    struct MessageControllers {
        /// bfloat16 `[4, R, 2D]`: the query, key, value and MLP controllers.
        var down: MLXArray
        /// bfloat16 `[4, R, D]`.
        var forget: MLXArray
        /// bfloat16 `[4, R, D]`.
        var gain: MLXArray

        /// The query, key and value controllers.
        var attention: MessageControllers {
            MessageControllers(
                down: down[..<LayerMessage.attentionCount],
                forget: forget[..<LayerMessage.attentionCount],
                gain: gain[..<LayerMessage.attentionCount])
        }

        /// The MLP controllers.
        var mlp: MessageControllers {
            MessageControllers(
                down: down[LayerMessage.mlp...], forget: forget[LayerMessage.mlp...],
                gain: gain[LayerMessage.mlp...])
        }
    }

    private let stackedControllers = MessageControllersBox()

    private var gates: [Inkling2MessageGate] {
        [queryMessageGate, keyMessageGate, valueMessageGate, mlpMessageGate].compactMap { $0 }
    }

    /// The controllers stacked for the decode kernels. The gates' weights become views of the
    /// stacked storage, in bfloat16, which holds the float32 checkpoint values exactly.
    func prepareMessageControllers() throws {
        guard gates.count == LayerMessage.count, stackedControllers.value == nil else { return }
        let stacked = MessageControllers(
            down: MLX.stacked(gates.map { $0.downProj.weight.asType(.bfloat16) }),
            forget: MLX.stacked(gates.map { $0.forgetProj.weight.T.asType(.bfloat16) }),
            gain: MLX.stacked(gates.map { $0.gainProj.weight.T.asType(.bfloat16) }))
        eval(stacked.down, stacked.forget, stacked.gain)
        stackedControllers.installing = true
        defer { stackedControllers.installing = false }
        for (index, gate) in gates.enumerated() {
            try gate.update(
                modules: ModuleChildren(values: [
                    "down_proj": .value(Linear(weight: stacked.down[index])),
                    "forget_gate_proj": .value(Linear(weight: stacked.forget[index].T)),
                    "message_gate_proj": .value(Linear(weight: stacked.gain[index].T)),
                ]), verify: [])
        }
        stackedControllers.value = stacked
    }

    private func messageControllers() -> MessageControllers? {
        if let value = stackedControllers.value { return value }
        guard gates.count == LayerMessage.count else { return nil }
        return MessageControllers(
            down: MLX.stacked(gates.map { $0.downProj.weight.asType(.bfloat16) }),
            forget: MLX.stacked(gates.map { $0.forgetProj.weight.T.asType(.bfloat16) }),
            gain: MLX.stacked(gates.map { $0.gainProj.weight.T.asType(.bfloat16) }))
    }

    override func update(
        parameters: ModuleParameters, verify: VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        let replacesGates = parameters.flattened().contains { key, _ in
            key.contains("_message_gate.")
        }
        defer { if replacesGates { stackedControllers.invalidate() } }
        return try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func updateModule(key: String, _ value: Any) throws {
        defer { if key.hasSuffix("_message_gate") { stackedControllers.invalidate() } }
        try super.updateModule(key: key, value)
    }
}

/// Hides the stacked controllers from `Module` reflection.
private final class MessageControllersBox {
    var value: Inkling2DecoderLayer.MessageControllers?
    var installing = false

    func invalidate() {
        if !installing { value = nil }
    }
}
