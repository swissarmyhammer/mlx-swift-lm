// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLLM/Models/DeepseekV4.swift @ b166896353b9c95d773de993990c20a0b5ba6905
// Manual transcription; no git ancestry.
//
// Six details do not come from that file.
//
//  1. **The name of each module.** The file above keys a decoder layer
//     `self_attn`, `mlp`, `input_layernorm` and `post_attention_layernorm`,
//     which are the DeepSeek-V3 names, and its `sanitize` maps the DeepSeek-V4
//     checkpoint onto them. The `quantization` block of the published
//     DeepSeek-V4-Flash-4bit checkpoint names `model.layers.N.attn.*` and
//     `model.layers.N.ffn.*` -- see the copy of that block in
//     Tests/MLXLMTests/Resources/DeepSeek-V4-Flash-4bit-config.json -- and
//     `quantize(model:filter:)` hands its filter the FLATTENED MODULE PATH,
//     thus a tree under the DeepSeek-V3 names resolves no per-layer entry of
//     that plan and quantizes every routed expert with the affine default
//     instead of mxfp4. This file therefore carries the names the DeepSeek-V4
//     Python reference gives -- Thump604/mlx-lm @ deepseek-v4-support-fixes,
//     mlx_lm/models/deepseek_v4.py, `DeepseekV4Block` and `DeepseekV4Model` --
//     which are the names the quantization block states: `attn`, `ffn`,
//     `attn_norm`, `ffn_norm`, `hc_attn`, `hc_ffn` and `hc_head`.
//  2. **The multi-token-prediction head is dropped.** The file above keeps
//     `mtp.*` and builds a DSpark drafter from it. The Python reference drops
//     every `mtp.` key, and this repository carries no DSpark type, thus the
//     load filter drops them here.
//  3. **The compressor and the indexer both load, and neither runs yet.** The
//     file above keeps `attn.compressor.*` and `attn.indexer.*` and wires
//     them into its attention. ``DeepSeekV4Attention`` holds the indexer and
//     the compressor, thus every one of those tensors loads, and the
//     attention path reads neither: sparse attention needs the pooled cache
//     that Libraries/MLXLLM/Models/DeepSeekV4Compressor.swift records. No
//     sparse-attention key is dropped.
//  4. **The language-model head is optional.** The file above declares a
//     non-optional `lm_head`. `MLXLMCommon.loadWeights` verifies with
//     `.allModelKeysSet`, thus a checkpoint whose `tie_word_embeddings` is
//     true -- which ships no head tensor at all -- cannot load into such a
//     tree. This file follows the pattern the repository already states in
//     Libraries/MLXLMCommon/Documentation.docc/porting.md and applies in
//     GLM4.swift: an optional head, built only for an untied checkpoint, and
//     the embedding table read as a linear projection otherwise.
//  5. **The numeric tracer reads `MLX_DSV4_NUMERIC_TRACE`.** The file above
//     reads `VMLX_DSV4_NUMERIC_TRACE`. Every environment variable of this
//     repository starts with `MLX_`, thus the name follows that convention.
//  6. **No compiled decode path and no stage profile.** The file above carries
//     `compile(shapeless:)` around the collapse and around the layer tail, and
//     a stage profiler behind `VMLX_DSV4_STAGE_PROFILE`. Both are performance
//     work of their own and neither changes a number.
//
// Two more points, which the card that ordered this file states differently
// from the references it names:
//
//  - **Every layer holds a mixture of experts.** There is no dense prefix.
//    `DeepseekV4Block.__init__` of the Python reference builds a
//    `DeepseekV4MoE` for every layer, and DeepSeek-V4 `config.json` carries no
//    `first_k_dense_replace` key for a dense prefix to read.
//  - **The cache is the one `KVCacheDimensionProvider` gives.** The file above
//    allocates a `RotatingKVCache` for a layer whose compress ratio is 0 and a
//    compressing cache for every other layer. A rotating window is correct
//    only where the attention path reads the pooled chunks of the compressor
//    beside it, and this attention path reads no pooled chunk yet, thus a
//    window here would silently lose context. A plain cache keeps every key
//    until sparse attention lands.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Numeric tracer

/// Writes one line for each stage of a DeepSeek-V4 forward pass, so that a
/// numeric mismatch against the Python reference can be bisected without a
/// debugger.
///
/// The trace is off unless the environment names
/// ``DeepSeekV4NumericTrace/environmentName``, and each function returns at
/// once when it is off, thus a production run pays one Boolean test for each
/// stage.
enum DeepSeekV4NumericTrace {

    /// The environment variable that turns the trace on.
    static let environmentName = "MLX_DSV4_NUMERIC_TRACE"

    /// The value of ``environmentName`` that turns the trace on.
    private static let onValue = "1"

    /// The number of leading values each line samples.
    private static let sampleCount = 4

    /// True when the environment turns the trace on.
    static let isEnabled = ProcessInfo.processInfo.environment[environmentName] == onValue

    /// Writes the token identifiers one forward pass reads.
    ///
    /// - Parameter inputs: The token identifiers, of any shape.
    static func tokens(_ inputs: MLXArray) {
        guard isEnabled else { return }
        let values = inputs.asType(.int32).reshaped(-1).asArray(Int32.self)
        write("[DSV4Numeric] tokens=\(values)\n")
    }

    /// Writes the shape, the mean, the root mean square, the largest absolute
    /// value and the leading values of one stage.
    ///
    /// - Parameters:
    ///   - label: The name of the stage.
    ///   - value: The tensor the stage answered.
    static func tensor(_ label: String, _ value: MLXArray) {
        guard isEnabled else { return }
        let flat = value.asType(.float32).reshaped(-1)
        MLX.eval(flat)
        let taken = min(sampleCount, flat.size)
        let sample = taken > 0 ? flat[0 ..< taken].asArray(Float.self) : []
        write(
            String(
                format: "[DSV4Numeric] %@ shape=%@ mean=%.9g rms=%.9g max=%.9g sample=%@\n",
                label, String(describing: value.shape),
                flat.mean().item(Float.self),
                sqrt((flat * flat).mean().item(Float.self)),
                abs(flat).max().item(Float.self),
                String(describing: sample)))
    }

    /// Writes one line to the standard error stream.
    private static func write(_ line: String) {
        FileHandle.standardError.write(Data(line.utf8))
    }
}

// MARK: - Decoder layer

/// The axes of the `(batch, tokens, copies, width)` residual stream DeepSeek-V4
/// carries between its blocks.
private enum ResidualStreamAxis {
    /// The axis that holds the parallel copies of the residual stream. The
    /// embedding gains this axis, and the hyper head takes it away again.
    static let copy = 2
}

/// One DeepSeek-V4 decoder layer.
///
/// The layer holds two halves, and each half wraps its block in a manifold
/// hyper-connection: the connection reads the parallel copies of the residual
/// stream down to one stream, the norm and the block read that stream, and the
/// connection writes the block output back into the copies.
///
/// The attention half runs first and the mixture-of-experts half reads the
/// stream it wrote. That order is the order of the Python reference,
/// `DeepseekV4Block.__call__`.
///
/// Every DeepSeek-V4 layer holds a mixture of experts. There is no dense
/// prefix, and the configuration names no layer count for one.
final class DeepSeekV4DecoderLayer: Module {

    /// The attention half of this layer.
    @ModuleInfo(key: "attn") var attention: DeepSeekV4Attention

    /// The mixture-of-experts half of this layer.
    @ModuleInfo(key: "ffn") var mixtureOfExperts: DeepSeekV4MoE

    /// The norm the attention half applies to the collapsed stream.
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm

    /// The norm the mixture-of-experts half applies to the collapsed stream.
    @ModuleInfo(key: "ffn_norm") var mixtureOfExpertsNorm: RMSNorm

    /// The manifold hyper-connection the attention half runs inside.
    @ModuleInfo(key: "hc_attn") var attentionConnection: DeepSeekV4HyperConnection

    /// The manifold hyper-connection the mixture-of-experts half runs inside.
    @ModuleInfo(key: "hc_ffn") var mixtureOfExpertsConnection: DeepSeekV4HyperConnection

    /// The index of this layer, which the trace lines name.
    private let layerIndex: Int

    /// Builds one decoder layer.
    ///
    /// - Parameters:
    ///   - configuration: The configuration of the checkpoint.
    ///   - layer: The index of this decoder layer.
    init(configuration: DeepSeekV4Configuration, layer: Int) {
        self.layerIndex = layer
        self._attention.wrappedValue = DeepSeekV4Attention(
            configuration: configuration, layer: layer)
        self._mixtureOfExperts.wrappedValue = DeepSeekV4MoE(
            configuration: configuration, layer: layer)
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        self._mixtureOfExpertsNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        self._attentionConnection.wrappedValue = DeepSeekV4HyperConnection(
            configuration: configuration)
        self._mixtureOfExpertsConnection.wrappedValue = DeepSeekV4HyperConnection(
            configuration: configuration)
    }

    /// Reads one block of tokens.
    ///
    /// - Parameters:
    ///   - stream: The residual stream, shape
    ///     `(batch, tokens, hcMult, hiddenSize)`.
    ///   - mask: The attention mask of this block.
    ///   - cache: The key/value cache of this layer, or `nil` for a run that
    ///     keeps no history.
    ///   - inputIds: The token identifiers, shape `(batch, tokens)`. A hash
    ///     layer routes its tokens by them.
    /// - Returns: The residual stream of the next layer, of the shape of
    ///   `stream`.
    func callAsFunction(
        _ stream: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        inputIds: MLXArray
    ) -> MLXArray {
        let afterAttention = wrapped(
            stream, connection: attentionConnection, norm: attentionNorm, half: "attn"
        ) { [attention] normalized in
            attention(normalized, mask: mask, cache: cache)
        }
        return wrapped(
            afterAttention, connection: mixtureOfExpertsConnection, norm: mixtureOfExpertsNorm,
            half: "ffn"
        ) { [mixtureOfExperts] normalized in
            mixtureOfExperts(normalized, inputIds: inputIds)
        }
    }

    /// Runs one half of the layer inside its manifold hyper-connection.
    ///
    /// - Parameters:
    ///   - stream: The residual stream this half reads.
    ///   - connection: The hyper-connection of this half.
    ///   - norm: The norm this half applies to the collapsed stream.
    ///   - half: The name of this half, which the trace lines carry.
    ///   - block: The block this half runs on the normalized stream.
    /// - Returns: The residual stream this half wrote, of the shape of
    ///   `stream`.
    private func wrapped(
        _ stream: MLXArray,
        connection: DeepSeekV4HyperConnection,
        norm: RMSNorm,
        half: String,
        block: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let label = "layer.\(layerIndex).\(half)"
        let (collapsed, post, comb) = connection.collapse(stream)
        DeepSeekV4NumericTrace.tensor("\(label).collapse", collapsed)
        let output = block(norm(collapsed))
        DeepSeekV4NumericTrace.tensor("\(label).block", output)
        let expanded = connection.expand(
            blockOutput: output, residual: stream, post: post, comb: comb)
        DeepSeekV4NumericTrace.tensor("\(label).expand", expanded)
        return expanded
    }
}

// MARK: - Decoder stack

/// The embedding table, the decoder layers, and the reduction and the norm at
/// the top of a DeepSeek-V4 stack.
public final class DeepSeekV4ModelInner: Module {

    /// The embedding table of the vocabulary.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// The head that reads the parallel copies of the residual stream down to
    /// one stream.
    @ModuleInfo(key: "hc_head") var hcHead: DeepSeekV4HyperHead

    /// The norm at the top of the stack.
    @ModuleInfo(key: "norm") var norm: RMSNorm

    /// The decoder layers, in order.
    let layers: [DeepSeekV4DecoderLayer]

    /// The number of parallel copies of the residual stream.
    private let hcMult: Int

    /// Builds the decoder stack of one checkpoint.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    init(configuration: DeepSeekV4Configuration) {
        self.hcMult = configuration.hcMult
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize, dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.numHiddenLayers).map {
            DeepSeekV4DecoderLayer(configuration: configuration, layer: $0)
        }
        self._hcHead.wrappedValue = DeepSeekV4HyperHead(configuration: configuration)
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    /// Reads one block of tokens.
    ///
    /// The embedding gains a copy axis on the way in, because every block reads
    /// `hc_mult` parallel copies of the residual stream, and ``hcHead`` takes
    /// that axis away again before the final norm.
    ///
    /// - Parameters:
    ///   - inputs: The token identifiers, shape `(batch, tokens)`.
    ///   - cache: One key/value cache for each layer, or `nil` for a run that
    ///     keeps no history.
    /// - Returns: The hidden states, shape `(batch, tokens, hiddenSize)`.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        DeepSeekV4NumericTrace.tokens(inputs)
        let embedded = embedTokens(inputs)
        DeepSeekV4NumericTrace.tensor("embedding", embedded)

        // The mask reads one copy of the stream, which is the embedding itself.
        let mask = createAttentionMask(h: embedded, cache: cache?.first)
        var stream = MLX.repeated(
            embedded.expandedDimensions(axis: ResidualStreamAxis.copy),
            count: hcMult, axis: ResidualStreamAxis.copy)

        for (index, layer) in layers.enumerated() {
            stream = layer(stream, mask: mask, cache: cache?[index], inputIds: inputs)
        }

        let reduced = hcHead(stream)
        DeepSeekV4NumericTrace.tensor("hc_head", reduced)
        let normalized = norm(reduced)
        DeepSeekV4NumericTrace.tensor("norm", normalized)
        return normalized
    }
}

// MARK: - Model

/// A DeepSeek-V4 language model.
public final class DeepSeekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {

    /// The number of latent key/value heads of each layer. DeepSeek-V4 keeps
    /// one, and sends it to every query head.
    public let kvHeads: [Int]

    /// The decoder stack.
    public let model: DeepSeekV4ModelInner

    /// The language-model head, or `nil` when the checkpoint ties it to the
    /// embedding table.
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// The configuration the load filter reads.
    private let configuration: DeepSeekV4Configuration

    /// Builds a DeepSeek-V4 model.
    ///
    /// - Parameter configuration: The configuration of the checkpoint.
    public init(_ configuration: DeepSeekV4Configuration) {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.numKeyValueHeads, count: configuration.numHiddenLayers)
        self.model = DeepSeekV4ModelInner(configuration: configuration)
        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabSize, bias: false)
        }
    }

    /// Reads one block of tokens and answers the logits of each one.
    ///
    /// - Parameters:
    ///   - inputs: The token identifiers, shape `(batch, tokens)`.
    ///   - cache: One key/value cache for each layer, or `nil` for a run that
    ///     keeps no history.
    /// - Returns: The logits, shape `(batch, tokens, vocabSize)`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        let logits = lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
        DeepSeekV4NumericTrace.tensor("logits", logits)
        return logits
    }

    /// The layers a LoRA adapter attaches to.
    public var loraLayers: [Module] {
        model.layers
    }

    /// The refusal that keeps DeepSeek-V4 off the plain-text prompt fallback.
    ///
    /// DeepSeek-V4 has no chat template, and `DeepSeekV4ChatEncoder` is the
    /// only correct prompt builder for this model family (card ^f0ymw6b,
    /// decision B). The plain-text fallback makes a prompt that looks correct
    /// and is wrong, thus the prompt path must throw this message instead.
    public var missingChatTemplateRefusal: String? {
        "DeepSeek-V4 has no chat template. Build the prompt with "
            + "DeepSeekV4ChatEncoder. The plain-text prompt fallback is not "
            + "permitted for this model, because it makes a wrong prompt and "
            + "gives no error."
    }

    /// The prompt path of DeepSeek-V4: the loaded tokenizer wrapped by
    /// ``DeepSeekV4EncodingTokenizer``.
    ///
    /// The `deepseek_v4` entry of the type registry is the detection rule
    /// that routes a checkpoint to this model, and this hook is where that
    /// rule turns into the encoder path: the wrapper renders every
    /// conversation with `DeepSeekV4ChatEncoder`, because the checkpoint
    /// ships no chat template. The refusal above stays as the guard of the
    /// non-encoder path.
    ///
    /// - Parameter tokenizer: the tokenizer loaded from the checkpoint.
    /// - Returns: the wrapping tokenizer.
    public func promptTokenizer(wrapping tokenizer: any Tokenizer) -> any Tokenizer {
        DeepSeekV4EncodingTokenizer(wrapping: tokenizer)
    }

    // MARK: - The load filter

    /// Maps a DeepSeek-V4 checkpoint onto the module paths of this file.
    ///
    /// The map is the one `Model.sanitize` of the Python reference makes, less
    /// the tensors this repository has no module for. A checkpoint an earlier
    /// conversion already wrote in module-path form passes through unchanged,
    /// because every rule below reads a checkpoint spelling that such a file no
    /// longer carries.
    ///
    /// - Parameter weights: The tensors of the checkpoint, by checkpoint key.
    /// - Returns: The tensors, by module path.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights
        where Self.isLoaded(key, layerCount: configuration.numHiddenLayers) {
            sanitized[Self.modulePath(of: key)] = value
        }
        stackRoutedExperts(in: &sanitized)
        if configuration.tieWordEmbeddings {
            sanitized[Self.languageModelHeadWeight] = nil
        }
        return sanitized
    }

    /// The `layers` segment of a checkpoint key.
    private static let layersSegment = "layers"

    /// The prefix of a key that names a decoder layer, before the load filter
    /// puts the module tree in front of it.
    private static let layersPrefix = "layers."

    /// The module the decoder stack sits under.
    private static let stackPrefix = "model."

    /// The prefix of every tensor of the multi-token-prediction head.
    private static let multiTokenPredictionPrefix = "mtp."

    /// The path of the language-model head weight, which a tied checkpoint
    /// does not carry.
    private static let languageModelHeadWeight = "lm_head.weight"

    /// The name each of the three expert projections carries in the
    /// checkpoint, beside the name the module tree gives it.
    private static let expertProjections = [
        (checkpoint: "w1", module: "gate_proj"),
        (checkpoint: "w2", module: "down_proj"),
        (checkpoint: "w3", module: "up_proj"),
    ]

    /// The two halves of a decoder layer that hold a hyper-connection, spelled
    /// as the checkpoint spells them.
    private static let hyperConnectionHalves = ["attn", "ffn"]

    /// The three tensors a hyper-connection holds.
    private static let hyperConnectionFields = ["fn", "base", "scale"]

    /// The tensor names one quantized projection carries. A high-precision
    /// projection carries the first alone.
    private static let projectionTensors = ["weight", "scales", "biases"]

    /// The name the mlx-community conversion gives the routing bias of a
    /// top-k gate. It is the score-correction name of the DeepSeek-V3
    /// lineage.
    private static let scoreCorrectionBiasPath = ".ffn.gate.e_score_correction_bias"

    /// The path ``DeepSeekV4MoEGate`` gives the same tensor.
    private static let gateBiasPath = ".ffn.gate.bias"

    /// The path of each top-level checkpoint tensor, by checkpoint key.
    private static let topLevelPaths = [
        "embed.weight": "model.embed_tokens.weight",
        "embed.scales": "model.embed_tokens.scales",
        "embed.biases": "model.embed_tokens.biases",
        "norm.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
        "head.scales": "lm_head.scales",
        "head.biases": "lm_head.biases",
        "hc_head_fn": "model.hc_head.fn",
        "hc_head_base": "model.hc_head.base",
        "hc_head_scale": "model.hc_head.scale",
    ]

    /// Tells whether the model has a place for one checkpoint tensor.
    ///
    /// - Parameters:
    ///   - key: The checkpoint key.
    ///   - layerCount: The number of decoder layers the configuration states.
    /// - Returns: True when the tensor loads.
    private static func isLoaded(_ key: String, layerCount: Int) -> Bool {
        // The multi-token-prediction head drafts tokens the generation loop of
        // this repository does not read.
        if key.hasPrefix(multiTokenPredictionPrefix) { return false }

        guard let layer = layerIndex(of: key) else { return true }
        return layer < layerCount
    }

    /// Reads the decoder-layer index a checkpoint key names.
    ///
    /// - Parameter key: The checkpoint key.
    /// - Returns: The index, or `nil` when the key names no layer.
    private static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let position = parts.firstIndex(of: Substring(layersSegment)),
            parts.index(after: position) < parts.endIndex
        else {
            return nil
        }
        return Int(parts[parts.index(after: position)])
    }

    /// Gives one checkpoint key its module path.
    ///
    /// - Parameter key: The checkpoint key.
    /// - Returns: The flattened module path of that tensor.
    private static func modulePath(of key: String) -> String {
        if let path = topLevelPaths[key] { return path }

        var path = key
        // The checkpoint spells the three expert projections `w1`, `w2` and
        // `w3`, both for a routed expert and for the shared expert. No other
        // DeepSeek-V4 tensor carries one of those three names.
        for projection in expertProjections {
            path = path.replacingOccurrences(
                of: ".\(projection.checkpoint).", with: ".\(projection.module).")
        }
        // The checkpoint flattens each hyper-connection into one name, where
        // the module tree carries a submodule with three tensors.
        for half in hyperConnectionHalves {
            for field in hyperConnectionFields {
                path = path.replacingOccurrences(
                    of: ".hc_\(half)_\(field)", with: ".hc_\(half).\(field)")
            }
        }
        // The mlx-community conversion spells the same hyper-connection
        // `<half>_hc.<field>`. The Python reference renames that order after
        // the rename above, thus the two spellings converge on the one
        // submodule.
        for half in hyperConnectionHalves {
            path = path.replacingOccurrences(of: ".\(half)_hc.", with: ".hc_\(half).")
        }
        // The mlx-community conversion gives the routing bias of a top-k gate
        // the score-correction name of the DeepSeek-V3 lineage.
        // ``DeepSeekV4MoEGate`` holds that tensor under `bias`, and adds it to
        // the expert scores only for the top-k selection, which is the place
        // the Python reference adds its `e_score_correction_bias`.
        path = path.replacingOccurrences(of: scoreCorrectionBiasPath, with: gateBiasPath)
        return path.hasPrefix(layersPrefix) ? stackPrefix + path : path
    }

    /// Stacks the per-expert tensors of each layer into the one tensor
    /// ``DeepSeekV4SwitchGLU`` reads.
    ///
    /// A checkpoint an earlier conversion already stacked carries no
    /// per-expert key, thus this pass finds nothing and changes nothing.
    ///
    /// - Parameter weights: The tensors, by module path. The per-expert
    ///   tensors leave, and the stacked tensor arrives.
    private func stackRoutedExperts(in weights: inout [String: MLXArray]) {
        let expertCount = configuration.nRoutedExperts
        for layer in 0 ..< configuration.numHiddenLayers {
            let expertPrefix = "\(Self.stackPrefix)\(Self.layersSegment).\(layer).ffn.experts"
            let switchPrefix = "\(Self.stackPrefix)\(Self.layersSegment).\(layer).ffn.switch_mlp"
            for projection in Self.expertProjections {
                for tensor in Self.projectionTensors {
                    let perExpert = (0 ..< expertCount).map {
                        "\(expertPrefix).\($0).\(projection.module).\(tensor)"
                    }
                    stackPerExpertWeights(
                        at: perExpert,
                        into: "\(switchPrefix).\(projection.module).\(tensor)",
                        in: &weights)
                }
            }
        }
    }

    /// Stacks one tensor of every routed expert of one layer into the one
    /// tensor ``DeepSeekV4SwitchGLU`` reads, and takes the per-expert tensors
    /// away.
    ///
    /// The stacked tensor holds one row for each expert, thus the checkpoint
    /// must carry that tensor for EVERY expert. A set that is short of one
    /// tensor makes no stack and leaves every tensor it found where it is.
    /// This is the set an already stacked checkpoint gives, which is empty,
    /// and it is also the set a high-precision projection gives for the
    /// `scales` name and the `biases` name.
    ///
    /// - Parameters:
    ///   - perExpert: The path of that tensor of each expert, in expert order.
    ///   - stackedPath: The path the stacked tensor takes.
    ///   - weights: The tensors, by module path.
    private func stackPerExpertWeights(
        at perExpert: [String], into stackedPath: String, in weights: inout [String: MLXArray]
    ) {
        let stack = perExpert.compactMap { weights[$0] }
        guard stack.count == perExpert.count else { return }
        weights[stackedPath] = stacked(stack)
        for key in perExpert {
            weights[key] = nil
        }
    }
}
