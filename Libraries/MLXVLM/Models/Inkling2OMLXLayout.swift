// Copyright © 2026 Apple Inc.
//
// The oMLX layout of an Inkling 2 checkpoint (jundot/omlx `omlx/patches/inkling2`, as in
// `mlx-community/Inkling-2-Mini-oQ4e-mtp`), mapped onto the names and shapes of the Hugging Face
// layout that the port's modules carry: fused projections split back into their rows, the loop
// controllers and message queries unstacked, the convolutions given their channel axis, and the
// per-layer quantization of `config.json` renamed to the paths of the split projections.

import Foundation
import MLX
import MLXLMCommon

/// Maps the tensors and the per-layer quantization of an oMLX checkpoint onto the port's layout.
///
/// Each key maps on its own, thus a checkpoint of only the vision tower, as an on-demand load
/// reads it, maps as the whole checkpoint does. A key of the Hugging Face layout passes through.
enum Inkling2OMLXLayout {

    /// The message inputs of a loop layer in the order oMLX stacks them: the query, key and
    /// value inputs of the attention, then the MLP input.
    private static let messageGates = ["q", "k", "v", "mlp"]

    /// The halves of the input of a loop controller's down projection: the normalized state and
    /// the normalized message.
    private static let controllerInputHalves = 2

    /// The prefix of the text tensors of an oMLX checkpoint.
    private static let textPrefix = "model."

    /// The scopes under `model.` that belong to the Hugging Face layout.
    private static let huggingFaceScopes = ["model.language_model.", "model.vision_tower."]

    /// The prefix of the vision backbone of an oMLX checkpoint.
    private static let visionBackbone = "vision_tower.backbone."

    /// Whether `weights` holds tensors of the oMLX layout.
    static func holdsOMLXLayout(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { isOMLXText($0) || isOMLXVision($0) }
    }

    /// The tensors of `weights` with each oMLX key mapped onto the port's layout.
    ///
    /// - Parameters:
    ///   - weights: the checkpoint tensors, with their names as the files store them.
    ///   - text: the text configuration, which gives the row counts of the fused projections.
    /// - Returns: the tensors under the names of the Hugging Face layout, with the `model.` scope
    ///   dropped from the text tensors as ``Inkling2/prepareCheckpoint(_:)`` drops it.
    static func remap(_ weights: [String: MLXArray], text: Inkling2TextConfiguration) -> [String:
        MLXArray]
    {
        var result = [String: MLXArray](minimumCapacity: weights.count)
        var messageQueries: [Int: MLXArray] = [:]
        for (key, value) in weights {
            if isOMLXVision(key) {
                result[visionKey(key)] = value
            } else if isOMLXText(key) {
                let name = String(key.dropFirst(textPrefix.count))
                if let (layer, rest) = layerKey(name) {
                    if rest == "loop_message.query" {
                        messageQueries[layer] = value
                    } else {
                        for (port, tensor) in layerTensors(layer, rest, value, text: text) {
                            result["language_model.layers.\(layer).\(port)"] = tensor
                        }
                    }
                } else {
                    let (port, tensor) = globalTensor(name, value)
                    result["language_model.\(port)"] = tensor
                }
            } else {
                result[key] = value
            }
        }
        if let loop = text.loop, !messageQueries.isEmpty {
            let core = loop.core.compactMap { messageQueries[$0] }
            if core.count == loop.core.count {
                result["language_model.message_input.query"] = stacked(core)
            } else {
                for (layer, value) in messageQueries {
                    result["language_model.layers.\(layer).loop_message.query"] = value
                }
            }
        }
        return result
    }

    /// The per-layer quantization of an oMLX `config.json` under the module paths of the port:
    /// each fused projection's entry for each of its parts. Entries of other modules, as the
    /// experts, whose format the tensor shapes give, drop out.
    ///
    /// - Parameter quantization: the quantization the checkpoint's configuration states.
    /// - Returns: the quantization under the port's paths.
    static func remap(_ quantization: BaseConfiguration.PerLayerQuantization)
        -> BaseConfiguration.PerLayerQuantization
    {
        var perLayer: [String: BaseConfiguration.QuantizationOption] = [:]
        for (path, option) in quantization.perLayerQuantization {
            guard path.hasPrefix(textPrefix) || path == "lm_head" else {
                perLayer[path] = option
                continue
            }
            for port in modulePaths(path) {
                perLayer[port] = option
            }
        }
        return .init(quantization: quantization.quantization, perLayerQuantization: perLayer)
    }

    // MARK: - Keys

    private static func isOMLXText(_ key: String) -> Bool {
        key.hasPrefix(textPrefix) && !huggingFaceScopes.contains { key.hasPrefix($0) }
    }

    private static func isOMLXVision(_ key: String) -> Bool {
        key.hasPrefix(visionBackbone) || key.hasPrefix("vision_tower.proj_in.")
            || key.hasPrefix("vision_tower.proj_out.")
    }

    /// The prefix of the layer tensors under the text scope.
    private static let layersPrefix = "layers."

    /// The layer and the rest of a `layers.N.rest` name, or `nil` for another name.
    private static func layerKey(_ name: String) -> (Int, String)? {
        guard name.hasPrefix(layersPrefix) else { return nil }
        let tail = name.dropFirst(layersPrefix.count)
        guard let dot = tail.firstIndex(of: "."), let layer = Int(tail[..<dot]) else {
            return nil
        }
        return (layer, String(tail[tail.index(after: dot)...]))
    }

    /// The renames of the vision tower: the backbone's layers, norms and layer scales, and the
    /// two projections of the merger.
    private static let visionRenames: [(String, String)] = [
        ("vision_tower.proj_in.", "vision_tower.merger.linear_1."),
        ("vision_tower.proj_out.", "vision_tower.merger.linear_2."),
        (".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."),
        (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2"),
        (".mlp.c_fc.", ".mlp.fc1."), (".mlp.c_proj.", ".mlp.fc2."),
        (".ln_pre.", ".pre_layernorm."),
    ]

    private static func visionKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix(visionBackbone) {
            key = "vision_tower." + key.dropFirst(visionBackbone.count)
        }
        if key == "vision_tower.embeddings.position_embedding" {
            return "vision_tower.embeddings.position_embedding.weight"
        }
        for (from, to) in visionRenames {
            key = key.replacingOccurrences(of: from, with: to)
        }
        return key
    }

    /// A text tensor outside the layers: the embedding, its norm, the final norm and the query
    /// of the final depth readout, which oMLX stores as one row.
    private static func globalTensor(_ name: String, _ value: MLXArray) -> (String, MLXArray) {
        if name.hasPrefix("embed.") {
            return ("embed_tokens." + name.dropFirst("embed.".count), value)
        }
        if name == "final_input.query" {
            return (name, singleRow(value))
        }
        return (name, value)
    }

    /// The renames inside a layer that keep the tensor as it is.
    private static let layerRenames: [String: String] = [
        "attn_norm.weight": "input_layernorm.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "q_input_norm.weight": "q_input_layernorm.weight",
        "k_input_norm.weight": "k_input_layernorm.weight",
        "v_input_norm.weight": "v_input_layernorm.weight",
        "attn.q_scale": "self_attn.q_l2_clip.scale",
        "attn.r_scale": "self_attn.rel_l2_clip.scale",
        "attn.rel_proj": "self_attn.rel_logits_proj.proj",
        "attn.k_static": "self_attn.k_sconv.static_weight",
        "attn.v_static": "self_attn.v_sconv.static_weight",
        "mlp.router_weight": "mlp.gate.weight",
        "mlp.router_bias": "mlp.gate.e_score_correction_bias",
    ]

    /// The renames inside a layer of the depthwise convolutions, which oMLX stores as
    /// `[channels, taps]` and the port as `[channels, taps, 1]`.
    private static let convolutionRenames: [String: String] = [
        "attn.k_conv": "self_attn.k_sconv.conv1d.weight",
        "attn.v_conv": "self_attn.v_sconv.conv1d.weight",
        "attn_conv": "post_attention_convolution.conv1d.weight",
        "mlp_conv": "post_mlp_convolution.conv1d.weight",
    ]

    /// The scope of the routed experts inside a layer.
    private static let expertsScope = "mlp.experts."

    /// The renames of the modules inside a layer whose tensors keep their suffixes.
    private static let moduleRenames: [String: String] = [
        "attn.o_proj": "self_attn.o_proj",
        "mlp.experts.w13": "mlp.experts.gate_up_proj",
        "mlp.experts.w2": "mlp.experts.down_proj",
    ]

    /// The port tensors of the tensor `rest` of `layer`.
    private static func layerTensors(
        _ layer: Int, _ rest: String, _ value: MLXArray, text: Inkling2TextConfiguration
    ) -> [(String, MLXArray)] {
        if let port = layerRenames[rest] {
            return [(port, value)]
        }
        if let port = convolutionRenames[rest] {
            return [(port, value.expandedDimensions(axis: -1))]
        }
        if rest == "attn_input.query" {
            return [("attention_input.query", singleRow(value))]
        }
        if rest == "mlp_input.query" {
            return [(rest, singleRow(value))]
        }
        if rest == "loop_down" {
            return loopDown(value)
        }
        if rest == "loop_gain_forget" {
            return loopGainForget(value)
        }
        let (module, suffix) = splitSuffix(rest)
        if let port = moduleRenames[module] {
            return [("\(port).\(suffix)", value)]
        }
        if let parts = fusedParts(module, layer: layer, text: text) {
            var start = 0
            return parts.map { name, rows in
                defer { start += rows }
                return ("\(name).\(suffix)", value[start ..< (start + rows)])
            }
        }
        return [(rest, value)]
    }

    /// The module and the tensor suffix of `rest`: `attn.in_proj` and `weight`.
    private static func splitSuffix(_ rest: String) -> (String, String) {
        guard let dot = rest.lastIndex(of: ".") else { return (rest, "") }
        return (String(rest[..<dot]), String(rest[rest.index(after: dot)...]))
    }

    /// The row count of a part of a fused projection, which the layer's attention gives.
    private enum PartRows {
        /// The queries, or the gate: one row per query head and head dimension.
        case queries
        /// The keys, or the values: one row per key-value head and head dimension.
        case keysOrValues
        /// The relative states: one row per query head and relative dimension.
        case relative
        /// The sinks: one row per query head.
        case heads
        /// The taps of a dynamic convolution: one row per key-value head and tap.
        case taps
    }

    /// The parts of each fused projection in the order of its rows, under the layer's
    /// `self_attn.` scope.
    ///
    /// A sliding layer fuses every projection of the attention input into `in_proj`; a full
    /// layer fuses the query side into `q_proj`, and each of `k_proj` and `v_proj` with the
    /// weight projection of its dynamic convolution.
    private static let fusedProjections: [String: [(String, PartRows)]] = {
        var table: [String: [(String, PartRows)]] = [
            "attn.in_proj": [
                ("q_proj", .queries), ("k_proj", .keysOrValues), ("v_proj", .keysOrValues),
                ("r_proj", .relative), ("gate_proj", .queries), ("sink_proj", .heads),
            ],
            "attn.q_proj": [
                ("q_proj", .queries), ("r_proj", .relative), ("gate_proj", .queries),
                ("sink_proj", .heads),
            ],
        ]
        for side in ["k", "v"] {
            table["attn.\(side)_proj"] = [
                ("\(side)_proj", .keysOrValues), ("\(side)_sconv.weight_proj", .taps),
            ]
        }
        return table
    }()

    /// The parts of a fused projection of `layer` with their row counts, or `nil` when `module`
    /// is not a fused projection.
    private static func fusedParts(
        _ module: String, layer: Int, text: Inkling2TextConfiguration
    ) -> [(String, Int)]? {
        guard let parts = fusedProjections[module] else { return nil }
        let sliding = text.isSliding(layer)
        let heads = sliding ? text.slidingAttentionHeads : text.attentionHeads
        let kvHeads = sliding ? text.slidingKVHeads : text.kvHeads
        let headDim = sliding ? text.slidingHeadDim : text.headDim
        return parts.map { name, rows in
            let count =
                switch rows {
                case .queries: heads * headDim
                case .keysOrValues: kvHeads * headDim
                case .relative: heads * text.relativeDim
                case .heads: heads
                case .taps: kvHeads * text.convolutionKernelSize
                }
            return ("self_attn.\(name)", count)
        }
    }

    /// The axes of one controller's down projection in oMLX, `[2, D, rank]`.
    private static let (halfAxis, widthAxis, rankAxis) = (0, 1, 2)

    /// The down projections of the loop controllers, `[4, 2, D, rank]` in oMLX, as the
    /// `[rank, 2 * D]` weight of each message gate. oMLX widens them to float32 from the
    /// bfloat16 of the source, thus the cast back is exact.
    private static func loopDown(_ value: MLXArray) -> [(String, MLXArray)] {
        messageGates.enumerated().map { index, gate in
            let down = value[index]
            let rank = down.dim(-1)
            // [2, D, rank] -> [rank, 2, D]: each row of the weight reads both halves in turn.
            let weight = down.transposed(rankAxis, halfAxis, widthAxis)
                .reshaped(rank, controllerInputHalves * down.dim(widthAxis))
            return ("\(gate)_message_gate.down_proj.weight", weight.asType(.bfloat16))
        }
    }

    /// The gain and forget projections of the loop controllers, `[4, rank, 2 * D]` in oMLX with
    /// the gain first, as the `[D, rank]` weights of each message gate.
    private static func loopGainForget(_ value: MLXArray) -> [(String, MLXArray)] {
        messageGates.enumerated().flatMap { index, gate in
            let gainForget = value[index]
            let width = gainForget.dim(-1) / controllerInputHalves
            return [
                ("\(gate)_message_gate.message_gate_proj.weight", gainForget[0..., ..<width].T),
                ("\(gate)_message_gate.forget_gate_proj.weight", gainForget[0..., width...].T),
            ]
        }
    }

    /// The axes of a query that oMLX stores as one row: `[1, D]`.
    private static let rowAxes = 2

    /// A query that oMLX stores as one row `[1, D]`, as the port's `[D]`.
    private static func singleRow(_ value: MLXArray) -> MLXArray {
        value.ndim == rowAxes && value.dim(0) == 1 ? value.reshaped(-1) : value
    }

    /// The module paths of the port that an oMLX quantization path covers.
    private static func modulePaths(_ path: String) -> [String] {
        if path == "lm_head" { return [path] }
        let name = String(path.dropFirst(textPrefix.count))
        if name == "embed" { return ["language_model.embed_tokens"] }
        guard let (layer, module) = layerKey(name) else { return [] }
        let prefix = "language_model.layers.\(layer)."
        // The experts take their format from the shapes of their tensors.
        if let port = moduleRenames[module], !port.hasPrefix(expertsScope) {
            return [prefix + port]
        }
        return (fusedProjections[module] ?? []).map { prefix + "self_attn." + $0.0 }
    }
}
