// Copyright © 2026 Apple Inc.
//
// The oMLX layout of an Inkling 2 checkpoint maps back onto the port's own parameters: the tests
// write the tiny model's parameters in the oMLX layout, as `omlx/patches/inkling2/loader.py`
// builds it from the Hugging Face layout, and require the mapping to give every parameter back.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("Inkling2 oMLX layout")
struct Inkling2OMLXLayoutTests {

    /// The message inputs in the order oMLX stacks them.
    private static let messageGates = ["q", "k", "v", "mlp"]

    /// The tensors of `parameters`, the parameters of a model of `text`, in the oMLX layout, built
    /// as oMLX builds them.
    private static func omlxTensors(
        of parameters: [String: MLXArray], text: Inkling2TextConfiguration
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        let layerRenames: [String: String] = [
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "mlp_norm.weight",
            "q_input_layernorm.weight": "q_input_norm.weight",
            "k_input_layernorm.weight": "k_input_norm.weight",
            "v_input_layernorm.weight": "v_input_norm.weight",
            "self_attn.q_l2_clip.scale": "attn.q_scale",
            "self_attn.rel_l2_clip.scale": "attn.r_scale",
            "self_attn.rel_logits_proj.proj": "attn.rel_proj",
            "self_attn.k_sconv.static_weight": "attn.k_static",
            "self_attn.v_sconv.static_weight": "attn.v_static",
            "mlp.gate.weight": "mlp.router_weight",
            "mlp.gate.e_score_correction_bias": "mlp.router_bias",
            "self_attn.o_proj.weight": "attn.o_proj.weight",
        ]
        let convolutions: [String: String] = [
            "self_attn.k_sconv.conv1d.weight": "attn.k_conv",
            "self_attn.v_sconv.conv1d.weight": "attn.v_conv",
            "post_attention_convolution.conv1d.weight": "attn_conv",
            "post_mlp_convolution.conv1d.weight": "mlp_conv",
        ]
        for layer in 0 ..< text.hiddenLayers {
            let from = "language_model.layers.\(layer)."
            let to = "model.layers.\(layer)."
            func take(_ name: String) -> MLXArray? { parameters[from + name] }
            for (name, value) in parameters where name.hasPrefix(from) {
                let rest = String(name.dropFirst(from.count))
                if let renamed = layerRenames[rest] {
                    out[to + renamed] = value
                } else if let renamed = convolutions[rest] {
                    out[to + renamed] = value.squeezed(axis: -1)
                } else if rest == "attention_input.query" || rest == "mlp_input.query" {
                    let renamed = rest == "mlp_input.query" ? rest : "attn_input.query"
                    out[to + renamed] = value.ndim == 1 ? value.reshaped(1, -1) : value
                } else if rest.hasPrefix("mlp.experts.gate_up_proj.") {
                    out[to + "mlp.experts.w13." + rest.split(separator: ".").last!] = value
                } else if rest.hasPrefix("mlp.experts.down_proj.") {
                    out[to + "mlp.experts.w2." + rest.split(separator: ".").last!] = value
                }
            }
            let attention = "self_attn."
            func rows(_ names: [String]) -> MLXArray {
                concatenated(names.map { take(attention + $0 + ".weight")! }, axis: 0)
            }
            if text.isSliding(layer) {
                out[to + "attn.in_proj.weight"] = rows([
                    "q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj",
                ])
            } else {
                out[to + "attn.q_proj.weight"] = rows([
                    "q_proj", "r_proj", "gate_proj", "sink_proj",
                ])
                out[to + "attn.k_proj.weight"] = rows(["k_proj", "k_sconv.weight_proj"])
                out[to + "attn.v_proj.weight"] = rows(["v_proj", "v_sconv.weight_proj"])
            }
            if let firstDown = take("q_message_gate.down_proj.weight") {
                let (rank, width) = (firstDown.dim(0), firstDown.dim(1) / 2)
                out[to + "loop_down"] = stacked(
                    messageGates.map { gate in
                        take("\(gate)_message_gate.down_proj.weight")!
                            .reshaped(rank, 2, width).transposed(1, 2, 0)
                    })
                out[to + "loop_gain_forget"] = stacked(
                    messageGates.map { gate in
                        concatenated(
                            [
                                take("\(gate)_message_gate.message_gate_proj.weight")!.T,
                                take("\(gate)_message_gate.forget_gate_proj.weight")!.T,
                            ], axis: 1)
                    })
            }
        }
        if let loop = text.loop, let queries = parameters["language_model.message_input.query"] {
            for (index, layer) in loop.core.enumerated() {
                out["model.layers.\(layer).loop_message.query"] = queries[index]
            }
        }
        out["model.embed.weight"] = parameters["language_model.embed_tokens.weight"]
        out["model.embed_norm.weight"] = parameters["language_model.embed_norm.weight"]
        out["model.norm.weight"] = parameters["language_model.norm.weight"]
        out["model.final_input.query"] = parameters["language_model.final_input.query"]?
            .reshaped(1, -1)
        out["lm_head.weight"] = parameters["lm_head.weight"]
        return out
    }

    @Test("Every parameter comes back from the oMLX layout")
    func everyParameterComesBack() throws {
        let model = try Inkling2ModelTests.makeModel()
        let text = model.config.textConfiguration
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let omlx = Self.omlxTensors(of: parameters, text: text)
        #expect(Inkling2OMLXLayout.holdsOMLXLayout(omlx))

        let mapped = Inkling2OMLXLayout.remap(omlx, text: text)
        #expect(Set(mapped.keys) == Set(parameters.keys))
        for (key, expected) in parameters {
            guard let value = mapped[key] else { continue }
            // The down projections of the loop controllers are bfloat16, as in the source.
            let want =
                key.hasSuffix("_message_gate.down_proj.weight")
                ? expected.asType(.bfloat16) : expected
            #expect(value.shape == want.shape, "\(key)")
            #expect(value.dtype == want.dtype, "\(key)")
            #expect(value.shape == want.shape && arrayEqual(value, want).item(Bool.self), "\(key)")
        }
    }

    /// In bfloat16, as a loaded checkpoint runs, the loop controllers' down projections are
    /// bfloat16 as the mapping gives them.
    @Test("A checkpoint of the oMLX layout loads into the model and decodes as the original")
    func checkpointLoads() throws {
        let model = try Inkling2ModelTests.makeModel(bfloat16: true)
        let text = model.config.textConfiguration
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let checkpoint = try model.prepareCheckpoint(
            ModelCheckpoint(weights: Self.omlxTensors(of: parameters, text: text)))
        let loaded = try Inkling2ModelTests.makeModel(bfloat16: true)
        try loaded.update(
            parameters: ModuleParameters.unflattened(checkpoint.weights), verify: .all)
        let tokens = MLXArray((0 ..< 12).map { Int32(($0 * 5 + 2) % 48) }).reshaped(1, -1)
        #expect(arrayEqual(model(tokens, cache: nil), loaded(tokens, cache: nil)).item(Bool.self))
    }

    @Test("The vision tower of the oMLX layout takes the Hugging Face names")
    func visionNames() throws {
        let value = MLXArray.zeros([2])
        let names = [
            "vision_tower.backbone.embeddings.position_embedding":
                "vision_tower.embeddings.position_embedding.weight",
            "vision_tower.backbone.embeddings.class_embedding":
                "vision_tower.embeddings.class_embedding",
            "vision_tower.backbone.layers.3.ln_1.weight":
                "vision_tower.layers.3.layernorm_before.weight",
            "vision_tower.backbone.layers.3.ln_2.bias":
                "vision_tower.layers.3.layernorm_after.bias",
            "vision_tower.backbone.layers.3.ls_1.gamma": "vision_tower.layers.3.lambda_1",
            "vision_tower.backbone.layers.3.ls_2.gamma": "vision_tower.layers.3.lambda_2",
            "vision_tower.backbone.layers.3.mlp.c_fc.weight":
                "vision_tower.layers.3.mlp.fc1.weight",
            "vision_tower.backbone.layers.3.mlp.c_proj.bias": "vision_tower.layers.3.mlp.fc2.bias",
            "vision_tower.backbone.layers.3.self_attn.out_proj.weight":
                "vision_tower.layers.3.self_attn.out_proj.weight",
            "vision_tower.backbone.ln_pre.weight": "vision_tower.pre_layernorm.weight",
            "vision_tower.proj_in.weight": "vision_tower.merger.linear_1.weight",
            "vision_tower.proj_out.weight": "vision_tower.merger.linear_2.weight",
        ]
        let mapped = Inkling2OMLXLayout.remap(
            names.mapValues { _ in value },
            text: try Inkling2ModelTests.makeModel().config.textConfiguration)
        #expect(Set(mapped.keys) == Set(names.values))
    }

    @Test("The quantization of each fused projection covers every part")
    func quantizationPaths() {
        let eight = BaseConfiguration.QuantizationOption.quantize(
            .init(groupSize: 64, bits: 8))
        let quantization = BaseConfiguration.PerLayerQuantization(
            quantization: .init(groupSize: 64, bits: 4),
            perLayerQuantization: [
                "model.embed": eight, "lm_head": eight,
                "model.layers.0.attn.in_proj": eight, "model.layers.0.attn.o_proj": eight,
                "model.layers.5.attn.q_proj": eight, "model.layers.5.attn.k_proj": eight,
                "model.layers.5.mlp.experts.w13": eight,
            ])
        let mapped = Inkling2OMLXLayout.remap(quantization)
        let attention = "language_model.layers.0.self_attn."
        let full = "language_model.layers.5.self_attn."
        let expected =
            ["language_model.embed_tokens", "lm_head", attention + "o_proj"]
            + ["q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj"].map {
                attention + $0
            }
            + ["q_proj", "r_proj", "gate_proj", "sink_proj", "k_proj", "k_sconv.weight_proj"].map {
                full + $0
            }
        #expect(Set(mapped.perLayerQuantization.keys) == Set(expected))
        #expect(mapped.quantization(layer: attention + "sink_proj")?.bits == 8)
        #expect(mapped.quantization(layer: "language_model.layers.1.self_attn.q_proj")?.bits == 4)
    }
}
