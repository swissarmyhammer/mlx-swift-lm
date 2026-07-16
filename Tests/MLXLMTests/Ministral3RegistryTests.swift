// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXLLM

// Regression coverage for `model_type: "ministral3"` — the type string that
// Mistral's large Devstral 2 checkpoints declare (e.g.
// `mlx-community/Devstral-2-123B-Instruct-2512-4bit`). The architecture is the
// one `Mistral3Text.swift` ports from mlx-lm's `ministral3.py`, so the registry
// aliases it to `Mistral3TextConfiguration`/`Mistral3TextModel` — the same
// implementation the 24B `mistral3` (Devstral-Small-2) checkpoints resolve to.

/// Config JSON mirroring the real Devstral-2-123B-Instruct-2512-4bit
/// `config.json` shape — yarn `rope_parameters` carrying `rope_theta` and
/// `llama_4_scaling_beta`, no `layer_types`, null `sliding_window`,
/// untied embeddings — with tiny dimensions so model construction stays cheap.
private let ministral3TinyConfigJSON = """
    {
      "architectures": ["Ministral3ForCausalLM"],
      "model_type": "ministral3",
      "head_dim": 2,
      "hidden_size": 4,
      "intermediate_size": 8,
      "max_position_embeddings": 262144,
      "num_attention_heads": 2,
      "num_hidden_layers": 1,
      "num_key_value_heads": 1,
      "rms_norm_eps": 1e-5,
      "rope_parameters": {
        "beta_fast": 4.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0,
        "original_max_position_embeddings": 4096,
        "llama_4_scaling_beta": 0.0,
        "rope_theta": 1000000.0,
        "rope_type": "yarn",
        "type": "yarn"
      },
      "sliding_window": null,
      "tie_word_embeddings": false,
      "vocab_size": 16
    }
    """

@Test
func ministral3RegistryResolvesToMistral3TextModel() async throws {
    let model = try await LLMTypeRegistry.shared.createModel(
        configuration: Data(ministral3TinyConfigJSON.utf8), modelType: "ministral3")
    #expect(model is Mistral3TextModel)
}

@Test
func ministral3ConfigurationDecodesDevstral2Shape() throws {
    let config = try JSONDecoder.json5().decode(
        Mistral3TextConfiguration.self, from: Data(ministral3TinyConfigJSON.utf8))

    #expect(config.modelType == "ministral3")
    #expect(config.tieWordEmbeddings == false)
    #expect(config.slidingWindow == nil)
    // No `layer_types` in the checkpoint config: every layer is full attention.
    #expect(config.layerTypes == Array(repeating: "full_attention", count: config.hiddenLayers))
    // rope_theta lives inside rope_parameters for this family.
    #expect(config.ropeParameters?["rope_theta"]?.asFloat() == 1_000_000.0)
    #expect(config.ropeParameters?["rope_type"] == .string("yarn"))
}
