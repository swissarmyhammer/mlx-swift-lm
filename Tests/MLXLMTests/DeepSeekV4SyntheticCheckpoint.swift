// Copyright © 2026 Apple Inc.

import Foundation

@testable import MLXLLM

/// The `config.json` of a small synthetic DeepSeek-V4 checkpoint.
///
/// More than one suite decodes this checkpoint. The model only has to exist —
/// no suite that uses it runs a forward pass with real weights.
enum DeepSeekV4SyntheticCheckpoint {

    /// The text of the `config.json`.
    ///
    /// The text names the model type, because a suite that reads this
    /// checkpoint through `LLMModelFactory` gets its model from that key. A
    /// suite that decodes `DeepSeekV4Configuration` on its own ignores the
    /// key.
    static let configJSON = """
        {
          "model_type": "deepseek_v4",
          "vocab_size": 12,
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "qk_rope_head_dim": 4,
          "q_lora_rank": 8,
          "rms_norm_eps": 1e-6,
          "max_position_embeddings": 64,
          "o_groups": 2,
          "o_lora_rank": 4,
          "n_routed_experts": 8,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 8,
          "num_hash_layers": 1,
          "norm_topk_prob": true,
          "routed_scaling_factor": 1.0,
          "swiglu_limit": 10.0,
          "hc_mult": 2,
          "hc_sinkhorn_iters": 4,
          "hc_eps": 1e-6,
          "rope_theta": 10000.0,
          "compress_ratios": [],
          "use_attn_sink": true,
          "tie_word_embeddings": false
        }
        """

    /// Decodes the checkpoint.
    /// - Returns: the configuration.
    static func configuration() throws -> DeepSeekV4Configuration {
        try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(configJSON.utf8))
    }
}
