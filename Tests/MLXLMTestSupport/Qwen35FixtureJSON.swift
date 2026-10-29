// Copyright © 2026 Apple Inc.

/// The JSON of small Qwen 3.5 configurations, for models on random weights.
///
/// Shared by the Qwen 3.5 suites of `MLXLMTests` (`Qwen35MTPTests`,
/// `Qwen35RecurrentCachePositionTests`, `HybridRecurrentCacheOffsetTests`,
/// `ChatSessionRecurrentCheckpointTests`) and by ``Qwen35MTPFixture``, thus one
/// configuration serves each suite.
public enum Qwen35FixtureJSON {

    /// The width of the hidden state of the default configuration.
    public static let defaultHiddenSize = 16

    /// The epsilon of each RMS norm of the configurations.
    public static let rmsNormEpsilon: Float = 1e-6

    /// The vocabulary size of the default configuration.
    public static let defaultVocabularySize = 16

    /// The width of each full-attention head of the default configuration.
    public static let defaultHeadDimension = 8

    /// The JSON of a small Qwen 3.5 text configuration.
    ///
    /// - Parameters:
    ///   - mtpLayers: The number of multi-token prediction layers.
    ///   - mtpUseDedicatedEmbeddings: Whether the prediction layers have their own embeddings.
    ///   - numExperts: The number of experts of each MoE block, or 0 for a dense model.
    ///   - expertsPerToken: The number of experts that each token uses in an MoE block.
    ///   - hiddenLayers: The number of decoder layers.
    ///   - fullAttentionInterval: One layer of each interval is a full-attention layer, and each
    ///     other layer is a linear (recurrent) layer. 1 makes every layer a full-attention layer.
    ///   - hiddenSize: The width of the hidden state.
    ///   - vocabularySize: The number of tokens of the vocabulary.
    ///   - tieWordEmbeddings: Whether the LM head reuses the token embeddings. A model with
    ///     its own LM head has a token-to-token map that is not the identity.
    ///   - headDimension: The width of each full-attention head. A quantized KV cache needs a
    ///     width its quantization group size divides.
    /// - Returns: The JSON text.
    public static func text(
        mtpLayers: Int,
        mtpUseDedicatedEmbeddings: Bool = false,
        numExperts: Int = 0,
        expertsPerToken: Int = 1,
        hiddenLayers: Int = 1,
        fullAttentionInterval: Int = 1,
        hiddenSize: Int = defaultHiddenSize,
        vocabularySize: Int = defaultVocabularySize,
        tieWordEmbeddings: Bool = true,
        headDimension: Int = defaultHeadDimension
    ) -> String {
        """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": \(hiddenSize),
          "num_hidden_layers": \(hiddenLayers),
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": \(headDimension),
          "linear_num_value_heads": 2,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 2,
          "rms_norm_eps": \(rmsNormEpsilon),
          "vocab_size": \(vocabularySize),
          "rope_theta": 100000.0,
          "partial_rotary_factor": 0.25,
          "max_position_embeddings": 64,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "attention_bias": false,
          "full_attention_interval": \(fullAttentionInterval),
          "mtp_num_hidden_layers": \(mtpLayers),
          "mtp_use_dedicated_embeddings": \(mtpUseDedicatedEmbeddings),
          "num_experts": \(numExperts),
          "num_experts_per_tok": \(numExperts == 0 ? 0 : expertsPerToken),
          "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16,
          "rope_parameters": {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 0.25
          }
        }
        """
    }

    /// The JSON of a small Qwen 3.5 vision-language configuration over a text configuration
    /// from ``text(mtpLayers:mtpUseDedicatedEmbeddings:numExperts:expertsPerToken:hiddenLayers:fullAttentionInterval:hiddenSize:vocabularySize:tieWordEmbeddings:headDimension:)``.
    ///
    /// - Parameters:
    ///   - mtpLayers: The number of multi-token prediction layers.
    ///   - hiddenSize: The width of the hidden state.
    ///   - vocabularySize: The number of tokens of the vocabulary.
    /// - Returns: The JSON text, with a tiny vision tower.
    public static func visionLanguage(
        mtpLayers: Int,
        hiddenSize: Int = defaultHiddenSize,
        vocabularySize: Int = defaultVocabularySize
    ) -> String {
        visionLanguage(
            textConfigJSON: text(
                mtpLayers: mtpLayers, hiddenSize: hiddenSize, vocabularySize: vocabularySize))
    }

    /// The JSON of a small Qwen 3.5 vision-language configuration.
    ///
    /// - Parameter textConfigJSON: The JSON of the text configuration, from
    ///   ``text(mtpLayers:mtpUseDedicatedEmbeddings:numExperts:expertsPerToken:hiddenLayers:fullAttentionInterval:hiddenSize:vocabularySize:tieWordEmbeddings:headDimension:)``.
    /// - Returns: The JSON text, with a tiny vision tower.
    public static func visionLanguage(textConfigJSON: String) -> String {
        """
        {
          "model_type": "qwen3_5",
          "text_config": \(textConfigJSON),
          "vision_config": {
            "model_type": "qwen3_5_vit",
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "out_hidden_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "num_position_embeddings": 16
          }
        }
        """
    }

    /// The JSON of a Qwen 3.5 configuration that wraps a small text configuration without a
    /// vision tower.
    ///
    /// - Parameters:
    ///   - modelType: The model type of the wrapper.
    ///   - textConfigJSON: The JSON of the text configuration.
    /// - Returns: The JSON text.
    public static func wrappedText(
        modelType: String, textConfigJSON: String = text(mtpLayers: 1)
    ) -> String {
        """
        {
          "model_type": "\(modelType)",
          "text_config": \(textConfigJSON)
        }
        """
    }

    /// The JSON of a standalone Qwen 3.5 MTP checkpoint configuration, the layout that
    /// publishes the drafter apart from its target.
    ///
    /// - Returns: The JSON text.
    public static func standaloneMTP() -> String {
        """
        {
          "model_type": "qwen3_5_mtp",
          "block_size": 3,
          "text_config": \(text(mtpLayers: 1)),
          "tie_word_embeddings": true,
          "vision_config": {}
        }
        """
    }
}
