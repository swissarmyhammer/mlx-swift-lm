// Copyright © 2026 Apple Inc.
//
// One small Qwen3.5 full-attention layer and a filled key cache, for the tests
// that check which attention kernel the layer reads its cache through.

import Foundation
import MLX

@testable import MLXLLM
@testable import MLXLMCommon

/// One small dense Qwen3.5 layer with full attention: six query heads for
/// one KV head, as in Qwen3.8, and a head width that each test picks.
struct Qwen35AttentionLayerFixture {

    /// The number of query heads; six for each KV head, as in Qwen3.8.
    static let queryHeads = 6

    /// The number of KV heads.
    static let kvHeads = 1

    /// The width of the residual stream.
    static let hiddenSize = 32

    /// The index of the full-attention layer: with a full-attention interval
    /// of 2, layer 1 is the first full-attention layer.
    private static let attentionLayer = 1

    /// The number of random tensors of a filled cache: the keys and the
    /// values.
    private static let keysAndValues = 2

    /// The width of one attention head.
    let headDim: Int

    /// A small dense Qwen3.5 text configuration with one full-attention
    /// layer.
    func configuration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": \(Self.hiddenSize),
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": \(Self.queryHeads),
                "num_key_value_heads": \(Self.kvHeads),
                "head_dim": \(headDim),
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 2
            }
            """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    /// The full-attention layer of the configuration.
    func layer() throws -> Qwen35DecoderLayer {
        Qwen35DecoderLayer(try configuration(), layerIdx: Self.attentionLayer)
    }

    /// A key cache that holds `keys` random keys and values.
    ///
    /// - Parameters:
    ///   - keys: The number of cached keys.
    ///   - seed: The seed of the random keys and values.
    /// - Returns: The filled cache.
    func filledCache(keys: Int, seed: UInt64) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let rows = MLXRandom.normal(
            [Self.keysAndValues, 1, Self.kvHeads, keys, headDim], key: MLXRandom.key(seed))
        _ = cache.update(keys: rows[0], values: rows[1])
        eval(cache.state)
        return cache
    }
}
