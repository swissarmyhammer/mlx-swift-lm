// Copyright © 2026 Apple Inc.
//
// A check of the LFM3 caches over long contexts: a tiny LFM3 model prefills 4k, 16k and 32k
// tokens in chunks of 257 and of 512, and the two chunkings must leave the same caches and the
// same final logits. It needs no checkpoint. It came from `MLXLMTests`, where the environment
// variable `MLX_LFM3_LONG_CONTEXT` gated it; here it runs in each run of the
// `IntegrationTesting` scheme. To run it alone:
//
//     xcodebuild test -skipPackagePluginValidation \
//         -project IntegrationTesting/IntegrationTesting.xcodeproj \
//         -scheme IntegrationTesting -destination 'platform=macOS' \
//         -only-testing:IntegrationTestingTests/LFM3LongContextProfileTests
//
// Each measurement line starts with `LFM3 context`.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFM3LongContextProfileTests: XCTestCase {

    /// The configuration of the tiny LFM3 model of `LFM3ReferenceTests` in `MLXLMTests`.
    private static let configuration = """
        {
          "model_type": "lfm3",
          "hidden_size": 8,
          "intermediate_size": 16,
          "num_hidden_layers": 3,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 8,
          "vocab_size": 16,
          "layer_types": [
            "conv",
            "full_attention",
            "sliding_attention"
          ],
          "sliding_window": 4,
          "conv_L_cache": 7,
          "conv_bias": true,
          "norm_eps": 1e-05,
          "use_sandwich_norm": true,
          "tie_word_embeddings": true,
          "pad_token_id": 0,
          "bos_token_id": 1,
          "eos_token_id": 2,
          "rope_parameters": {
            "full_attention": {
              "rope_type": "default",
              "rope_theta": 1000000.0,
              "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
              "rope_type": "default",
              "rope_theta": 10000.0
            }
          }
        }
        """

    /// The context lengths of the check.
    private static let contextLengths = [4096, 16384, 32768]

    /// The prefill chunk sizes whose results must agree.
    private static let chunkSizes = [257, 512]

    /// The largest final-logit difference of the two chunkings with IEEE float32 matmuls.
    private static let float32Tolerance: Float = 1e-3

    /// The largest final-logit difference of the two chunkings with TF32 matmuls (neural
    /// accelerators), which raise the noise of splitting a forward to a few times 1e-3.
    private static let reducedFloat32Tolerance: Float = 2e-2

    /// Whether the GPU computes float32 matmuls in a reduced precision (TF32).
    private var usesReducedFloat32: Bool {
        let a = withRandomState(MLXRandom.RandomState(seed: 0)) { MLXRandom.normal([64, 512]) }
        let b = withRandomState(MLXRandom.RandomState(seed: 1)) { MLXRandom.normal([512, 64]) }
        let cpu = matmul(a, b, stream: .cpu)
        let gpu = matmul(a, b, stream: .gpu)
        return abs(gpu - cpu).max().item(Float.self) / abs(cpu).max().item(Float.self) > 1e-5
    }

    /// The tiny LFM3 model with the deterministic weights of `LFM3ReferenceTests`.
    private func makeModel() throws -> LFM3Model {
        let config = try JSONDecoder().decode(
            LFM3Configuration.self, from: Data(Self.configuration.utf8))
        let model = LFM3Model(config)
        var weights = [String: MLXArray]()
        for (i, entry) in model.parameters().flattened().sorted(by: { $0.0 < $1.0 }).enumerated() {
            let (name, parameter) = entry
            let shift: Float = parameter.ndim == 1 && parameter.size == config.hiddenSize ? 1 : 0
            let values = (0 ..< parameter.size).map { Float(($0 + i * 3) % 17 - 8) / 32 + shift }
            weights[name] = MLXArray(values, parameter.shape)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    func testLongContextChunking() throws {
        let model = try makeModel()
        let config = model.configuration
        let tolerance =
            usesReducedFloat32 ? Self.reducedFloat32Tolerance : Self.float32Tolerance
        for length in Self.contextLengths {
            let tokens = MLXArray((0 ..< length).map { Int32(($0 * 7 + 3) % 16) }).reshaped(
                1, length)
            var results = [MLXArray]()
            for chunkSize in Self.chunkSizes {
                let cache = try model.newCache(parameters: nil)
                for start in stride(from: 0, to: length, by: chunkSize) {
                    let hidden = model.hiddenStates(
                        tokens[0..., start ..< min(start + chunkSize, length)], cache: cache)
                    eval(hidden)
                }
                let output = model(MLXArray([Int32(1)]).reshaped(1, 1), cache: cache)
                eval(output)
                results.append(output)
                for (i, layerCache) in cache.enumerated() {
                    if config.layerTypes[i] == .conv {
                        XCTAssertEqual(
                            (layerCache as? MambaCache)?[0]?.dim(1), config.convLCache - 1)
                    } else {
                        XCTAssertEqual(layerCache.offset, length + 1)
                        if config.layerTypes[i] == .slidingAttention {
                            XCTAssertLessThanOrEqual(
                                layerCache.state[0].dim(2), config.slidingWindow)
                        }
                    }
                }
            }
            let error = abs(results[0] - results[1]).max().item(Float.self)
            print("LFM3 context \(length): final-logit max_abs=\(error)")
            XCTAssertLessThan(error, tolerance)
            XCTAssertTrue(all(isFinite(results[0])).item(Bool.self))
        }
    }
}
