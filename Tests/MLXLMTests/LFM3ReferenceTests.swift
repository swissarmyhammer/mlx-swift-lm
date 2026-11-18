import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class LFM3ReferenceTests: XCTestCase {
    private func makeModel() throws -> LFM3Model {
        let config = try JSONDecoder().decode(
            LFM3Configuration.self, from: Data(LFM3ReferenceValues.configuration.utf8))
        let model = LFM3Model(config)
        var weights = [String: MLXArray]()
        // Match the Python oracle independently of module traversal order.
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

    func testOriginalHuggingFaceFloat32Parity() throws {
        // CPU pins IEEE FP32, including on devices that use TF32 for GPU matmuls.
        try Device.withDefaultDevice(.cpu) {
            let model = try makeModel()
            let config = model.configuration
            let tokens = MLXArray((0 ..< 38).map { Int32(($0 * 7 + 3) % 16) }).reshaped(2, 19)

            func compare(_ actual: MLXArray, _ key: String) throws {
                let values = try XCTUnwrap(LFM3ReferenceValues.outputs[key], key)
                XCTAssertEqual(actual.size, values.count, key)
                let expected = MLXArray(values, actual.shape)
                let maximum = abs(actual - expected).max().item(Float.self)
                print("LFM3 FP32 \(key): max_abs=\(maximum)")
                XCTAssertLessThan(maximum, 2e-5, key)
            }

            var h = model.model.embed(tokens)
            try compare(h, "model.embed_tokens")
            for (i, layer) in model.model.layers.enumerated() {
                let mask = makeAttentionMask(
                    n: tokens.dim(1), cache: nil,
                    windowSize: layer.layerType == .slidingAttention ? config.slidingWindow : nil)
                let normalized = layer.operatorNorm(h)
                if let conv = layer.conv {
                    try compare(conv(normalized, cache: nil), "model.layers.\(i).conv")
                } else if let attention = layer.attention {
                    try compare(
                        attention(normalized, mask: mask, cache: nil), "model.layers.\(i).self_attn"
                    )
                }
                h = layer(h, mask: mask, cache: nil)
                try compare(h, "model.layers.\(i)")
            }
            let hidden = model.hiddenStates(tokens, cache: nil)
            try compare(hidden, "model.embedding_norm")
            let logits = model.projectLogits(hidden)
            try compare(logits, "logits")
            XCTAssertEqual(
                argMax(logits, axis: -1).asArray(Int.self),
                argMax(
                    MLXArray(try XCTUnwrap(LFM3ReferenceValues.outputs["logits"]), logits.shape),
                    axis: -1
                ).asArray(Int.self))

            let cache = try model.newCache(parameters: nil)
            var outputs = [MLXArray]()
            for (start, end) in [(0, 2), (2, 7), (7, 18), (18, 19)] {
                outputs.append(model(tokens[0..., start ..< end], cache: cache))
                eval(outputs.last!)
            }
            try compare(concatenated(outputs, axis: 1), "logits")
        }
    }
}
