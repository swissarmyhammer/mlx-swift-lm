// Copyright © 2026 Apple Inc.
//
// A profile of the prompt checkpoint of upstream #716 (`TokenIterator(checkpointAt:)` and
// `KVCacheStorage.Checkpoint`) on small random Mellum and Qwen 3.5 models at 512 and 2048
// prompt tokens: a cold prefill, a restore plus the suffix, a split prefill against an
// unsplit one, and an append after each.
//
// The profile holds nothing of the times. It came from `MLXLMTests`
// (`PromptCheckpointNumericsTests`), where the environment variable
// `MLX_RUN_CHECKPOINT_BENCHMARKS` gated it; here it runs in each run of the
// `IntegrationTesting` scheme. Each measurement line starts with `[CHECKPOINT]`.
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`. Use
// `xcodebuild build-for-testing -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
// as the compile evidence for any change to it.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class PromptCheckpointBenchmarkProfileTests: XCTestCase {
    private final class CountedModel: Module, LLMModel {
        let base: any LanguageModel
        var forwards = 0
        var loraLayers: [Module] { [] }

        init(_ base: any LanguageModel) {
            self.base = base
        }

        func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try base.newCache(parameters: parameters)
        }

        func callAsFunction(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            forwards += 1
            return base(input, cache: cache, state: state)
        }

        func nextTokenLogits(
            _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
        ) -> LMOutput {
            forwards += 1
            return base.nextTokenLogits(input, cache: cache, state: state)
        }
    }

    private func mellum() throws -> any LanguageModel {
        let json = #"""
            {
                "model_type": "mellum", "hidden_size": 32, "num_hidden_layers": 3,
                "intermediate_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "num_experts": 4, "num_experts_per_tok": 2,
                "moe_intermediate_size": 32, "norm_topk_prob": true, "rms_norm_eps": 1e-6,
                "vocab_size": 128, "max_position_embeddings": 16384, "sliding_window": 32,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
                "mlp_layer_types": ["sparse", "dense", "sparse"],
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                }
            }
            """#
        return MellumModel(
            try JSONDecoder().decode(MellumConfiguration.self, from: Data(json.utf8)))
    }

    private func qwen() throws -> any LanguageModel {
        let json = #"""
            {
                "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 2,
                "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 32, "vocab_size": 128, "full_attention_interval": 2,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "max_position_embeddings": 16384,
                "rope_parameters": {"mrope_section": [8,4,4], "rope_theta": 100000,
                    "partial_rotary_factor": 1.0, "type": "default"}
            }
            """#
        return Qwen35Model(
            try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8)))
    }

    private func tokens(_ count: Int, seed: Int = 3) -> [Int] {
        (0 ..< count).map { ($0 * 7 + seed) % 128 }
    }

    private func storage(_ model: any LanguageModel) throws -> KVCacheStorage {
        KVCacheStorage(try model.newCache(parameters: nil), plan: .disabled)
    }

    func testCheckpointBenchmark() throws {
        for (name, base) in [("Mellum", try mellum()), ("Qwen35", try qwen())] {
            quantize(model: base, groupSize: 32, bits: 4)
            base.apply { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 }
            eval(base)
            let model = CountedModel(base)
            for count in [512, 2048] {
                let prefix = tokens(count)
                let suffix = tokens(33, seed: 11)
                let live = try storage(model)
                let parameters = GenerateParameters(temperature: 0)
                func run(
                    _ input: [Int], cache: KVCacheStorage, checkpointAt position: Int? = nil,
                    state: LMOutput.State? = nil
                )
                    throws -> TokenIterator
                {
                    let iterator = try TokenIterator(
                        input: LMInput(tokens: MLXArray(input)), model: model, cacheStorage: cache,
                        state: state, parameters: parameters, checkpointAt: position)
                    _ = iterator.pendingInput.tokens.item(Int.self)
                    return iterator
                }
                let iterator = try run(prefix + [12], cache: live, checkpointAt: count)
                let checkpoint = try XCTUnwrap(iterator.prefillCheckpoint)
                _ = try run(tokens(64, seed: 19), cache: live)
                var cold: [Double] = []
                var warm: [Double] = []
                var split: [Double] = []
                var unsplit: [Double] = []
                var appendSplit: [Double] = []
                var appendUnsplit: [Double] = []
                var splitForwards = 0
                var unsplitForwards = 0
                var appendSplitForwards = 0
                var appendUnsplitForwards = 0
                for _ in 0 ..< 4 {
                    let fresh = try storage(model)
                    let coldStart = Date.timeIntervalSinceReferenceDate
                    _ = try run(prefix + suffix, cache: fresh)
                    cold.append(Date.timeIntervalSinceReferenceDate - coldStart)
                    let warmStart = Date.timeIntervalSinceReferenceDate
                    XCTAssertTrue(live.restore(checkpoint.cache))
                    _ = try run(suffix, cache: live, state: checkpoint.state)
                    warm.append(Date.timeIntervalSinceReferenceDate - warmStart)
                    let splitCache = try storage(model)
                    model.forwards = 0
                    let splitStart = Date.timeIntervalSinceReferenceDate
                    let splitIterator = try run(
                        prefix + [12], cache: splitCache, checkpointAt: count)
                    split.append(Date.timeIntervalSinceReferenceDate - splitStart)
                    splitForwards = model.forwards
                    let unsplitCache = try storage(model)
                    model.forwards = 0
                    let unsplitStart = Date.timeIntervalSinceReferenceDate
                    let unsplitIterator = try run(prefix + [12], cache: unsplitCache)
                    unsplit.append(Date.timeIntervalSinceReferenceDate - unsplitStart)
                    unsplitForwards = model.forwards

                    // Both prompts extend the live cache. Neither restores its checkpoint.
                    model.forwards = 0
                    let appendSplitStart = Date.timeIntervalSinceReferenceDate
                    _ = try withExtendedLifetime(splitIterator.prefillCheckpoint) {
                        try run(suffix, cache: splitCache, state: splitIterator.state)
                    }
                    appendSplit.append(Date.timeIntervalSinceReferenceDate - appendSplitStart)
                    appendSplitForwards = model.forwards
                    model.forwards = 0
                    let appendUnsplitStart = Date.timeIntervalSinceReferenceDate
                    _ = try run(suffix, cache: unsplitCache, state: unsplitIterator.state)
                    appendUnsplit.append(Date.timeIntervalSinceReferenceDate - appendUnsplitStart)
                    appendUnsplitForwards = model.forwards
                    XCTAssertEqual(appendSplitForwards, appendUnsplitForwards)
                }
                func median(_ samples: [Double]) -> Double {
                    samples.dropFirst().sorted()[1] * 1000
                }
                print(
                    String(
                        format:
                            "[CHECKPOINT] %@ prefix=%d cold=%.3fms restore=%.3fms split=%.3fms unsplit=%.3fms saved=%d bytes",
                        name, count, median(cold), median(warm), median(split), median(unsplit),
                        checkpoint.cache.retainedBytes))
                print(
                    "[CHECKPOINT] \(name) prefix=\(count) forwards=\(splitForwards)/\(unsplitForwards) "
                        + "append=\(median(appendSplit))/\(median(appendUnsplit))ms "
                        + "append_forwards=\(appendSplitForwards)/\(appendUnsplitForwards)")
            }
        }
    }
}
