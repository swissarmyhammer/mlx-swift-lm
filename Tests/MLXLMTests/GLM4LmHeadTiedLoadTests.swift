// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

// MARK: - Regression: unconditional lm_head crashes on genuinely tied checkpoints
//
// `GLM4Model` and `GLM4MoELiteModel` both declared `lm_head` as a non-optional
// `Linear`, created unconditionally in `init`, unlike the rest of this
// codebase's convention (documented in
// `Libraries/MLXLMCommon/Documentation.docc/porting.md`, and correctly
// implemented by the sibling `GLM4MoEModel` in `GLM4MOE.swift`): `lmHead:
// Linear?`, created only `if !tieWordEmbeddings`, with `callAsFunction`
// falling back to `model.embedTokens.asLinear(out)` when nil.
//
// Because `MLXLMCommon.Load.loadWeights` calls
// `model.update(parameters:, verify: [.all])`, which requires every module
// parameter path to have a matching value, a genuinely tied checkpoint --
// which the standard HF/mlx_lm convention omits `lm_head.weight` from
// entirely -- crashed with `UpdateError.keyNotFound` for both architectures.
//
// These tests pin the guarded-`Optional` construction/fallback pattern (and,
// for `GLM4MoELiteModel`, wires the previously-inert `tieWordEmbeddings` into
// `sanitize`) so the two can't drift from the correct pattern again.

struct GLM4LmHeadTiedLoadTests {
    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - GLM4Model

    /// End-to-end reproduction of the production load failure: build a
    /// checkpoint from a "dense" (untied) sibling, drop `lm_head.weight` the
    /// way a genuinely tied HF/mlx_lm checkpoint omits it, then load it into
    /// a tied model through the exact call the production loader uses.
    /// Before the fix this threw `keyNotFound(… lm_head … weight …)`.
    @Test("Tied GLM4 checkpoint (omitting lm_head.weight) loads without keyNotFound")
    func glm4TiedCheckpointLoadsWithoutKeyNotFound() throws {
        let dense = GLM4Model(Self.glm4Config(tieWordEmbeddings: false))
        eval(dense)

        var checkpoint = [String: MLXArray]()
        for (key, value) in dense.parameters().flattened() {
            if key == "lm_head.weight" { continue }  // omitted by a genuinely tied checkpoint
            checkpoint[key] = value
        }

        let model = GLM4Model(Self.glm4Config(tieWordEmbeddings: true))
        // `[.all]` is what `MLXLMCommon.loadWeights` applies -- it throws on
        // any module parameter that has no matching weight (the original bug).
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
    }

    /// An untied checkpoint must still load and the separate `lm_head` must
    /// actually be used: after loading the same weights a "dense" sibling was
    /// built from, the two models must produce identical logits.
    @Test("Untied GLM4 checkpoint loads and uses its own lm_head")
    func glm4UntiedCheckpointUsesSeparateLmHead() throws {
        let dense = GLM4Model(Self.glm4Config(tieWordEmbeddings: false))
        eval(dense)
        let checkpoint = Dictionary(uniqueKeysWithValues: dense.parameters().flattened())

        let model = GLM4Model(Self.glm4Config(tieWordEmbeddings: false))
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)

        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let denseOutput = dense(tokens, cache: nil)
        let modelOutput = model(tokens, cache: nil)
        eval(denseOutput, modelOutput)
        #expect(allClose(denseOutput, modelOutput).item(Bool.self))
    }

    // MARK: - GLM4MoELiteModel

    @Test("Tied GLM4MoELite checkpoint (omitting lm_head.weight) loads without keyNotFound")
    func glm4MoELiteTiedCheckpointLoadsWithoutKeyNotFound() throws {
        let dense = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: false))
        eval(dense)

        var checkpoint = [String: MLXArray]()
        for (key, value) in dense.parameters().flattened() {
            if key == "lm_head.weight" { continue }
            checkpoint[key] = value
        }

        let model = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: true))
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)
    }

    @Test("Untied GLM4MoELite checkpoint loads and uses its own lm_head")
    func glm4MoELiteUntiedCheckpointUsesSeparateLmHead() throws {
        let dense = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: false))
        eval(dense)
        let checkpoint = Dictionary(uniqueKeysWithValues: dense.parameters().flattened())

        let model = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: false))
        try model.update(
            parameters: ModuleParameters.unflattened(checkpoint), verify: [.all])
        eval(model)

        let tokens = MLXArray([1, 2, 3]).reshaped([1, 3])
        let denseOutput = dense(tokens, cache: nil)
        let modelOutput = model(tokens, cache: nil)
        eval(denseOutput, modelOutput)
        #expect(allClose(denseOutput, modelOutput).item(Bool.self))
    }

    /// `tieWordEmbeddings` was previously decoded but never referenced --
    /// `sanitize` must now strip `lm_head.weight` when the config says the
    /// checkpoint's embeddings are tied, mirroring `GLM4MoEModel.sanitize`.
    @Test("GLM4MoELite sanitize drops lm_head.weight when tied")
    func glm4MoELiteSanitizeDropsLmHeadWhenTied() {
        let model = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: true))
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([12, 8]),
            "lm_head.weight": MLXArray.zeros([12, 8]),
        ]
        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    @Test("GLM4MoELite sanitize keeps lm_head.weight when not tied")
    func glm4MoELiteSanitizeKeepsLmHeadWhenNotTied() {
        let model = GLM4MoELiteModel(Self.glm4MoELiteConfig(tieWordEmbeddings: false))
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([12, 8]),
            "lm_head.weight": MLXArray.zeros([12, 8]),
        ]
        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["lm_head.weight"] != nil)
    }

    // MARK: - Helpers

    private static func glm4Config(tieWordEmbeddings: Bool) -> GLM4Configuration {
        let json = """
            {
              "model_type": "glm4",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "attention_bias": false,
              "head_dim": 4,
              "rms_norm_eps": 1e-6,
              "vocab_size": 12,
              "num_key_value_heads": 1,
              "partial_rotary_factor": 1.0,
              "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        return try! JSONDecoder().decode(GLM4Configuration.self, from: Data(json.utf8))
    }

    private static func glm4MoELiteConfig(tieWordEmbeddings: Bool) -> GLM4MoELiteConfiguration {
        let json = """
            {
              "model_type": "glm4_moe_lite",
              "vocab_size": 12,
              "hidden_size": 8,
              "intermediate_size": 16,
              "moe_intermediate_size": 16,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "routed_scaling_factor": 1.0,
              "kv_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "qk_nope_head_dim": 2,
              "v_head_dim": 4,
              "norm_topk_prob": true,
              "n_group": 1,
              "topk_group": 1,
              "num_experts_per_tok": 1,
              "first_k_dense_replace": 0,
              "max_position_embeddings": 32,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000.0,
              "attention_bias": false,
              "partial_rotary_factor": 1.0,
              "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        return try! JSONDecoder().decode(
            GLM4MoELiteConfiguration.self, from: Data(json.utf8))
    }
}
