// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4 has no chat template, and `DeepSeekV4ChatEncoder` is the only
// correct prompt builder for that model family (card ^f0ymw6b, decision B).
// The generic prompt path used to fall back to a plain-text prompt when a
// tokenizer has no chat template. For DeepSeek-V4 that fallback is a wrong
// prompt with no error. These tests pin the refusal: the prompt path must
// throw for DeepSeek-V4, and it must keep the fallback for models that
// permit it.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4PromptFallbackTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - Fixtures

    /// A tokenizer with no chat template. `applyChatTemplate` throws
    /// `TokenizerError.missingChatTemplate`, and `encode` maps each UTF-8
    /// byte of the text to one token id.
    private struct NoTemplateTokenizer: MLXLMCommon.Tokenizer {
        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map { Int($0) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map { UInt8(clamping: $0) }, as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    /// The `config.json` of a small synthetic DeepSeek-V4 checkpoint. The
    /// model only has to exist — no forward pass runs in this suite.
    private static func configuration() throws -> DeepSeekV4Configuration {
        let json = """
            {
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
        return try JSONDecoder().decode(DeepSeekV4Configuration.self, from: Data(json.utf8))
    }

    /// Builds the prompt processor under test with the given refusal.
    private func makeProcessor(missingChatTemplateRefusal: String?) -> LLMUserInputProcessor {
        LLMUserInputProcessor(
            tokenizer: NoTemplateTokenizer(),
            configuration: ModelConfiguration(id: "test/deepseek-v4-synthetic"),
            messageGenerator: DefaultMessageGenerator(),
            missingChatTemplateRefusal: missingChatTemplateRefusal)
    }

    // MARK: - The model forbids the fallback

    @Test("DeepSeekV4Model forbids the plain-text prompt fallback")
    func deepSeekV4ForbidsThePlainTextPromptFallback() throws {
        let model = DeepSeekV4Model(try Self.configuration())

        let refusal = try #require(model.missingChatTemplateRefusal)
        #expect(refusal.contains("DeepSeekV4ChatEncoder"))
    }

    // MARK: - The processor obeys the refusal

    @Test("the prompt path throws the refusal instead of a plain-text prompt")
    func promptPathThrowsTheRefusal() throws {
        let refusal = "DeepSeek-V4 has no chat template."
        let processor = makeProcessor(missingChatTemplateRefusal: refusal)

        do {
            _ = try processor.prepare(input: UserInput(prompt: "Hello"))
            Issue.record("expected PromptPreparationError.plainTextFallbackForbidden")
        } catch let error as PromptPreparationError {
            #expect(error.errorDescription == refusal)
        }
    }

    @Test("the prompt path keeps the plain-text fallback when the model permits it")
    func promptPathKeepsTheFallbackWhenPermitted() throws {
        let processor = makeProcessor(missingChatTemplateRefusal: nil)

        let input = try processor.prepare(input: UserInput(prompt: "Hello"))

        let tokens = input.text.tokens.asArray(Int.self)
        let text = NoTemplateTokenizer().decode(tokenIds: tokens, skipSpecialTokens: false)
        #expect(text.contains("Hello"))
    }
}
