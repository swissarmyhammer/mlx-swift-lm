// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXGuidedGeneration
@testable import MLXLMCommon

/// Guided generation on an Inkling 2 target with its DSpark drafter: each draft takes the grammar
/// mask, the target verifies the round in place, and the output is the plain output.
///
/// The fixtures are a tiny Inkling 2 target and drafter with seeded random weights, over the
/// 64 tokens of ``ToolCallAlphabetTokenizer``. A real model's calls are
/// `Inkling2FoundationModelsMTPTests` in the integration project.
@Suite(.serialized)
struct GuidedGenerationInklingMTPTests {

    /// The tiny Inkling 2 target of `Inkling2ModelTests` over the vocabulary of
    /// ``ToolCallAlphabetTokenizer``, with the DSpark head of `Inkling2DraftTests`: two draft
    /// layers, a window of six, and blocks of four.
    private static let json = """
        {
          "model_type": "inkling2_loop",
          "text_config": {
            "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 64,
            "unpadded_vocab_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "sliding_window_size": 8, "local_layer_ids": [0, 1, 2, 4],
            "d_rel": 8, "rel_extent": 16, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
            "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2, "n_routed_experts": 4,
            "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 2.0, "final_logit_softcapping": 30.0,
            "loop_config": {
              "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
              "is_varloop": false
            }
          },
          "mtp_config": {
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
            "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "local_layer_ids": [0],
            "sliding_window_size": 6, "block_size": 4, "markov_rank": 8,
            "markov_head_type": "rnn_hidden", "q_bias": false, "o_bias": false,
            "target_layer_ids": [1, 3, 5, 8]
          }
        }
        """

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The block sizes of the generations: one draft, and chains of three drafts.
    private static let blockSizes = [2, 4]

    /// A prefill step shorter than ``prompt``, whose chunks feed the drafter.
    private static let shortPrefillStep = 3

    /// The prefills of the generations: the whole prompt in one forward, and in chunks.
    private static let prefillSteps: [Int?] = [nil, shortPrefillStep]

    /// Every schema of the fixtures speculates to the plain output, with the whole
    /// prompt in one forward and in chunks, and reports no fallback.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Self.blockSizes, Self.prefillSteps)
    func aGuidedGenerationSpeculatesToThePlainOutput(blockSize: Int, prefillStep: Int?) throws {
        let fixture = try Qwen35MTPFixture()
        let target = try Inkling2TinyModels.target(json: Self.json)
        let drafter = try Inkling2TinyModels.drafter(json: Self.json)
        let prefill = PrefillParameters(stepSize: prefillStep)
        for schema in GuidedFixtureSchema.allCases {
            let plain = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil,
                target: target, prefill: prefill)
            let speculative = try GuidedFixtureRun.generate(
                schema: schema.json, prompt: Self.prompt, fixture: fixture,
                blockSize: blockSize, drafter: drafter, target: target, prefill: prefill)

            #expect(!plain.text.isEmpty, "\(schema)")
            #expect(speculative.text == plain.text, "\(schema)")
            #expect(speculative.tokenCount == plain.tokenCount, "\(schema)")
            #expect(speculative.incomplete == plain.incomplete, "\(schema)")
            #expect(speculative.sink.draftedTokenCount > 0, "\(schema)")
            #expect(speculative.sink.speculationFallbacks.isEmpty, "\(schema)")
        }
    }

    /// The guided drafter of an Inkling 2 speculation is the DSpark head.
    @Test
    func theDSparkHeadDraftsUnderAGrammar() throws {
        let target = try Inkling2TinyModels.target(json: Self.json)
        let speculation = MTPSpeculation(
            drafter: try Inkling2TinyModels.drafter(json: Self.json),
            blockSize: Self.blockSizes.last ?? 2, minimumDraftProbability: 0)
        let guidedTarget = try GuidedTarget(model: target, parameters: GenerateParameters())

        #expect(GuidedSpeculation.drafter(of: speculation, for: guidedTarget) is Inkling2DraftModel)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
