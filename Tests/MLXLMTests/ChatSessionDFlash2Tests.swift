// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A chat session with a DFlash2 drafter generates what a plain session
/// generates, and reuses its prompt cache the same way across turns and forks.
///
/// The target is a small hybrid Qwen 3.8 model, and the drafter is a small
/// DFlash2 model with random weights. Random drafts are mostly rejected, which
/// tests the rewind of every rejected tail.
@Suite(.serialized)
struct ChatSessionDFlash2Tests {

    private typealias TemplateToken = ChatSessionRecurrentCheckpointTests.TemplateToken

    private static let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
    private static let instructions = "answer briefly"
    private static let turns = ["first", "second", "third"]

    private static func target() throws -> Qwen35TextModel {
        let json = """
            {
                "model_type": "qwen3_8_text",
                "hidden_size": 64,
                "num_hidden_layers": 5,
                "layer_types": ["linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "full_attention"],
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "vocab_size": \(TemplateToken.vocabularySize),
                "num_experts": 0,
                "num_experts_per_tok": 0
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        MLXRandom.seed(3)
        let model = Qwen35TextModel(configuration)
        eval(model)
        return model
    }

    private static func drafter() throws -> DFlash2DrafterContainer {
        let vocab = TemplateToken.vocabularySize
        let json = """
            {
              "architectures": ["DFlash2DraftModel"],
              "is_causal": false,
              "dflash_config": {
                "block_size": 5, "conv_group_size": 16, "conv_kernel_size": 2,
                "mask_token_id": \(vocab - 2), "selector_rank": 8, "selector_top_k": 4,
                "target_layer_ids": [1, 3]
              },
              "head_dim": 16, "hidden_size": 64, "intermediate_size": 96,
              "layer_types": ["sliding_attention", "sliding_attention"],
              "max_position_embeddings": 1024, "model_type": "qwen3",
              "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
              "num_target_layers": 5, "rms_norm_eps": 1e-06,
              "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
              "sliding_window": 64, "vocab_size": \(vocab)
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            DFlash2Configuration.self, from: Data(json.utf8))
        MLXRandom.seed(5)
        let model = DFlash2DraftModel(configuration)
        eval(model)
        return DFlash2DrafterContainer(
            context: DFlash2DrafterContext(
                configuration: ModelConfiguration(id: "dflash2-test"), model: model))
    }

    private static func session(drafter: DFlash2DrafterContainer?) throws -> ChatSession {
        let tokenizer = ChatSessionRecurrentCheckpointTests.CheckpointTemplateTokenizer(
            historyShape: .closesTheThinkBlock)
        let modelConfiguration = ModelConfiguration(id: "dflash2-session-test")
        let context = ModelContext(
            configuration: modelConfiguration, model: try target(),
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: modelConfiguration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        let session = ChatSession(
            context, instructions: instructions, generateParameters: parameters)
        session.dflash2Drafter = drafter
        return session
    }

    private struct Turn: Equatable {
        var text = ""
        var cachedPromptTokenCount = 0
        var promptTokenCount = 0
        var proposedDraftTokens = 0
    }

    private static func respond(_ session: ChatSession, to prompt: String) async throws -> Turn {
        var turn = Turn()
        for try await item in session.streamDetails(to: prompt) {
            turn.text += item.chunk ?? ""
            if let info = item.info {
                turn.cachedPromptTokenCount = info.cachedPromptTokenCount
                turn.promptTokenCount = info.promptTokenCount
                turn.proposedDraftTokens = info.proposedDraftTokens ?? 0
            }
        }
        return turn
    }

    /// Three turns, then a fork with an empty history.
    private static func conversation(drafter: DFlash2DrafterContainer?) async throws -> [Turn] {
        let session = try session(drafter: drafter)
        var result: [Turn] = []
        for prompt in turns {
            result.append(try await respond(session, to: prompt))
        }
        result.append(try await respond(await session.fork(history: []), to: turns[1]))
        return result
    }

    @Test("a session with a drafter writes what a plain session writes, over turns and a fork")
    func aSessionWithADrafterWritesWhatAPlainSessionWrites() async throws {
        let plain = try await Self.conversation(drafter: nil)
        let drafted = try await Self.conversation(drafter: try Self.drafter())

        #expect(drafted.map(\.text) == plain.map(\.text))
        #expect(drafted.allSatisfy { $0.proposedDraftTokens > 0 })
    }

    @Test("a session with a drafter reuses its prompt cache as a plain session does")
    func aSessionWithADrafterReusesItsPromptCacheAsAPlainSessionDoes() async throws {
        let plain = try await Self.conversation(drafter: nil)
        let drafted = try await Self.conversation(drafter: try Self.drafter())

        #expect(drafted.map(\.cachedPromptTokenCount) == plain.map(\.cachedPromptTokenCount))
        #expect(drafted.map(\.promptTokenCount) == plain.map(\.promptTokenCount))
        #expect(drafted.dropFirst().allSatisfy { $0.cachedPromptTokenCount > 0 })
    }
}
