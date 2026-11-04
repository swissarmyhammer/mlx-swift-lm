import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A `ChatSession` with chained Qwen MTP rounds keeps its prompt cache across turns: each
/// follow-up prefills only the new message, an edit goes back to a checkpoint of the cached
/// prefix, the drafter resumes its continuation rather than starting again, and every turn writes
/// the text of a session without speculation. Each test runs at the shipped default and at every
/// chained block size.
@Suite(.serialized)
struct ChatSessionQwenChainedMTPTests {

    /// The tokens each turn generates: several rounds of the largest block size.
    private static let turnTokenCount = 12

    /// The instructions of each chat.
    private static let instructions = "answer briefly"

    /// The user messages of a chat.
    private static let prompts = ["first", "second", "third"]

    /// The user message an edit asks in place of the second message.
    private static let editPrompt = "edited"

    /// The messages one turn adds to a history: the user message and the reply.
    private static let messagesOfOneTurn = 2

    /// The tokens a follow-up turn of an MTP session prefills beyond what a plain session does:
    /// the final token of the previous reply, which no forward pass of the MTP stream fed.
    private static let unfedFinalTokenCount = 1

    /// The shipped default, then every block size that chains drafts after the seed without a
    /// floor, then a chained block size with a floor.
    static let settings =
        [
            Qwen35ChainedMTPDraftTests.ChainSetting(
                blockSize: SpeculativeDecodingConfig.defaultMTPBlockSize,
                minimumDraftProbability: SpeculativeDecodingConfig
                    .defaultMTPMinimumDraftProbability)
        ]
        + Qwen35ChainedMTPDraftTests.chainedBlockSizes.map {
            Qwen35ChainedMTPDraftTests.ChainSetting(blockSize: $0, minimumDraftProbability: 0)
        }
        + [
            Qwen35ChainedMTPDraftTests.ChainSetting(
                blockSize: Qwen35ChainedMTPDraftTests.chainedBlockSizes[1],
                minimumDraftProbability: Qwen35ChainedMTPDraftTests.reportedFloor)
        ]

    /// A greedy session over the fixture target with ``instructions``, whose template writes a
    /// Qwen-like transcript and takes checkpoints, with chained MTP rounds of `setting` or
    /// without speculation.
    private func session(
        _ fixture: Qwen35MTPFixture, setting: Qwen35ChainedMTPDraftTests.ChainSetting?
    ) throws -> ChatSession {
        let tokenizer = ChatSessionRecurrentCheckpointTests.CheckpointTemplateTokenizer(
            historyShape: .closesTheThinkBlock)
        let context = ModelContext(
            configuration: fixture.modelConfiguration, model: fixture.target,
            processor: TestInputProcessor(
                tokenizer: tokenizer, configuration: fixture.modelConfiguration,
                messageGenerator: DefaultMessageGenerator()),
            tokenizer: tokenizer)
        return ChatSession(
            context, instructions: Self.instructions,
            speculativeDecoding: try setting.map {
                try SpeculativeDecodingConfig(
                    mtpDrafter: MTPDrafterContainer(
                        context: MTPDrafterContext(
                            configuration: fixture.modelConfiguration, model: fixture.drafter)),
                    blockSize: $0.blockSize, minimumDraftProbability: $0.minimumDraftProbability)
            },
            generateParameters: GenerateParameters(
                maxTokens: Self.turnTokenCount, temperature: 0))
    }

    /// Expects the turn of the speculative session to write the plain turn's text over the same
    /// prompt, from the cached prefix of the plain turn but its unfed final token, and to
    /// speculate throughout.
    private func expect(
        _ speculative: (text: String, info: GenerateCompletionInfo),
        matches plain: (text: String, info: GenerateCompletionInfo), turn: Int
    ) {
        #expect(speculative.text == plain.text, "turn \(turn)")
        #expect(
            speculative.info.cachedPromptTokenCount + speculative.info.promptTokenCount
                == plain.info.cachedPromptTokenCount + plain.info.promptTokenCount,
            "turn \(turn)")
        #expect(
            speculative.info.promptTokenCount
                <= plain.info.promptTokenCount + Self.unfedFinalTokenCount,
            "turn \(turn)")
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0, "turn \(turn)")
        #expect(speculative.info.passthroughReason == nil, "turn \(turn)")
        #expect(speculative.info.speculativeDecodingFallbackReason == nil, "turn \(turn)")
    }

    @Test(
        "each follow-up turn of a chained chat prefills only its message and writes the plain text",
        arguments: settings)
    func followUpTurnsReuseThePromptCache(
        setting: Qwen35ChainedMTPDraftTests.ChainSetting
    ) async throws {
        let fixture = try Qwen35MTPFixture()
        let speculative = try session(fixture, setting: setting)
        let plain = try session(fixture, setting: nil)

        for (turn, prompt) in Self.prompts.enumerated() {
            let speculativeTurn = try await collectChatTurn(speculative.streamDetails(to: prompt))
            let plainTurn = try await collectChatTurn(plain.streamDetails(to: prompt))

            expect(speculativeTurn, matches: plainTurn, turn: turn)
            #expect(turn == 0 || speculativeTurn.info.cachedPromptTokenCount > 0, "turn \(turn)")
        }
        // The drafter resumed its continuation on each follow-up turn instead of starting again
        // at the cached position: its entries still begin at the start of the chat.
        let continuation = try #require(await speculative.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == 0)
    }

    @Test(
        "an edit of a chained chat goes back to the cached prefix and writes the plain text",
        arguments: settings)
    func editResumesTheCachedPrefix(setting: Qwen35ChainedMTPDraftTests.ChainSetting) async throws {
        let fixture = try Qwen35MTPFixture()
        let speculative = try session(fixture, setting: setting)
        let plain = try session(fixture, setting: nil)
        var history: [Chat.Message] = []
        for prompt in Self.prompts.prefix(Self.messagesOfOneTurn) {
            let reply = try await collectChatTurn(speculative.streamDetails(to: prompt))
            _ = try await collectChatTurn(plain.streamDetails(to: prompt))
            history += [.user(prompt), .assistant(reply.text)]
        }
        let keptHistory = Array(history.prefix(Self.messagesOfOneTurn))

        let edit = try await collectChatTurn(
            speculative.fork(history: keptHistory).streamDetails(to: Self.editPrompt))
        let expected = try await collectChatTurn(
            plain.fork(history: keptHistory).streamDetails(to: Self.editPrompt))

        expect(edit, matches: expected, turn: Self.prompts.count)
        #expect(edit.info.cachedPromptTokenCount > 0)
    }
}
