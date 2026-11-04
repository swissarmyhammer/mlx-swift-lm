// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXNN
import Testing

@Suite(.serialized)
struct ChatSessionMTPEarlyTerminationTests {
    @Test("early EOS finalizes MTP lookahead before the next turn")
    func earlyEOSFinalizesLookaheadBeforeNextTurn() async throws {
        let session = ChatSession(
            makeModelContext(),
            speculativeDecoding: try makeMTPConfiguration(),
            generateParameters: .init(maxTokens: 4, temperature: 0)
        )

        let first = try await collect(session.streamDetails(to: "a"))

        #expect(first.text == "x")
        #expect(first.info.stopReason == .stop)
        #expect(first.info.generationTokenCount == 1)
        #expect(first.info.proposedDraftTokens == 2)
        #expect(first.info.acceptedDraftTokens == 2)

        await session.synchronize()

        let stoppedStatus = try await session.cacheStatus()
        #expect(stoppedStatus.phase == .realized)
        // One prompt token plus the emitted bonus and terminating EOS are
        // represented. The other accepted EOS draft was lookahead and must
        // have been removed by generation finalization.
        #expect(stoppedStatus.processedTokenCount == 3)

        let second = try await collect(session.streamDetails(to: "b"))

        #expect(second.text == "xxxx")
        // The cached transcript includes the committed EOS boundary; only
        // the new user token is prefilled on the resumed pass.
        #expect(second.info.promptTokenCount == 1)
        #expect(second.info.cachedPromptTokenCount == 3)
        #expect(second.info.stopReason == .length)
        #expect(second.info.proposedDraftTokens == 2)
        #expect(second.info.acceptedDraftTokens == 2)

        let continuedStatus = try await session.cacheStatus()
        #expect(continuedStatus.phase == .realized)
        // Three cached tokens, one suffix token, and three generated tokens
        // represented in K/V; the final sample remains the next decode input.
        #expect(continuedStatus.processedTokenCount == 7)
    }

    private func makeModelContext() -> ModelContext {
        let tokenizer = EarlyTerminationMTPTokenizer()
        let processor = EarlyTerminationMTPInputProcessor(tokenizer: tokenizer)
        return ModelContext(
            configuration: processor.configuration,
            model: EarlyTerminationMTPTarget(),
            processor: processor,
            tokenizer: tokenizer
        )
    }

    private func makeMTPConfiguration() throws -> SpeculativeDecodingConfig {
        let container = MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(id: "mtp-early-termination-test"),
                model: EarlyTerminationMTPDrafter()
            )
        )
        return try SpeculativeDecodingConfig(mtpDrafter: container, blockSize: 4)
    }

    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>
    ) async throws -> (text: String, info: GenerateCompletionInfo) {
        var text = ""
        var info: GenerateCompletionInfo?
        for try await event in stream {
            if let chunk = event.chunk {
                text += chunk
            }
            if let completion = event.info {
                info = completion
            }
        }
        return (text, try #require(info))
    }
}

private struct EarlyTerminationMTPTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { $0 == "x" ? 4 : 7 }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(repeating: "x", count: tokenIds.filter { $0 != 15 }.count)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<eos>" ? 15 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 15 ? "<eos>" : "x"
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            let content = message["content"] as? String ?? ""
            var rendered = encode(text: content, addSpecialTokens: false)
            if message["role"] as? String == "assistant" {
                rendered.append(15)
            }
            return rendered
        }
    }
}

private struct EarlyTerminationMTPInputProcessor: UserInputProcessor {
    let tokenizer: EarlyTerminationMTPTokenizer
    let configuration = ModelConfiguration(id: "mtp-early-termination-test")

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}
