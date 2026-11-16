// Copyright © 2026 Apple Inc.

// DFlash2 speculation over a tiny hybrid Qwen 3.5 target: the target-layer taps the
// drafter reads, greedy MTP-iterator decoding against plain decoding, and a ChatSession
// that keeps its prompt cache across turns.

import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Target-layer taps

@Suite
struct Qwen35TargetLayerTapTests {

    /// The prompt the target reads before a verify.
    private static var prompt: MLXArray {
        seededTokens(9, vocabularySize: DFlash2Fixture.vocabularySize, seed: 1)
    }

    /// The state that asks the target for its drafter state, with the taps of `layers`.
    private static func emitState(layers: [Int]?, verify: Bool = false) -> LMOutput.State {
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpTargetLayerIdsKey] = layers
        state[mtpVerifyKeepsSnapshotsKey] = verify ? true : nil
        return state
    }

    /// The hidden rows the target emits for `tokens` after `prompt` on a new cache.
    private static func emittedHidden(
        _ target: Qwen35TextModel, tokens: MLXArray, layers: [Int]?, verify: Bool,
        compiled: Bool
    ) throws -> MLXArray {
        let cache = try target.newCache(parameters: nil)
        _ = target(LMInput.Text(tokens: prompt[.newAxis]), cache: cache, state: nil)
        let output = target.forward(
            LMInput.Text(tokens: tokens[.newAxis]), cache: cache,
            state: emitState(layers: layers, verify: verify), lastTokenOnly: false,
            useCompiledDecode: compiled)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        eval(hidden)
        return hidden
    }

    @Test("the taps are the layer outputs, side by side")
    func tapsAreLayerOutputs() throws {
        let target = try DFlash2Fixture.target()
        let lastLayer = DFlash2Fixture.targetLayers - 1
        let tokens = seededTokens(4, vocabularySize: DFlash2Fixture.vocabularySize, seed: 2)
        let taps = try Self.emittedHidden(
            target, tokens: tokens, layers: [0, lastLayer], verify: false, compiled: false)
        let normed = try Self.emittedHidden(
            target, tokens: tokens, layers: nil, verify: false, compiled: false)

        #expect(taps.dim(-1) == 2 * DFlash2Fixture.hiddenSize)
        let lastTap = taps[.ellipsis, DFlash2Fixture.hiddenSize...]
        #expect(allClose(target.model.norm(lastTap), normed, atol: 1e-5).item(Bool.self))
    }

    @Test(
        "the compiled verify taps what the general path taps",
        arguments: Array(CompiledVerifyShape.tokenRange))
    func compiledVerifyTapsMatchGeneralPath(tokens count: Int) throws {
        let target = try DFlash2Fixture.target()
        let tokens = seededTokens(count, vocabularySize: DFlash2Fixture.vocabularySize, seed: 3)
        let general = try Self.emittedHidden(
            target, tokens: tokens, layers: DFlash2Fixture.targetLayerIds, verify: true,
            compiled: false)
        let compiled = try Self.emittedHidden(
            target, tokens: tokens, layers: DFlash2Fixture.targetLayerIds, verify: true,
            compiled: true)

        #expect(target.model.compiledVerifySegmentCount > 0)
        #expect(allClose(compiled, general, atol: 1e-4).item(Bool.self))
    }
}

// MARK: - Iterator

@Suite
struct DFlash2IteratorTests {

    /// The prompt, longer than the drafter window.
    private static var prompt: MLXArray {
        seededTokens(20, vocabularySize: DFlash2Fixture.vocabularySize, seed: 11)
    }

    /// The tokens each decode generates.
    private static let generatedTokens = 24

    private static var parameters: GenerateParameters {
        GenerateParameters(maxTokens: generatedTokens, temperature: 0)
    }

    private static func decode<Iterator: TokenIteratorProtocol>(_ iterator: inout Iterator) -> [Int]
    {
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return tokens
    }

    @Test(
        "greedy DFlash2 decodes the tokens plain greedy decoding decodes",
        arguments: [4, 5, DFlash2Fixture.trainedBlockSize])
    func greedyMatchesPlain(blockSize: Int) throws {
        let target = try DFlash2Fixture.target()
        let drafter = try DFlash2Fixture.drafter()
        var plain = try TokenIterator(
            input: LMInput(tokens: Self.prompt), model: target, parameters: Self.parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: Self.prompt), mainModel: target, drafter: drafter,
            parameters: Self.parameters, blockSize: blockSize)

        #expect(Self.decode(&speculative) == Self.decode(&plain))
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
        #expect(speculative.blockSize == min(blockSize, drafter.maximumBlockSize ?? blockSize))
    }

    @Test("a stream that resumes a continuation decodes what a cold stream decodes")
    func continuationMatchesColdStream() throws {
        let target = try DFlash2Fixture.target()
        let drafter = try DFlash2Fixture.drafter()
        let followUp = seededTokens(6, vocabularySize: DFlash2Fixture.vocabularySize, seed: 12)

        let cache = try target.newCache(parameters: nil)
        var first = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: Self.prompt), mainModel: target, drafter: drafter,
            mainCache: cache, parameters: Self.parameters, blockSize: 5)
        let firstTokens = Self.decode(&first)
        first.finalizeGeneration()
        let continuation = try #require(first.mtpDrafterContinuation)

        let suffix = concatenated([MLXArray(firstTokens.map(Int32.init)), followUp])
        let cachedCount = KVCacheStorage(cache, plan: .disabled).processedTokenCount
        let pending = suffix[(cachedCount - Self.prompt.dim(0))...]
        var warm = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: pending), mainModel: target, drafter: drafter,
            mainCache: cache, mtpContinuation: continuation, parameters: Self.parameters,
            blockSize: 5)
        var plain = try TokenIterator(
            input: LMInput(tokens: concatenated([Self.prompt, suffix])), model: target,
            parameters: Self.parameters)

        #expect(Self.decode(&warm) == Self.decode(&plain))
        #expect(warm.passthroughReason == nil)
    }
}

// MARK: - Chat session

/// A chat session with a DFlash2 drafter writes what a plain session writes, and
/// reuses its prompt cache as a plain session does, over turns and a fork.
@Suite(.serialized)
struct ChatSessionDFlash2Tests {

    private static let parameters = GenerateParameters(maxTokens: 12, temperature: 0)
    private static let instructions = "answer briefly"
    private static let turns = ["first", "second", "third"]

    private struct Turn: Equatable {
        var text = ""
        var cachedPromptTokenCount = 0
        var promptTokenCount = 0
        var proposedDraftTokens = 0
    }

    private static func session(speculates: Bool) throws -> ChatSession {
        let target = try DFlash2Fixture.target()
        let tokenizer = ScalarTokenizer(vocabularySize: DFlash2Fixture.vocabularySize)
        let configuration = ModelConfiguration(id: "dflash2-session-test")
        let context = ModelContext(
            configuration: configuration, model: target,
            processor: ScalarTokenizerInputProcessor(
                tokenizer: tokenizer, configuration: configuration),
            tokenizer: tokenizer)
        let speculation =
            speculates
            ? try SpeculativeDecodingConfig(
                mtpDrafter: MTPDrafterContainer(
                    context: MTPDrafterContext(
                        configuration: ModelConfiguration(id: "dflash2-test"),
                        model: try DFlash2Fixture.drafter())),
                blockSize: 5)
            : nil
        return ChatSession(
            context, instructions: instructions, speculativeDecoding: speculation,
            generateParameters: parameters)
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
    private static func conversation(speculates: Bool) async throws -> [Turn] {
        let session = try session(speculates: speculates)
        var result: [Turn] = []
        for prompt in turns {
            result.append(try await respond(session, to: prompt))
        }
        result.append(try await respond(await session.fork(history: []), to: turns[1]))
        return result
    }

    @Test("a session with a DFlash2 drafter writes what a plain session writes")
    func writesWhatAPlainSessionWrites() async throws {
        let plain = try await Self.conversation(speculates: false)
        let drafted = try await Self.conversation(speculates: true)

        #expect(drafted.map(\.text) == plain.map(\.text))
        #expect(plain.allSatisfy { !$0.text.isEmpty })
        #expect(drafted.allSatisfy { $0.proposedDraftTokens > 0 })
    }

    /// The tokens of a turn's prompt, cached and prefilled.
    private static func promptLength(_ turn: Turn) -> Int {
        turn.cachedPromptTokenCount + turn.promptTokenCount
    }

    @Test("a session with a DFlash2 drafter reuses its prompt cache on every follow-up")
    func reusesItsPromptCache() async throws {
        let plain = try await Self.conversation(speculates: false)
        let drafted = try await Self.conversation(speculates: true)

        #expect(drafted.map(Self.promptLength) == plain.map(Self.promptLength))
        #expect(drafted.dropFirst().allSatisfy { $0.cachedPromptTokenCount > 0 })
    }

    @Test("a follow-up prefills the new message and at most the last token of the turn before")
    func followUpsPrefillOnlyTheNewMessage() async throws {
        let plain = try await Self.conversation(speculates: false)
        let drafted = try await Self.conversation(speculates: true)

        // A speculative turn ends on the token its last verify sampled, which no forward
        // pass fed; the next turn prefills it with its own message.
        #expect(
            zip(drafted, plain).allSatisfy { drafted, plain in
                drafted.promptTokenCount <= plain.promptTokenCount + 1
            })
    }
}
