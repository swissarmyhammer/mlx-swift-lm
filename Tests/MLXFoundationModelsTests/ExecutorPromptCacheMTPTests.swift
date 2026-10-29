// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels

/// The prompt cache of the executor keeps the MTP drafter state of a session beside its main
/// caches: a later turn resumes it, a checkpoint restore takes the state of the checkpoint, a
/// spill writes it to disk and a restore reads it back, and the memory budget counts its bytes.
///
/// Each test runs the shared Qwen 3.5 + MTP fixture through the executor, with its drafter and
/// without it, and holds the speculative text to the plain text. A drafter state that a turn
/// resumed still begins at position zero, where the first turn started it. A drafter that a turn
/// started cold begins where the cached prefix ends.
@Suite(.serialized)
struct ExecutorPromptCacheMTPTests {

    /// The identifier of the instructions entry, which names the session of every pass.
    private static let sessionID = "executor-prompt-cache-mtp"

    /// The tokens of the instructions.
    private static let instructions = [7, 21, 4, 33, 12, 18]

    /// The tokens of the first prompt.
    private static let firstPrompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of the follow-up prompt.
    private static let followUpPrompt = [11, 40, 8, 27, 19]

    /// The tokens an edit appends to the first prompt.
    private static let editTokens = [50, 51, 52]

    /// The tokens each turn generates.
    private static let generatedTokenCount = 16

    /// The position in the greedy decode of the instructions and the first prompt of a token
    /// that an MTP round verifies as a draft, thus feeds into the caches, when it ends the
    /// generation.
    private static let fedStopIndex = 2

    /// The identity of a drafter that is not the fixture's drafter.
    private static let otherDrafterID = MTPDrafterID("another-drafter")

    /// The tool of the passes that take a checkpoint. The fixture never calls it.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var tool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "lookup", description: "Looks up text.", parameters: String.generationSchema)
    }

    /// What the second pass of a conversation renders.
    private enum SecondTurn {

        /// The first prompt, the first response and a follow-up prompt: the render extends
        /// the ledger of the first pass.
        case followUp

        /// The first prompt with ``editTokens`` appended, and no response: the render parts
        /// from the ledger where the first pass started to generate.
        case edit
    }

    /// What the two passes of one conversation left.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Conversation {

        /// What the first pass streamed.
        let first: MTPTurn

        /// What the second pass streamed.
        let second: MTPTurn

        /// The ledger the first pass checked in.
        let firstLedger: [Int]

        /// Whether the first pass checked in a drafter state that stands at the end of its
        /// ledger.
        let firstDrafterStateIsAligned: Bool

        /// The entry the second pass checked in, or `nil` when it checked in none.
        let entry: ExecutorPromptCacheEntry?

        /// The length of the ledger the first pass checked in.
        var firstLedgerCount: Int { firstLedger.count }
    }

    /// The fixture's target, once plain and once with its drafter.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Models {
        let fixture: Qwen35MTPFixture
        let plain: MLXLanguageModel
        let speculative: MLXLanguageModel
    }

    /// Makes the plain and the speculative model of one fixture.
    ///
    /// - Parameters:
    ///   - weights: The directory that makes the models available.
    ///   - capabilities: The capabilities each model declares.
    ///   - eosTokenIds: The tokens that end a generation.
    /// - Returns: The models.
    /// - Throws: The error of the fixture or of the speculative decoding.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeModels(
        weights: URL, capabilities: [LanguageModelCapabilities.Capability] = [],
        eosTokenIds: Set<Int> = []
    ) throws -> Models {
        let fixture = try Qwen35MTPFixture()
        return Models(
            fixture: fixture,
            plain: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, eosTokenIds: eosTokenIds),
            speculative: fixture.makeLanguageModel(
                weights: weights, capabilities: capabilities, eosTokenIds: eosTokenIds,
                speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding()))
    }

    /// A store that spills to a new folder of its own.
    ///
    /// - Parameter writes: Whether a spill writes its file. A store that writes none keeps no
    ///   entry on disk.
    /// - Returns: The store, and its folder, which the caller removes.
    private func makeStore(writes: Bool = true) -> (store: ExecutorPromptCacheStore, folder: URL) {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("ExecutorPromptCacheMTPTests-\(UUID().uuidString)")
        let store =
            writes
            ? ExecutorPromptCacheStore(directory: folder)
            : ExecutorPromptCacheStore(directory: folder, writer: { _, _ in })
        return (store, folder)
    }

    /// The key of the session of `model`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func key(of model: MLXLanguageModel) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: model.modelID, sessionID: Self.sessionID)
    }

    /// The transcript of the instructions, each prompt, and the response after each prompt
    /// that `responses` holds.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func transcript(
        prompts: [[Int]], responses: [String], fixture: Qwen35MTPFixture
    ) -> Transcript {
        var entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    id: Self.sessionID,
                    segments: [.text(Transcript.TextSegment(content: fixture.text(for: Self.instructions)))],
                    toolDefinitions: []))
        ]
        for (index, prompt) in prompts.enumerated() {
            entries.append(
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: fixture.text(for: prompt)))])))
            if responses.indices.contains(index) {
                entries.append(
                    .response(
                        Transcript.Response(
                            assetIDs: [],
                            segments: [.text(Transcript.TextSegment(content: responses[index]))])))
            }
        }
        return Transcript(entries: entries)
    }

    /// Runs one greedy executor pass of `model` over `transcript` inside `store`.
    ///
    /// - Parameters:
    ///   - model: The model of the pass.
    ///   - transcript: The transcript of the request.
    ///   - tools: Whether the request enables ``tool``, which makes the pass take a checkpoint.
    ///   - store: The prompt cache store the pass binds.
    /// - Returns: What the pass streamed.
    /// - Throws: The error of the pass or of its channel.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func respond(
        _ model: MLXLanguageModel, over transcript: Transcript, tools: Bool,
        inside store: ExecutorPromptCacheStore
    ) async throws -> MTPTurn {
        let request = makeExecutorRequest(
            transcript: transcript, enabledTools: tools ? [Self.tool] : [],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount))
        let turn = try await ExecutorPromptCacheStore.$current.withValue(store) {
            try await MTPExecutorTurns.respond(to: request, model: model)
        }
        try #require(turn.failure == nil, "The pass failed: \(turn.failure ?? "")")
        try #require(turn.channelFailure == nil, "The channel failed: \(turn.channelFailure ?? "")")
        return turn
    }

    /// Runs the two passes of one conversation of `model` in a store of its own.
    ///
    /// - Parameters:
    ///   - model: The model of the conversation.
    ///   - fixture: The fixture whose tokens the prompts spell.
    ///   - secondTurn: What the second pass renders.
    ///   - tools: Whether the passes enable ``tool``.
    ///   - between: Runs between the two passes, with the store and the key of the session.
    /// - Returns: What the passes left.
    /// - Throws: The error of a pass or of `between`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func converse(
        _ model: MLXLanguageModel, fixture: Qwen35MTPFixture, secondTurn: SecondTurn,
        tools: Bool = false,
        between: (ExecutorPromptCacheStore, ExecutorPromptCacheKey) async throws -> Void = { _, _ in }
    ) async throws -> Conversation {
        let (store, folder) = makeStore()
        defer { try? FileManager.default.removeItem(at: folder) }
        let sessionKey = key(of: model)

        let first = try await respond(
            model, over: transcript(prompts: [Self.firstPrompt], responses: [], fixture: fixture),
            tools: tools, inside: store)
        let firstEntry = try #require(await store.peek(sessionKey), "The first pass must check in.")
        let firstLedger = firstEntry.tokens
        let firstDrafterStateIsAligned =
            firstEntry.drafterState.map {
                $0.isAligned(withMainTokenCount: firstLedger.count, drafterID: $0.drafterID)
            } ?? false
        try await between(store, sessionKey)

        let secondTranscript =
            switch secondTurn {
            case .followUp:
                transcript(
                    prompts: [Self.firstPrompt, Self.followUpPrompt], responses: [first.responseText],
                    fixture: fixture)
            case .edit:
                transcript(prompts: [Self.firstPrompt + Self.editTokens], responses: [], fixture: fixture)
            }
        let second = try await respond(model, over: secondTranscript, tools: tools, inside: store)
        await store.waitForSpills()
        return Conversation(
            first: first, second: second, firstLedger: firstLedger,
            firstDrafterStateIsAligned: firstDrafterStateIsAligned,
            entry: await store.peek(sessionKey))
    }

    /// Expects the speculative conversation to write the plain text and to speculate in its
    /// second pass, which reused the whole ledger of the first pass.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectPlainTextFromTheCachedPrefix(
        _ speculative: Conversation, plain: Conversation
    ) {
        #expect(!plain.second.responseText.isEmpty)
        #expect(speculative.first.responseText == plain.first.responseText)
        #expect(speculative.second.responseText == plain.second.responseText)
        #expect(speculative.second.speculated)
        #expect(speculative.second.usage.first?.cachedPromptTokenCount == speculative.firstLedgerCount)
    }

    /// The drafter state of the entry the conversation left.
    ///
    /// - Parameter conversation: The conversation.
    /// - Returns: The drafter state, which stands where the main caches of the entry end.
    /// - Throws: When the entry or its drafter state is missing, or they do not align.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func drafterState(of conversation: Conversation) throws -> MTPDrafterContinuation {
        let entry = try #require(conversation.entry, "The second pass must check in.")
        let drafterState = try #require(entry.drafterState, "The entry must keep a drafter state.")
        #expect(
            drafterState.isAligned(
                withMainTokenCount: entry.tokens.count, drafterID: drafterState.drafterID))
        return drafterState
    }

    // MARK: - Reuse

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpTurnResumesTheDrafterStateOfTheFirstTurn() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)

        let plain = try await converse(models.plain, fixture: models.fixture, secondTurn: .followUp)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .followUp)

        expectPlainTextFromTheCachedPrefix(speculative, plain: plain)
        #expect(try drafterState(of: speculative).state.cacheStartPosition == 0)
        #expect(plain.entry?.drafterState == nil)
    }

    /// A first turn that stops on a stop token an MTP round fed into the caches holds that
    /// token in its ledger, and the drafter state the turn checks in stands there too.
    ///
    /// The follow-up render leaves the stop token out, thus it goes back to the checkpoint at
    /// the end of the first prompt, as plain decode does, and resumes the drafter state of
    /// that checkpoint.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aTurnThatFedItsStopTokenKeepsTheDrafterStateAtTheEndOfItsLedger() async throws {
        let fixture = try Qwen35MTPFixture()
        let decode = try await fixture.decodeGreedily(
            prompt: Self.instructions + Self.firstPrompt, maxTokens: Self.generatedTokenCount,
            speculative: false)
        let stopToken = decode.tokens[Self.fedStopIndex]
        try #require(decode.tokens.firstIndex(of: stopToken) == Self.fedStopIndex)
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights, eosTokenIds: [stopToken])

        let plain = try await converse(models.plain, fixture: models.fixture, secondTurn: .followUp)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .followUp)

        try #require(speculative.first.completions.map(\.stopTokenFedToCache) == [stopToken])
        #expect(speculative.firstLedger == plain.firstLedger)
        #expect(speculative.firstLedger.last == stopToken)
        #expect(speculative.firstDrafterStateIsAligned)
        #expect(!plain.second.responseText.isEmpty)
        #expect(speculative.second.responseText == plain.second.responseText)
        #expect(speculative.second.speculated)
        #expect(
            speculative.second.usage.first?.cachedPromptTokenCount
                == plain.second.usage.first?.cachedPromptTokenCount)
        #expect(try drafterState(of: speculative).state.cacheStartPosition == 0)
    }

    // MARK: - Checkpoint

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anEditResumesTheDrafterStateOfTheCheckpoint() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights, capabilities: [.toolCalling])

        let plain = try await converse(
            models.plain, fixture: models.fixture, secondTurn: .edit, tools: true)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .edit, tools: true)

        #expect(!plain.second.responseText.isEmpty)
        #expect(speculative.second.responseText == plain.second.responseText)
        #expect(speculative.second.speculated)
        // The edit parts from the ledger after the first prompt, where the first pass took its
        // checkpoint, and the recurrent layer cannot rewind there.
        let firstPromptCount = try #require(speculative.first.usage.first?.promptTokenCount)
        #expect(speculative.second.usage.first?.cachedPromptTokenCount == firstPromptCount)
        #expect(try drafterState(of: speculative).state.cacheStartPosition == 0)
    }

    /// A template that writes a generation prompt makes the first pass feed its prompt up to
    /// the end of the last message before the generation, and take its checkpoint there. The
    /// follow-up render writes the response where the first render wrote the generation
    /// prompt, thus it goes back to that checkpoint, and resumes the drafter state that the
    /// feed left there.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aFollowUpTurnResumesTheDrafterStateFedToTheCheckpoint() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let fixture = try Qwen35MTPFixture()
        let tokenizer = GenerationPromptTokenizer()
        let plainModel = fixture.makeLanguageModel(weights: weights, tokenizer: tokenizer)
        let speculativeModel = fixture.makeLanguageModel(
            weights: weights, tokenizer: tokenizer,
            speculativeDecoding: try Qwen35MTPFixture.makeSpeculativeDecoding())

        let plain = try await converse(plainModel, fixture: fixture, secondTurn: .followUp)
        let speculative = try await converse(
            speculativeModel, fixture: fixture, secondTurn: .followUp)

        #expect(!plain.second.responseText.isEmpty)
        #expect(speculative.second.responseText == plain.second.responseText)
        #expect(speculative.second.speculated)
        let checkpointPosition = Self.instructions.count + Self.firstPrompt.count
        #expect(speculative.second.usage.first?.cachedPromptTokenCount == checkpointPosition)
        #expect(try drafterState(of: speculative).state.cacheStartPosition == 0)
    }

    // MARK: - Spill and restore

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpilledEntryRestoresItsDrafterState() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)
        let spill: (ExecutorPromptCacheStore, ExecutorPromptCacheKey) async throws -> Void = {
            store, key in
            await store.configure(memoryBudgetBytes: 0)
            await store.waitForSpills()
            try #require(await store.peek(key) == nil, "The entry must leave memory.")
            await store.configure(memoryBudgetBytes: .max)
        }

        let plain = try await converse(models.plain, fixture: models.fixture, secondTurn: .followUp)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .followUp, between: spill)

        expectPlainTextFromTheCachedPrefix(speculative, plain: plain)
        #expect(try drafterState(of: speculative).state.cacheStartPosition == 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aSpillFileWithoutADrafterStateRestoresWithAColdDrafter() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)
        let spillMainCachesOnly: (ExecutorPromptCacheStore, ExecutorPromptCacheKey) async throws -> Void = {
            store, key in
            let entry = try #require(await store.checkOut(key).entry)
            await store.configure(memoryBudgetBytes: 0)
            await store.checkIn(key, Self.mainCaches(of: entry))
            await store.waitForSpills()
            await store.configure(memoryBudgetBytes: .max)
        }

        let plain = try await converse(models.plain, fixture: models.fixture, secondTurn: .followUp)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .followUp,
            between: spillMainCachesOnly)

        expectPlainTextFromTheCachedPrefix(speculative, plain: plain)
        #expect(
            try drafterState(of: speculative).state.cacheStartPosition
                == speculative.firstLedgerCount)
    }

    // MARK: - Drafter identity

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func aDrafterStateOfAnotherDrafterStartsTheDrafterCold() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)
        let relabel: (ExecutorPromptCacheStore, ExecutorPromptCacheKey) async throws -> Void = {
            store, key in
            let entry = try #require(await store.checkOut(key).entry)
            let drafterState = try #require(entry.drafterState)
            await store.checkIn(
                key,
                ExecutorPromptCacheEntry(
                    caches: entry.caches, tokens: entry.tokens, renderTokens: entry.renderTokens,
                    state: entry.state, checkpoint: entry.checkpoint,
                    drafterState: MTPDrafterContinuation(
                        state: drafterState.state,
                        targetBoundaryHidden: drafterState.targetBoundaryHidden,
                        targetProcessedTokenCount: drafterState.targetProcessedTokenCount,
                        drafterID: Self.otherDrafterID)))
        }

        let plain = try await converse(models.plain, fixture: models.fixture, secondTurn: .followUp)
        let speculative = try await converse(
            models.speculative, fixture: models.fixture, secondTurn: .followUp, between: relabel)

        expectPlainTextFromTheCachedPrefix(speculative, plain: plain)
        let drafterState = try drafterState(of: speculative)
        #expect(drafterState.drafterID != Self.otherDrafterID)
        #expect(drafterState.state.cacheStartPosition == speculative.firstLedgerCount)
    }

    // MARK: - Memory budget

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func theMemoryBudgetCountsTheDrafterBytes() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let models = try makeModels(weights: weights)
        let (store, folder) = makeStore(writes: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        let sessionKey = key(of: models.speculative)
        _ = try await respond(
            models.speculative,
            over: transcript(prompts: [Self.firstPrompt], responses: [], fixture: models.fixture),
            tools: false, inside: store)
        let entry = try #require(await store.checkOut(sessionKey).entry)
        let drafterState = try #require(entry.drafterState)
        let mainCaches = Self.mainCaches(of: entry)

        #expect(drafterState.byteCount > 0)
        #expect(entry.byteCount == mainCaches.byteCount + drafterState.byteCount)

        await store.configure(memoryBudgetBytes: mainCaches.byteCount)
        await store.checkIn(sessionKey, entry)
        #expect(await store.peek(sessionKey) == nil)

        await store.configure(memoryBudgetBytes: entry.byteCount)
        await store.checkIn(sessionKey, entry)
        #expect(await store.peek(sessionKey) === entry)
    }

    /// `entry` without its drafter state: the entry an executor that keeps no drafter state
    /// checks in.
    private static func mainCaches(of entry: ExecutorPromptCacheEntry) -> ExecutorPromptCacheEntry {
        ExecutorPromptCacheEntry(
            caches: entry.caches, tokens: entry.tokens, renderTokens: entry.renderTokens,
            state: entry.state, checkpoint: entry.checkpoint)
    }
}

/// A ``ScalarTokenizer`` whose chat template writes a generation prompt after the last message,
/// as the Qwen 3.8 template writes `<think>\n`.
private struct GenerationPromptTokenizer: Tokenizer {

    /// The tokens the template writes after the last message.
    static let generationPrompt = [60, 61]

    /// The tokenizer that encodes, decodes and renders the messages.
    private let base = ScalarTokenizer(vocabularySize: Qwen35MTPFixture.vocabularySize)

    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }

    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }

    /// The tokens of the contents of `messages`, then ``generationPrompt``.
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
            + Self.generationPrompt
    }
}

#endif
