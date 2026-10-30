import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Qwen MTP decode whose drafter attends to a window of the context
/// (``GenerateParameters/mtpDrafterAttentionWindow``) while the main cache holds all of it.
///
/// The drafter only proposes and the target verifies, so a windowed drafter writes the plain
/// greedy tokens of a full main cache: the window changes how many drafts the target accepts,
/// never the tokens. The drafter cache is a ring of the window with a rewind reserve, so its
/// rejected entries rewind after the ring wraps. A continuation saved with a window restores
/// only into a ring of that window; a stream of another window, or of the whole context,
/// starts its drafter cold.
@Suite(.serialized)
struct Qwen35MTPDrafterWindowTests {

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The drafter window: shorter than the tokens a decode holds, so each drafter ring wraps
    /// early in the decode.
    private static let window = 12

    /// A window other than ``window``, which a windowed continuation must not restore into.
    private static let otherWindow = 20

    /// The prefix a prefix-keeping drafter window keeps, as `maxKVSize` keeps it.
    private static let keptPrefix = 4

    /// The tokens each decode generates: several windows past the wrap.
    private static let generatedTokenCount = 48

    /// The block sizes of the decodes: one draft for each round, and the longest chain.
    private static let blockSizes = [Qwen35MTPFixture.blockSize, GatedDeltaDecodeKernel.mostTokens]

    /// The prompt a restored session appends after the cached tokens.
    private static let followUp = [8, 30, 2]

    /// A drafter window of `tokens` that keeps `keptPrefix` tokens.
    private static func drafterWindow(
        _ tokens: Int = window, keptPrefix: Int = 0
    ) throws -> KVCacheConfiguration.Capacity {
        try KVCacheConfiguration.Capacity(maxTokens: tokens, preservedPrefixTokens: keptPrefix)
    }

    /// The greedy parameters of a decode whose drafter attends to `drafterWindow`, over a main
    /// cache of `kvCache`, or of the whole context when `kvCache` is `nil`.
    private static func parameters(
        drafterWindow: KVCacheConfiguration.Capacity?, kvCache: KVCacheConfiguration? = nil
    ) -> GenerateParameters {
        var parameters = GenerateParameters(
            maxTokens: generatedTokenCount, kvCache: kvCache, temperature: 0)
        parameters.mtpDrafterAttentionWindow = drafterWindow
        return parameters
    }

    /// The plain greedy tokens of `target` after ``prompt`` under `parameters`.
    private func plainTokens(
        _ target: any LanguageModel, parameters: GenerateParameters
    ) throws -> [Int] {
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))), model: target,
            parameters: parameters)
        return drain(&plain)
    }

    /// The most rows a drafter cache of ``window`` holds: the window, the rewind reserve, and
    /// the rows a commit appends at once.
    private func drafterRowBound(blockSize: Int) -> Int {
        Self.window
            + ShiftedMTPDrafter.rewindReserve(maximumBlockSize: GatedDeltaDecodeKernel.mostTokens)
            + blockSize
    }

    // MARK: - The drafter cache

    @Test("a drafter window gives each drafter layer a ring of the window and its kept prefix")
    func windowGivesEachDrafterLayerARing() throws {
        let fixture = try Qwen35MTPFixture()
        let window = try Self.drafterWindow(keptPrefix: Self.keptPrefix)

        let state = fixture.drafter.makeState(parameters: Self.parameters(drafterWindow: window))

        for layer in state.cache {
            let ring = try #require(layer as? RotatingKVCache)
            #expect(ring.maxSize == Self.window)
            #expect(ring.keepCount == Self.keptPrefix)
        }
    }

    @Test("a drafter without a window keeps the whole context")
    func noWindowKeepsTheWholeContext() throws {
        let fixture = try Qwen35MTPFixture()

        let state = fixture.drafter.makeState(parameters: Self.parameters(drafterWindow: nil))

        #expect(state.cache.allSatisfy { type(of: $0) == KVCacheSimple.self })
    }

    @Test(
        "the narrower of the main window and the drafter window bounds the drafter",
        arguments: [(Self.window, Self.otherWindow), (Self.otherWindow, Self.window)])
    func narrowerWindowBoundsTheDrafter(mainWindow: Int, drafterWindow: Int) throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = Self.parameters(
            drafterWindow: try Self.drafterWindow(drafterWindow),
            kvCache: KVCacheConfiguration(capacity: try Self.drafterWindow(mainWindow)))

        let state = fixture.drafter.makeState(parameters: parameters)

        for layer in state.cache {
            #expect((layer as? RotatingKVCache)?.maxSize == min(mainWindow, drafterWindow))
        }
    }

    @Test(
        "a windowed drafter cache stays bounded past the window and stays aligned",
        arguments: blockSizes)
    func windowedDrafterCacheStaysBounded(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize,
            parameters: Self.parameters(drafterWindow: try Self.drafterWindow()))
        _ = drain(&mtp)
        let state = try #require(mtp.drafterState)

        #expect(state.isAligned)
        #expect(state.entryCount > drafterRowBound(blockSize: blockSize))
        for layer in state.cache {
            let rows = try #require(layer.state.first).dim(HeadRowsAxis.row)
            #expect(rows <= drafterRowBound(blockSize: blockSize))
        }
    }

    /// The rows of each chunk a windowed drafter state takes in turn: single-row writes and
    /// multi-row writes, as decode commits them.
    private static let drafterChunkRows = [1, 3]

    /// Appends the positions `range` of `tokens` to `state`: position `j` takes token `j + 1`
    /// with the target hidden row of position `j`.
    private static func append(
        _ range: Range<Int>, of tokens: [Int], hidden: MLXArray, to state: inout MTPDrafterState,
        fixture: Qwen35MTPFixture
    ) throws {
        let appended = fixture.drafter.appendDrafterState(
            target: fixture.target,
            tokens: MLXArray(tokens[(range.lowerBound + 1) ... range.upperBound].map(Int32.init))[
                .newAxis],
            targetHidden: hidden[0..., range, 0...], positionDeltas: nil, state: &state,
            sampler: ArgMaxSampler())
        try #require(appended, "the drafter must take positions \(range)")
    }

    @Test(
        "a windowed drafter attends to its window alone: its seed is the seed of a whole-context drafter fed only the window"
    )
    func windowedDrafterAttendsToItsWindowAlone() throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let full = Self.parameters(drafterWindow: nil)
        let tokens = Self.prompt + (try plainTokens(fixture.target, parameters: full))
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        let hidden = try #require(
            fixture.target(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis]),
                cache: try fixture.target.newCache(parameters: nil), state: emit
            ).state?[mtpLastHiddenStatesKey])
        let positionCount = tokens.count - 1
        var windowed = fixture.drafter.makeState(
            parameters: Self.parameters(drafterWindow: try Self.drafterWindow()))
        var start = 0
        for rows in Array(repeating: Self.drafterChunkRows, count: positionCount).joined()
        where start < positionCount {
            let end = min(start + rows, positionCount)
            try Self.append(start ..< end, of: tokens, hidden: hidden, to: &windowed, fixture: fixture)
            start = end
        }
        let windowStart = positionCount - Self.window
        var reference = MTPDrafterState(
            cache: fixture.drafter.makeState(parameters: full).cache,
            cacheStartPosition: windowStart, nextPosition: windowStart)
        try Self.append(
            windowStart ..< positionCount, of: tokens, hidden: hidden, to: &reference,
            fixture: fixture)

        let gap = AttentionKernelTolerance.worstGap(
            try #require(windowed.seedHidden), try #require(reference.seedHidden))
        #expect(gap <= Qwen35MTPFixture.rebuiltCacheTolerance, "worst gap \(gap)")
    }

    // MARK: - Output identity

    @Test(
        "greedy MTP decode with a windowed drafter speculates past the window and equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, blockSizes)
    func windowedDrafterEqualsPlainDecode(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let parameters = Self.parameters(drafterWindow: try Self.drafterWindow())
        let expected = try plainTokens(fixture.target, parameters: parameters)

        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: parameters)
        let tokens = drain(&mtp)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
        #expect(try #require(mtp.drafterState).entryCount > Self.window)
    }

    @Test(
        "greedy MTP decode with a prefix-keeping drafter window equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, blockSizes)
    func prefixKeepingDrafterWindowEqualsPlainDecode(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let parameters = Self.parameters(
            drafterWindow: try Self.drafterWindow(keptPrefix: Self.keptPrefix))
        let expected = try plainTokens(fixture.target, parameters: parameters)

        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: parameters)
        let tokens = drain(&mtp)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
    }

    @Test(
        "greedy MTP decode of the vision-language target with a windowed drafter equals plain decode",
        arguments: blockSizes)
    func windowedVLMDrafterEqualsPlainDecode(blockSize: Int) throws {
        let fixture = try Qwen35VLMMTPFixture()
        let parameters = Self.parameters(drafterWindow: try Self.drafterWindow())
        let expected = try plainTokens(fixture.target, parameters: parameters)

        var mtp = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, parameters: parameters,
            blockSize: blockSize)
        let tokens = drain(&mtp)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
        #expect(
            try #require(mtp.drafterState).cache.allSatisfy {
                ($0 as? RotatingKVCache)?.maxSize == Self.window
            })
    }

    @Test(
        "a reject after the windowed drafter ring wraps rewinds it exactly and keeps it aligned",
        arguments: blockSizes)
    func rejectAfterTheWindowWrapsKeepsTheDrafterAligned(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let parameters = Self.parameters(drafterWindow: try Self.drafterWindow())
        let expected = try plainTokens(fixture.target, parameters: parameters)
        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: parameters)
        var tokens: [Int] = []
        var rejectsAfterWrap = 0
        var rejected = 0

        while let token = mtp.next() {
            tokens.append(token)
            let state = try #require(mtp.drafterState)
            let rejectedNow = mtp.proposedCount - mtp.acceptedCount
            if rejectedNow > rejected, state.entryCount > Self.window { rejectsAfterWrap += 1 }
            rejected = rejectedNow
            #expect(state.isAligned, "token \(tokens.count)")
        }

        #expect(rejectsAfterWrap > 0)
        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
    }

    // MARK: - Continuations

    /// The serialized drafter continuation a finished decode with a drafter of ``window``
    /// leaves.
    private func windowedContinuation(
        _ fixture: Qwen35MTPFixture
    ) throws -> SerializedMTPDrafterContinuation {
        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: Qwen35MTPFixture.blockSize,
            parameters: Self.parameters(drafterWindow: try Self.drafterWindow()))
        _ = drain(&mtp)
        mtp.finalizeGeneration()
        return try SerializedMTPDrafterContinuation(
            try #require(mtp.mtpDrafterContinuation, "\(mtp.passthroughReason ?? "")"))
    }

    @Test("a windowed drafter continuation restores into a ring of its window, aligned")
    func windowedContinuationRestoresIntoItsWindow() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try windowedContinuation(fixture)

        let restored = try serialized.restored(
            by: fixture.drafter,
            parameters: Self.parameters(drafterWindow: try Self.drafterWindow()))

        for layer in restored.state.cache {
            #expect((layer as? RotatingKVCache)?.maxSize == Self.window)
        }
        #expect(restored.state.isAligned)
    }

    @Test("a windowed drafter continuation does not restore into a whole-context drafter")
    func windowedContinuationRefusesTheWholeContext() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try windowedContinuation(fixture)

        #expect(throws: KVCacheError.self) {
            try serialized.restored(
                by: fixture.drafter, parameters: Self.parameters(drafterWindow: nil))
        }
    }

    @Test("a windowed drafter continuation does not restore into a drafter of another window")
    func windowedContinuationRefusesAnotherWindow() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try windowedContinuation(fixture)
        let otherParameters = Self.parameters(
            drafterWindow: try Self.drafterWindow(Self.otherWindow))

        #expect(throws: KVCacheError.self) {
            try serialized.restored(by: fixture.drafter, parameters: otherParameters)
        }
    }

    /// The drafter continuation a session restored from the prompt cache file of a session
    /// with a drafter of ``window`` has after one follow-up turn, beside the continuation the
    /// file saved.
    ///
    /// - Parameter restoredWindow: The drafter window of the restored session.
    /// - Returns: The saved and the resumed continuation, and the follow-up turn.
    private func restoredSessionContinuation(
        restoredWindow: KVCacheConfiguration.Capacity?
    ) async throws -> (
        saved: MTPDrafterContinuation, resumed: MTPDrafterContinuation,
        info: GenerateCompletionInfo
    ) {
        let fixture = try Qwen35MTPFixture()
        let source = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: Self.parameters(drafterWindow: try Self.drafterWindow()))
        _ = try await collectChatTurn(source.streamDetails(to: fixture.text(for: Self.prompt)))
        let saved = try #require(await source.drafterContinuation())
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try await source.saveCache(to: url)

        let restored = ChatSession(
            fixture.makeModelContext(), promptCache: try loadPromptCacheSnapshot(url: url),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: Self.parameters(drafterWindow: restoredWindow))
        let turn = try await collectChatTurn(
            restored.streamDetails(to: fixture.text(for: Self.followUp)))
        let resumed = try #require(await restored.drafterContinuation())
        return (saved, resumed, turn.info)
    }

    @Test("a session restored from the prompt cache file of a windowed session resumes its drafter")
    func restoredWindowedSessionResumesTheDrafter() async throws {
        let session = try await restoredSessionContinuation(
            restoredWindow: try Self.drafterWindow())

        // A resume keeps the entries the source drafter began; a cold start begins them at the
        // position the turn found cached.
        #expect(session.resumed.state.cacheStartPosition == session.saved.state.cacheStartPosition)
        #expect(session.info.passthroughReason == nil)
    }

    @Test(
        "a session of another drafter window restored from a windowed session starts its drafter cold"
    )
    func restoredSessionOfAnotherWindowStartsTheDrafterCold() async throws {
        let session = try await restoredSessionContinuation(
            restoredWindow: try Self.drafterWindow(Self.otherWindow))

        #expect(
            session.resumed.state.cacheStartPosition == session.saved.targetProcessedTokenCount)
        #expect(session.info.passthroughReason == nil)
    }
}
