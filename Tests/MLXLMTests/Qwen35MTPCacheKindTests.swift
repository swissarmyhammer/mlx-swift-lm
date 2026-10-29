import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Qwen MTP decode when the main KV cache rotates (`maxKVSize`) or quantizes (`kvBits`).
///
/// A rotating main cache gives the drafter a cache that rotates with the same window and keeps a
/// rewind reserve, so the drafter stays bounded and rewinds its rejected entries after its ring
/// wraps. A ring without a kept prefix speculates past its window; a ring that keeps a prefix
/// stops speculating before a verify would attend past it. A quantized main cache leaves the
/// drafter cache in full precision. In each case the greedy MTP tokens are the plain greedy
/// tokens of the same cache configuration.
@Suite(.serialized)
struct Qwen35MTPCacheKindTests {

    /// The prompt of each decode.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The window of the rotating main cache: shorter than the tokens a decode holds, so each
    /// ring wraps early in the decode.
    private static let window = 12

    /// A window other than ``window``, which a rotating continuation must not restore into.
    private static let otherWindow = 20

    /// The tokens each decode generates: several windows past the wrap.
    private static let generatedTokenCount = 48

    /// The block sizes of the decodes: one draft for each round, and the longest chain.
    private static let blockSizes = [Qwen35MTPFixture.blockSize, GatedDeltaDecodeKernel.mostTokens]

    /// The width of each attention head of the quantized-KV fixture: one quantization group.
    private static let quantizedHeadDimension = 32

    /// The bits of the quantized main KV cache.
    private static let kvBits = 8

    /// The group size of the quantized main KV cache.
    private static let kvGroupSize = 32

    /// The greedy parameters of a decode whose main attention cache rotates with `window` and
    /// keeps no prefix, so each token attends to the `window` tokens up to it.
    private static func rotatingParameters(window: Int = window) throws -> GenerateParameters {
        GenerateParameters(
            maxTokens: generatedTokenCount,
            kvCache: KVCacheConfiguration(
                capacity: try .init(maxTokens: window, preservedPrefixTokens: 0)),
            temperature: 0)
    }

    /// The greedy parameters of a decode whose main attention cache rotates with ``window`` as
    /// `maxKVSize` makes it: the ring keeps the first tokens of the prompt.
    private static func prefixKeepingParameters() -> GenerateParameters {
        GenerateParameters(maxTokens: generatedTokenCount, maxKVSize: window, temperature: 0)
    }

    /// The greedy parameters of a decode whose main attention cache quantizes from the start.
    private static func quantizedParameters() -> GenerateParameters {
        GenerateParameters(
            maxTokens: generatedTokenCount, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: 0, temperature: 0)
    }

    /// The plain greedy tokens of `fixture` after ``prompt`` under `parameters`.
    private func plainTokens(
        _ fixture: Qwen35MTPFixture, parameters: GenerateParameters
    ) throws -> [Int] {
        var plain = try TokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))), model: fixture.target,
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

    @Test(
        "greedy MTP decode over a ring without a kept prefix speculates past the window and equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, blockSizes)
    func ringWithoutPrefixSpeculatesPastTheWindow(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let parameters = try Self.rotatingParameters()
        let expected = try plainTokens(fixture, parameters: parameters)

        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: parameters)
        let tokens = drain(&mtp)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
        #expect(try #require(mtp.drafterState).entryCount > Self.window)
    }

    @Test(
        "greedy MTP decode over a ring that keeps a prefix stops speculating at the window and equals plain decode",
        arguments: Qwen35MTPFixture.DrafterMode.allCases, blockSizes)
    func prefixKeepingRingStopsSpeculatingAtTheWindow(
        mode: Qwen35MTPFixture.DrafterMode, blockSize: Int
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let parameters = Self.prefixKeepingParameters()
        let expected = try plainTokens(fixture, parameters: parameters)

        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: parameters)
        let tokens = drain(&mtp)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == MTPSpeculativeTokenIterator.keptPrefixWindowReason)
        #expect(mtp.proposedCount > 0)
    }

    @Test(
        "the drafter cache of a rotating main cache rotates with its window and stays bounded",
        arguments: blockSizes)
    func drafterCacheRotatesWithTheMainWindow(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: blockSize, parameters: try Self.rotatingParameters())
        _ = drain(&mtp)
        let state = try #require(mtp.drafterState)

        #expect(state.isAligned)
        #expect(state.entryCount > drafterRowBound(blockSize: blockSize))
        for layer in state.cache {
            let ring = try #require(layer as? RotatingKVCache)
            let rows = try #require(ring.state.first).dim(HeadRowsAxis.row)
            #expect(ring.maxSize == Self.window)
            #expect(rows <= drafterRowBound(blockSize: blockSize))
        }
    }

    @Test(
        "a reject after the drafter ring wraps rewinds it exactly and keeps it aligned",
        arguments: blockSizes)
    func rejectAfterTheRingWrapsKeepsTheDrafterAligned(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture(drafter: .independent)
        let parameters = try Self.rotatingParameters()
        let expected = try plainTokens(fixture, parameters: parameters)
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

    @Test(
        "greedy MTP decode over an 8-bit main KV cache speculates with a full-precision drafter and equals plain 8-bit decode"
    )
    func quantizedMainCacheMatchesPlainDecode() throws {
        let fixture = try Qwen35MTPFixture(headDimension: Self.quantizedHeadDimension)
        let parameters = Self.quantizedParameters()
        let expected = try plainTokens(fixture, parameters: parameters)

        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: Qwen35MTPFixture.blockSize, parameters: parameters)
        let tokens = drain(&mtp)
        let state = try #require(mtp.drafterState)

        #expect(tokens == expected)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
        #expect(mtp.mainCache.contains { $0 is QuantizedKVCache })
        #expect(state.cache.allSatisfy { $0 is KVCacheSimple })
    }

    /// The serialized drafter continuation a finished decode over a ring without a kept prefix
    /// leaves.
    private func rotatingContinuation(
        _ fixture: Qwen35MTPFixture
    ) throws -> SerializedMTPDrafterContinuation {
        var mtp = try fixture.mtpIterator(
            over: Self.prompt, blockSize: Qwen35MTPFixture.blockSize,
            parameters: try Self.rotatingParameters())
        _ = drain(&mtp)
        mtp.finalizeGeneration()
        return try SerializedMTPDrafterContinuation(
            try #require(mtp.mtpDrafterContinuation, "\(mtp.passthroughReason ?? "")"))
    }

    @Test("a rotating drafter continuation does not restore into a simple drafter cache")
    func rotatingContinuationRefusesASimpleCache() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try rotatingContinuation(fixture)

        #expect(throws: KVCacheError.self) {
            try serialized.restored(by: fixture.drafter)
        }
    }

    @Test("a rotating drafter continuation does not restore into a ring of another window")
    func rotatingContinuationRefusesAnotherWindow() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try rotatingContinuation(fixture)
        let otherParameters = try Self.rotatingParameters(window: Self.otherWindow)

        #expect(throws: KVCacheError.self) {
            try serialized.restored(by: fixture.drafter, parameters: otherParameters)
        }
    }

    @Test("a rotating drafter continuation restores into a ring of its window")
    func rotatingContinuationRestoresIntoItsWindow() throws {
        let fixture = try Qwen35MTPFixture()
        let serialized = try rotatingContinuation(fixture)

        let restored = try serialized.restored(
            by: fixture.drafter, parameters: try Self.rotatingParameters())

        for layer in restored.state.cache {
            #expect((layer as? RotatingKVCache)?.maxSize == Self.window)
        }
        #expect(restored.state.isAligned)
    }

    /// The prompt a restored session appends after the cached tokens.
    private static let followUp = [8, 30, 2]

    @Test(
        "a session restored from the prompt cache file of a rotating session resumes its drafter continuation"
    )
    func restoredRotatingSessionResumesTheSavedContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = try Self.rotatingParameters()
        let source = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: parameters)
        _ = try await collectChatTurn(source.streamDetails(to: fixture.text(for: Self.prompt)))
        let saved = try #require(await source.drafterContinuation())
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try await source.saveCache(to: url)

        let restored = ChatSession(
            fixture.makeModelContext(), promptCache: try loadPromptCacheSnapshot(url: url),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: parameters)
        let turn = try await collectChatTurn(
            restored.streamDetails(to: fixture.text(for: Self.followUp)))

        // A resume keeps the entries the source drafter began; a cold start begins them at the
        // position the turn found cached.
        let continuation = try #require(await restored.drafterContinuation())
        #expect(continuation.state.cacheStartPosition == saved.state.cacheStartPosition)
        #expect(turn.info.passthroughReason == nil)
    }
}
