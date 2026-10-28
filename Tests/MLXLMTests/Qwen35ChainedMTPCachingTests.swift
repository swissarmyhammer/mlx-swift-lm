import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The caches a chained Qwen MTP stream leaves: a stop inside a round leaves the drafter
/// continuation of the tokens it returned, and that continuation saves to a prompt cache file
/// whose restore continues with the plain greedy tokens.
@Suite(.serialized)
struct Qwen35ChainedMTPCachingTests {

    /// The prompt of each first call.
    private static let prompt = Qwen35VerifyRewindTests.decodePrompt

    /// The tokens each first call generates at most.
    private static let generatedTokenCount = 16

    /// The prompt tokens a second call appends after the tokens the main cache holds.
    private static let followUp = [8, 30, 2]

    /// The tokens each second call generates.
    private static let continuedTokenCount = 12

    /// What a first call left: the main cache, the tokens it holds, the drafter continuation,
    /// and whether the main cache held the last returned token before the finalize.
    private struct StoppedCall {
        let cache: [KVCache]
        let heldTokens: [Int]
        let continuation: MTPDrafterContinuation?
        let stoppedOnACachedToken: Bool
        let passthroughReason: String?
    }

    /// A greedy MTP iterator of the fixture's drafter over `input`.
    private func iterator(
        _ fixture: Qwen35MTPFixture, input: [Int], cache: [KVCache],
        continuation: MTPDrafterContinuation? = nil, blockSize: Int, maxTokens: Int
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(input.map(Int32.init))), mainModel: fixture.target,
            drafter: fixture.drafter, mainCache: cache, mtpContinuation: continuation,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            blockSize: blockSize)
    }

    /// Decodes after ``prompt``, stops after `stop` tokens, and finalizes the decode.
    private func stoppedCall(
        _ fixture: Qwen35MTPFixture, blockSize: Int, stop: Int
    ) throws -> StoppedCall {
        let cache = try fixture.target.newCache(parameters: nil)
        var decode = try iterator(
            fixture, input: Self.prompt, cache: cache, blockSize: blockSize,
            maxTokens: Self.generatedTokenCount)
        var tokens: [Int] = []
        while tokens.count < stop, let token = decode.next() {
            tokens.append(token)
        }
        let stoppedOnACachedToken = decode.hasCachedLastToken
        decode.finalizeGeneration()
        let held = KVCacheStorage(cache, plan: .disabled).processedTokenCount
        return StoppedCall(
            cache: cache, heldTokens: Array((Self.prompt + tokens).prefix(held)),
            continuation: decode.mtpDrafterContinuation,
            stoppedOnACachedToken: stoppedOnACachedToken,
            passthroughReason: decode.passthroughReason)
    }

    /// The continuation a prefill of `tokens` holds at their end.
    private func prefillContinuation(
        _ fixture: Qwen35MTPFixture, tokens: [Int], blockSize: Int
    ) throws -> MTPDrafterContinuation? {
        try iterator(
            fixture, input: tokens, cache: try fixture.target.newCache(parameters: nil),
            blockSize: blockSize, maxTokens: Self.generatedTokenCount
        ).promptBoundaryContinuation
    }

    @Test(
        "a stop inside a chained round leaves the drafter continuation of the returned tokens",
        arguments: Qwen35ChainedMTPDraftTests.chainedBlockSizes)
    func stopInsideAChainedRoundLeavesThePrefillContinuation(blockSize: Int) throws {
        let fixture = try Qwen35MTPFixture()
        var stopsOnCachedTokens = 0

        for stop in 1 ... Self.generatedTokenCount {
            let call = try stoppedCall(fixture, blockSize: blockSize, stop: stop)
            let continuation = try #require(
                call.continuation, "stop \(stop): \(call.passthroughReason ?? "")")
            let reference = try #require(
                try prefillContinuation(fixture, tokens: call.heldTokens, blockSize: blockSize))
            if call.stoppedOnACachedToken { stopsOnCachedTokens += 1 }

            #expect(continuation.targetProcessedTokenCount == call.heldTokens.count, "stop \(stop)")
            #expect(continuation.state.nextPosition == reference.state.nextPosition, "stop \(stop)")
            for (got, expected) in zip(
                continuation.state.cache.flatMap(\.state), reference.state.cache.flatMap(\.state))
            {
                let gap = AttentionKernelTolerance.worstGap(got, expected)
                #expect(
                    gap <= Qwen35MTPFixture.rebuiltCacheTolerance, "stop \(stop): worst gap \(gap)")
            }
        }
        // Some stop landed on an accepted draft, so its finalize discarded verified lookahead.
        #expect(stopsOnCachedTokens > 0)
    }

    @Test(
        "a prompt cache file of a chained call keeps its continuation, and its restore continues with the plain tokens",
        arguments: Qwen35ChainedMTPDraftTests.chainedBlockSizes)
    func promptCacheFileOfAChainedCallResumes(blockSize: Int) async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try stoppedCall(fixture, blockSize: blockSize, stop: Self.generatedTokenCount)
        let continuation = try #require(first.continuation)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let plain = try await fixture.decodeGreedily(
            prompt: first.heldTokens + Self.followUp, maxTokens: Self.continuedTokenCount,
            speculative: false)

        try savePromptCache(
            url: url, cache: first.cache,
            drafterContinuation: try SerializedMTPDrafterContinuation(continuation))
        let snapshot = try loadPromptCacheSnapshot(url: url)
        let restored = try #require(snapshot.drafterContinuation).restored(by: fixture.drafter)
        var second = try iterator(
            fixture, input: Self.followUp, cache: snapshot.cache, continuation: restored,
            blockSize: blockSize, maxTokens: Self.continuedTokenCount)
        let resumedState = try #require(second.drafterState)
        var tokens: [Int] = []
        while let token = second.next() {
            tokens.append(token)
        }

        #expect(resumedState.cacheStartPosition == 0)
        #expect(tokens == plain.tokens)
        #expect(second.passthroughReason == nil)
        #expect(second.proposedCount > 0)
    }
}
