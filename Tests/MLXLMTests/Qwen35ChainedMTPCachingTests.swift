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

    /// Decodes after ``prompt``, stops after `stop` tokens, and finalizes the decode.
    private func stoppedCall(
        _ fixture: Qwen35MTPFixture, blockSize: Int, stop: Int
    ) throws -> StoppedCall {
        let cache = try fixture.target.newCache(parameters: nil)
        var decode = try fixture.greedyMTPIterator(
            over: Self.prompt, blockSize: blockSize, maxTokens: Self.generatedTokenCount,
            cache: cache)
        let tokens = drain(&decode, limit: stop)
        let stoppedOnACachedToken = decode.hasCachedLastToken
        decode.finalizeGeneration()
        return StoppedCall(
            cache: cache,
            heldTokens: Array((Self.prompt + tokens).prefix(heldTokenCount(of: cache))),
            continuation: decode.mtpDrafterContinuation,
            stoppedOnACachedToken: stoppedOnACachedToken,
            passthroughReason: decode.passthroughReason)
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
                try fixture.prefillContinuation(of: call.heldTokens, blockSize: blockSize))
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
        var second = try fixture.greedyMTPIterator(
            over: Self.followUp, blockSize: blockSize, maxTokens: Self.continuedTokenCount,
            cache: snapshot.cache, mtpContinuation: restored)
        let resumedState = try #require(second.drafterState)
        let tokens = drain(&second)

        #expect(resumedState.cacheStartPosition == 0)
        #expect(tokens == plain.tokens)
        #expect(second.passthroughReason == nil)
        #expect(second.proposedCount > 0)
    }
}
