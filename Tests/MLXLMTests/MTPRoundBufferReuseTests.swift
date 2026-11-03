// Copyright © 2026 Apple Inc.
//
// The buffers an MTP round of the Qwen 3.5 text target reuses from round to round.
//
// A verify writes the K/V of its tokens into the K/V buffer of each attention layer. MLX writes
// in place only when nothing else holds a view of that buffer; any other holder makes the write
// copy the whole buffer, which at 100k context tokens is about 0.4 GiB for each such layer and
// round. The shared K/V that the target emits for its drafter is such a view, and the Qwen MTP
// drafter never reads it: a round drops it before its verify, and the buffer stays the same
// from round to round within one capacity step of the cache.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct MTPRoundBufferReuseTests {

    /// The prompt of each decode of these tests.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each decode generates: a few rounds, within one capacity step of the K/V
    /// buffers, so that no round grows them.
    private static let generatedTokenCount = 40

    /// The address of the first byte of the buffer of `array`.
    private static func bufferAddress(_ array: MLXArray) -> UInt {
        array.asData(access: .noCopy).data.withUnsafeBytes {
            UInt(bitPattern: $0.baseAddress)
        }
    }

    /// The buffer addresses of the keys and the values of every attention layer of `cache`.
    private static func keyValueBufferAddresses(_ cache: [KVCache]) -> [UInt] {
        cache.compactMap { $0 as? KVCacheSimple }.flatMap { layer -> [UInt] in
            [layer.keys, layer.values].compactMap { $0 }.map(bufferAddress)
        }
    }

    /// Each round of a greedy MTP decode writes the K/V of its verify into the buffers the
    /// previous round wrote: no round copies the K/V buffer of an attention layer, the layer
    /// whose K/V the target emits for its drafter included.
    @Test(arguments: [Qwen35MTPFixture.DrafterMode.copiesTarget, .independent])
    func everyRoundWritesTheKeyValueBuffersInPlace(mode: Qwen35MTPFixture.DrafterMode) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        var iterator = try fixture.greedyMTPIterator(
            over: Self.prompt, blockSize: Qwen35MTPFixture.blockSize,
            maxTokens: Self.generatedTokenCount)
        #expect(iterator.next() != nil)
        let addresses = Self.keyValueBufferAddresses(iterator.mainCache)
        try #require(!addresses.isEmpty)

        var roundCount = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
        var checkedRounds = 0
        while iterator.next() != nil {
            let rounds = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            guard rounds > roundCount else { continue }
            roundCount = rounds
            checkedRounds += 1
            #expect(
                Self.keyValueBufferAddresses(iterator.mainCache) == addresses,
                "round \(rounds) copied a K/V buffer")
        }
        #expect(checkedRounds > 1)
    }

    /// Dropping the shared K/V before the verify changes no token: the greedy MTP stream still
    /// equals plain decode.
    @Test(arguments: [Qwen35MTPFixture.DrafterMode.copiesTarget, .independent])
    func theGreedyStreamStillEqualsPlainDecode(mode: Qwen35MTPFixture.DrafterMode) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true)

        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
    }
}
