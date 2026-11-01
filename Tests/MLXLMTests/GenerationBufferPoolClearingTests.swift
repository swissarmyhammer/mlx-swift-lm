// Copyright © 2026 Apple Inc.

import MLX
import MLXLMTestSupport
import Testing
import os

@testable import MLXLMCommon

/// Counts the clears of an iterator, so that a test does not read the process-wide MLX pool.
private struct ClearLog: Sendable {
    private let clears = OSAllocatedUnfairLock(initialState: 0)

    /// The clears so far.
    var count: Int { clears.withLock { $0 } }

    /// Counts one clear.
    func recordClear() {
        clears.withLock { $0 += 1 }
    }
}

@Suite("Periodic clear of the MLX buffer pool")
struct GenerationBufferPoolClearingTests {

    /// The tokens each test decodes: past two whole intervals.
    private static let decodedTokenCount = 600

    /// The token the model samples at each position and the drafter drafts.
    private static let repeatedToken: Int32 = 7

    /// The token the target samples where it disagrees with the drafter.
    private static let disagreeingToken: Int32 = 9

    /// The block size of a speculating iterator. A round that accepts its drafts emits this many
    /// tokens after the prefill token, and this size does not divide an interval. Thus 256 is the
    /// first token of its round and 512 the middle one, and a clear checked only at one end of a
    /// round misses a multiple.
    private static let blockSize = 3

    /// The input of each test: a prompt of three tokens.
    private static func makeInput() -> LMInput {
        LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    }

    private static func parameters() -> GenerateParameters {
        GenerateParameters(maxTokens: decodedTokenCount, temperature: 0)
    }

    /// A model that samples ``repeatedToken`` at every position of the decode.
    private static func makeMain() -> MockMainModel {
        MockMainModel(
            nextLogitTokens: [0, 0]
                + Array(repeating: repeatedToken, count: 2 * decodedTokenCount))
    }

    /// A model whose verify passes alternate: one accepts all its drafts and emits ``blockSize``
    /// tokens, the next rejects its last draft and emits one token fewer.
    ///
    /// The stream emits five tokens for each six rows of the target, and 256 is the first and 512
    /// the middle token of a round that accepts all its drafts.
    private static func makeDisagreeingMain() -> MockMainModel {
        let acceptingRound = Array(repeating: repeatedToken, count: blockSize)
        let rejectingRound = [repeatedToken, disagreeingToken, repeatedToken]
        let verifyRows = (0 ..< decodedTokenCount).flatMap { _ in acceptingRound + rejectingRound }
        return MockMainModel(nextLogitTokens: [0, 0, repeatedToken] + verifyRows)
    }

    /// The indices of the tokens `next` was emitting when a clear happened.
    private static func clearedTokenIndices(
        log: ClearLog, next: () -> Int?
    ) -> [Int] {
        (0 ..< decodedTokenCount).filter { _ in
            let clearsBefore = log.count
            #expect(next() != nil)
            return log.count > clearsBefore
        }
    }

    private static func makeCountingClearing(log: ClearLog) -> GenerationBufferPoolClearing {
        GenerationBufferPoolClearing(clearPool: { log.recordClear() })
    }

    @Test func speculatingIteratorClearsEveryIntervalOfEmittedTokens() throws {
        var iterator = try MTPSpeculativeTokenIterator(
            input: Self.makeInput(),
            mainModel: Self.makeMain(),
            drafter: MockDrafter(draftedTokenValue: Self.repeatedToken),
            parameters: Self.parameters(),
            blockSize: Self.blockSize)
        let log = ClearLog()
        iterator.bufferPoolClearing = Self.makeCountingClearing(log: log)

        let indices = Self.clearedTokenIndices(log: log) { iterator.next() }

        #expect(iterator.passthroughReason == nil)
        #expect(indices == [0, 256, 512])
    }

    @Test func speculatingIteratorDoesNotCountRejectedDrafts() throws {
        var iterator = try MTPSpeculativeTokenIterator(
            input: Self.makeInput(),
            mainModel: Self.makeDisagreeingMain(),
            drafter: MockDrafter(draftedTokenValue: Self.repeatedToken),
            parameters: Self.parameters(),
            blockSize: Self.blockSize)
        let log = ClearLog()
        iterator.bufferPoolClearing = Self.makeCountingClearing(log: log)

        let indices = Self.clearedTokenIndices(log: log) { iterator.next() }

        #expect(iterator.passthroughReason == nil)
        #expect(iterator.acceptedCount < iterator.proposedCount)
        #expect(indices == [0, 256, 512])
    }

    @Test func passthroughIteratorClearsEveryIntervalOfEmittedTokens() throws {
        let main = Self.makeMain()
        main.omitDrafterState = true
        var iterator = try MTPSpeculativeTokenIterator(
            input: Self.makeInput(),
            mainModel: main,
            drafter: MockDrafter(draftedTokenValue: Self.repeatedToken),
            parameters: Self.parameters(),
            blockSize: Self.blockSize)
        let log = ClearLog()
        iterator.bufferPoolClearing = Self.makeCountingClearing(log: log)

        let indices = Self.clearedTokenIndices(log: log) { iterator.next() }

        #expect(iterator.passthroughReason != nil)
        #expect(indices == [0, 256, 512])
    }

    @Test func tokenIteratorClearsEveryIntervalOfEmittedTokens() throws {
        var iterator = try TokenIterator(
            input: Self.makeInput(),
            model: Self.makeMain(),
            parameters: Self.parameters())
        let log = ClearLog()
        iterator.bufferPoolClearing = Self.makeCountingClearing(log: log)

        let indices = Self.clearedTokenIndices(log: log) { iterator.next() }

        #expect(indices == [0, 256, 512])
    }
}
