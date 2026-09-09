// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

/// The generation loop reports the stop token it fed into the caches.
///
/// `TokenIterator` feeds a token before it answers it, thus the stop token
/// stands in the caches when the loop reads it. The loop leaves that token out
/// of `generationTokenCount` and names it in `stopTokenFedToCache`, thus a
/// caller that keeps a token ledger, or reports an output count beside that
/// ledger, can count it.
@Suite("The generation loop and the stop token")
struct GenerateLoopStopTokenTests {

    /// The stop token of the fixture.
    private static let stopToken = 5

    /// The tokens the fixture answers before the stop token.
    private static let answeredTokens = [11, 22]

    /// An iterator that answers `tokens` in order, then nothing.
    private struct ScriptedIterator: TokenIteratorProtocol {
        let tokens: [Int]
        var index = 0
        var tokenCount = 0
        let maxTokens: Int? = nil
        let promptPrefillTime: TimeInterval = 0

        mutating func next() -> Int? {
            guard index < tokens.count else { return nil }
            defer {
                index += 1
                tokenCount += 1
            }
            return tokens[index]
        }
    }

    @Test("names the stop token it fed and leaves it out of the generated count")
    func namesTheStopTokenItFedAndLeavesItOutOfTheGeneratedCount() async throws {
        let configuration = ModelConfiguration(id: "test/stop", eosTokenIds: [Self.stopToken])
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: configuration,
            tokenizer: MarkerTokenizer(identifierOfMarker: [:]),
            iterator: ScriptedIterator(tokens: Self.answeredTokens + [Self.stopToken]))

        var info: GenerateCompletionInfo?
        for await generation in stream {
            if case .info(let completion) = generation {
                info = completion
            }
        }
        let recorded = await task.value

        #expect(recorded == Self.answeredTokens + [Self.stopToken])
        #expect(info?.generationTokenCount == Self.answeredTokens.count)
        #expect(info?.stopTokenFedToCache == Self.stopToken)
    }

    @Test("names no stop token when the iterator ends without one")
    func namesNoStopTokenWhenTheIteratorEndsWithoutOne() async throws {
        let configuration = ModelConfiguration(id: "test/stop", eosTokenIds: [Self.stopToken])
        let (stream, task) = generateTaskRecordingTokens(
            promptTokenCount: 1, modelConfiguration: configuration,
            tokenizer: MarkerTokenizer(identifierOfMarker: [:]),
            iterator: ScriptedIterator(tokens: Self.answeredTokens))

        var info: GenerateCompletionInfo?
        for await generation in stream {
            if case .info(let completion) = generation {
                info = completion
            }
        }
        _ = await task.value

        #expect(info?.generationTokenCount == Self.answeredTokens.count)
        #expect(info?.stopTokenFedToCache == nil)
    }
}
