// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXLMCommon

/// A spill file of the executor's prompt cache keeps the drafter state of a stream whose drafter
/// attends to a window (``GenerateParameters/mtpDrafterAttentionWindow``): a turn of the same
/// window reads it back as a ring of that window, aligned with the ledger, and a turn of another
/// window, or of the whole context, reads the main caches alone and starts its drafter cold.
@Suite(.serialized)
struct ExecutorPromptCacheDrafterWindowTests {

    /// The drafter window of the spilled stream: shorter than the tokens it holds, so its rings
    /// wrap.
    private static let window = 12

    /// A drafter window other than ``window``.
    private static let otherWindow = 20

    /// The prompt of the spilled stream.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens the spilled stream generates.
    private static let generatedTokenCount = 48

    /// The session of the spill file.
    private static let key = ExecutorPromptCacheKey(
        modelID: "tiny-qwen35-mtp", sessionID: "executor-prompt-cache-drafter-window")

    /// The greedy parameters of a stream whose drafter attends to `window` tokens, or to the
    /// whole context when `window` is `nil`.
    private static func parameters(window: Int?) throws -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: generatedTokenCount, temperature: 0)
        parameters.mtpDrafterAttentionWindow = try window.map {
            try KVCacheConfiguration.Capacity(maxTokens: $0, preservedPrefixTokens: 0)
        }
        return parameters
    }

    /// Decodes ``prompt`` with a drafter of ``window`` and spills the cache the stream left, with
    /// its drafter state, to a new file.
    ///
    /// - Parameter fixture: The fixture that decodes.
    /// - Returns: The file, which the caller removes.
    private func spillWindowedStream(_ fixture: Qwen35MTPFixture) throws -> URL {
        var mtp = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter,
            parameters: try Self.parameters(window: Self.window),
            blockSize: Qwen35MTPFixture.blockSize)
        var generated: [Int] = []
        while let token = mtp.next() {
            generated.append(token)
        }
        mtp.finalizeGeneration()
        let drafterState = try #require(mtp.mtpDrafterContinuation, "\(mtp.passthroughReason ?? "")")
        let ledger = Array((Self.prompt + generated).prefix(drafterState.targetProcessedTokenCount))
        let entry = ExecutorPromptCacheEntry(
            caches: mtp.mainCache, tokens: ledger, renderTokens: Self.prompt,
            drafterState: drafterState)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ExecutorPromptCacheDrafterWindowTests-\(UUID().uuidString)")
            .appendingPathExtension("safetensors")
        try ExecutorPromptCacheFile.write(
            try ExecutorPromptCacheFile.prepare(entry, key: Self.key), to: url)
        return url
    }

    /// Reads the spill file at `url` for a turn whose drafter attends to `window` tokens.
    ///
    /// - Parameters:
    ///   - url: The spill file.
    ///   - window: The drafter window of the turn, or `nil` for the whole context.
    ///   - fixture: The fixture whose target and drafter read the file.
    /// - Returns: The entry, and the log lines of a drafter state that did not read back.
    private func read(
        _ url: URL, window: Int?, fixture: Qwen35MTPFixture
    ) throws -> (entry: ExecutorPromptCacheEntry, report: [String]) {
        let parameters = try Self.parameters(window: window)
        var report: [String] = []
        let entry = try ExecutorPromptCacheFile.read(
            from: url, key: Self.key, templates: try fixture.target.newCache(parameters: parameters),
            drafter: fixture.drafter, parameters: parameters, report: { report.append($0) })
        return (entry, report)
    }

    @Test("a spilled windowed drafter state reads back as rings of its window, aligned with the ledger")
    func windowedDrafterStateReadsBack() throws {
        let fixture = try Qwen35MTPFixture()
        let url = try spillWindowedStream(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        let (entry, report) = try read(url, window: Self.window, fixture: fixture)

        let drafterState = try #require(entry.drafterState)
        #expect(
            drafterState.isAligned(
                withMainTokenCount: entry.tokens.count, drafterID: fixture.drafter.drafterID))
        #expect(
            drafterState.state.cache.allSatisfy {
                ($0 as? RotatingKVCache)?.maxSize == Self.window
            })
        #expect(report.isEmpty)
    }

    @Test(
        "a spilled windowed drafter state does not read back for a turn of another window or of the whole context",
        arguments: [Self.otherWindow, nil] as [Int?])
    func windowedDrafterStateStartsAnotherTurnCold(window: Int?) throws {
        let fixture = try Qwen35MTPFixture()
        let url = try spillWindowedStream(fixture)
        defer { try? FileManager.default.removeItem(at: url) }

        let (entry, report) = try read(url, window: window, fixture: fixture)

        #expect(entry.drafterState == nil)
        #expect(report.count == 1)
    }
}

#endif
