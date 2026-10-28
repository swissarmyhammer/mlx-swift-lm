// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The MTP drafter continuation as a portable cache value: a deep copy, its byte count, its
/// alignment with a main cache, and its serialization, each checked by continuing a stream on
/// the shared Qwen 3.5 fixture.
@Suite(.serialized)
struct MTPDrafterContinuationTests {

    /// Prompt tokens of the fixture vocabulary.
    private let prompt = [3, 9, 14, 22, 5, 40]

    /// Tokens the first call generates.
    private let firstCallTokenCount = 8

    /// Prompt tokens a second call appends after the sequence the main cache holds.
    private let followUpPrompt = [8, 30, 2]

    /// Tokens a second call generates.
    private let secondCallTokenCount = 8

    /// The drafter identity no drafter of the fixture has.
    private let foreignDrafterID = MTPDrafterID("another drafter")

    /// The layers of a drafter whose configuration differs from the fixture's.
    private let foreignDrafterLayerCount = 2

    /// What a first MTP call leaves for a second call.
    private struct FirstCall {

        /// The main cache after the call.
        let cache: [KVCache]

        /// The tokens `cache` holds.
        let cachedTokens: [Int]

        /// The continuation the call gave back.
        let continuation: MTPDrafterContinuation
    }

    /// What a second call that continues a ``FirstCall`` produced.
    private struct SecondCall {

        /// The generated tokens.
        let tokens: [Int]

        /// Whether the drafter kept the entries of the continuation rather than cold-starting at
        /// the cached position.
        let resumed: Bool

        /// Why the stream stopped speculating, or `nil` when it speculated throughout.
        let passthroughReason: String?
    }

    /// A greedy generation of `maxTokens` tokens.
    private func greedyParameters(maxTokens: Int) -> GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0)
    }

    /// Decodes ``firstCallTokenCount`` greedy MTP tokens after ``prompt`` from a fresh cache.
    private func firstCall(_ fixture: Qwen35MTPFixture) async throws -> FirstCall {
        let cache = try fixture.target.newCache(parameters: nil)
        let (stream, task) = try generateTokensTask(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), cache: cache,
            parameters: greedyParameters(maxTokens: firstCallTokenCount),
            context: fixture.makeModelContext(), mtpDrafter: fixture.drafter,
            blockSize: Qwen35MTPFixture.blockSize)
        let decode = try await Qwen35MTPFixture.collect(stream)
        let continuation = try #require(await task.value)
        let cachedTokenCount = KVCacheStorage(cache, plan: .disabled).processedTokenCount
        return FirstCall(
            cache: cache, cachedTokens: Array((prompt + decode.tokens).prefix(cachedTokenCount)),
            continuation: continuation)
    }

    /// Decodes ``secondCallTokenCount`` greedy MTP tokens after ``followUpPrompt`` on a copy of
    /// the main cache of `first`, carrying `continuation`.
    private func secondCall(
        _ fixture: Qwen35MTPFixture, after first: FirstCall,
        carrying continuation: MTPDrafterContinuation
    ) throws -> SecondCall {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(followUpPrompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter,
            mainCache: first.cache.map { $0.copy() }, mtpContinuation: continuation,
            parameters: greedyParameters(maxTokens: secondCallTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        let primed = try #require(iterator.drafterState)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return SecondCall(
            tokens: tokens, resumed: primed.cacheStartPosition == 0,
            passthroughReason: iterator.passthroughReason)
    }

    /// The tokens plain greedy decode generates after the second call's whole prompt.
    private func plainSecondCallTokens(_ fixture: Qwen35MTPFixture, after first: FirstCall)
        async throws -> [Int]
    {
        try await fixture.decodeGreedily(
            prompt: first.cachedTokens + followUpPrompt, maxTokens: secondCallTokenCount,
            speculative: false
        ).tokens
    }

    /// A Qwen 3.5 MTP drafter whose configuration has more layers than the fixture's drafter.
    private func foreignConfigurationDrafter() throws -> Qwen35MTPDraftModel {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: foreignDrafterLayerCount,
                    hiddenLayers: Qwen35MTPFixture.hiddenLayers,
                    fullAttentionInterval: Qwen35MTPFixture.fullAttentionInterval,
                    hiddenSize: Qwen35MTPFixture.hiddenSize,
                    vocabularySize: Qwen35MTPFixture.vocabularySize,
                    tieWordEmbeddings: false
                ).utf8))
        return Qwen35MTPDraftModel(configuration)
    }

    @Test("a copy and its original each resume a second call with the plain greedy tokens")
    func copyContinuesIndependently() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let plain = try await plainSecondCallTokens(fixture, after: first)
        let copy = first.continuation.copy()

        let original = try secondCall(fixture, after: first, carrying: first.continuation)
        let copied = try secondCall(fixture, after: first, carrying: copy)

        #expect(original.resumed)
        #expect(original.tokens == plain)
        #expect(copied.resumed)
        #expect(copied.passthroughReason == nil)
        #expect(copied.tokens == plain)
    }

    @Test("a round trip through arrays and metadata resumes with the plain greedy tokens")
    func roundTripResumesWithTheSameTokens() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let plain = try await plainSecondCallTokens(fixture, after: first)

        let restored = try MTPDrafterContinuation(
            arrays: try first.continuation.arrays(), metadata: try first.continuation.metadata(),
            drafter: fixture.drafter)
        let second = try secondCall(fixture, after: first, carrying: restored)

        #expect(restored.targetProcessedTokenCount == first.cachedTokens.count)
        #expect(second.resumed)
        #expect(second.passthroughReason == nil)
        #expect(second.tokens == plain)
    }

    @Test(
        "a prompt cache file keeps the continuation, and its restore resumes with the plain tokens")
    func promptCacheFileRoundTripResumesWithTheSameTokens() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let plain = try await plainSecondCallTokens(fixture, after: first)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(
            url: url, cache: first.cache,
            drafterContinuation: try SerializedMTPDrafterContinuation(first.continuation))
        let snapshot = try loadPromptCacheSnapshot(url: url)
        let restored = try #require(snapshot.drafterContinuation).restored(by: fixture.drafter)
        let loaded = FirstCall(
            cache: snapshot.cache, cachedTokens: first.cachedTokens, continuation: restored)
        let second = try secondCall(fixture, after: loaded, carrying: restored)

        #expect(snapshot.cache.count == first.cache.count)
        #expect(second.resumed)
        #expect(second.passthroughReason == nil)
        #expect(second.tokens == plain)
    }

    @Test("a template restore of a prompt cache file keeps the drafter continuation")
    func promptCacheTemplateRestoreKeepsTheContinuation() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(
            url: url, cache: first.cache,
            drafterContinuation: try SerializedMTPDrafterContinuation(first.continuation))
        let snapshot = try loadPromptCacheSnapshot(
            url: url, into: try fixture.target.newCache(parameters: nil))
        let restored = try #require(snapshot.drafterContinuation).restored(by: fixture.drafter)

        #expect(
            restored.isAligned(
                withMainTokenCount: first.cachedTokens.count, drafterID: fixture.drafter.drafterID))
    }

    @Test("a prompt cache file written without a continuation loads with none")
    func promptCacheFileWithoutAContinuationLoadsWithNone() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let url = TemporaryPromptCacheFile.makeURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: first.cache)
        let snapshot = try loadPromptCacheSnapshot(url: url)

        #expect(snapshot.cache.count == first.cache.count)
        #expect(snapshot.drafterContinuation == nil)
    }

    @Test("a serialized continuation keeps only the entries under the reserved prefix")
    func serializedContinuationKeepsOnlyReservedEntries() async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        let foreignKey = "0.0"

        let serialized = SerializedMTPDrafterContinuation(
            arrays: try continuation.arrays().merging([foreignKey: MLXArray(0)]) { kept, _ in kept
            },
            metadata: try continuation.metadata().merging([foreignKey: "value"]) { kept, _ in kept }
        )

        #expect(serialized.arrays[foreignKey] == nil)
        #expect(serialized.metadata[foreignKey] == nil)
        #expect(Set(serialized.arrays.keys) == Set(try continuation.arrays().keys))
        #expect(serialized.metadata == (try continuation.metadata()))
    }

    @Test("every key of the arrays and the metadata carries the reserved prefix")
    func serializedKeysCarryTheReservedPrefix() async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        let keys = Array(try continuation.arrays().keys) + Array(try continuation.metadata().keys)

        #expect(!keys.isEmpty)
        #expect(keys.allSatisfy { $0.hasPrefix(MTPDrafterContinuation.reservedKeyPrefix) })
    }

    @Test("the byte count is the drafter cache buffers plus the boundary hidden row")
    func byteCountCountsCachesAndBoundaryRow() async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        let cacheBytes = continuation.state.cache.reduce(0) { $0 + $1.residentByteCount }

        #expect(cacheBytes > 0)
        #expect(continuation.byteCount == cacheBytes + continuation.targetBoundaryHidden.nbytes)
    }

    @Test("a continuation is aligned with the main token count it was made at, for its drafter")
    func alignedAtItsMainTokenCount() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let count = first.cachedTokens.count
        let drafterID = fixture.drafter.drafterID

        #expect(first.continuation.isAligned(withMainTokenCount: count, drafterID: drafterID))
        #expect(!first.continuation.isAligned(withMainTokenCount: count + 1, drafterID: drafterID))
        #expect(!first.continuation.isAligned(withMainTokenCount: count - 1, drafterID: drafterID))
    }

    @Test("the aligned boundary is the boundary hidden row only at the main token count")
    func alignedBoundaryOnlyAtItsMainTokenCount() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let count = first.cachedTokens.count
        let drafterID = fixture.drafter.drafterID
        let continuation = first.continuation

        let boundary = try #require(
            continuation.alignedBoundary(withMainTokenCount: count, drafterID: drafterID))
        #expect(boundary === continuation.targetBoundaryHidden)
        #expect(
            continuation.alignedBoundary(withMainTokenCount: count + 1, drafterID: drafterID)
                == nil)
        #expect(
            continuation.alignedBoundary(withMainTokenCount: count, drafterID: foreignDrafterID)
                == nil)
    }

    @Test("a continuation is not aligned for another drafter")
    func notAlignedForAnotherDrafter() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)

        #expect(
            !first.continuation.isAligned(
                withMainTokenCount: first.cachedTokens.count, drafterID: foreignDrafterID))
    }

    @Test("a stream cold-starts the drafter on a continuation of another drafter")
    func continuationOfAnotherDrafterColdStarts() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let plain = try await plainSecondCallTokens(fixture, after: first)
        let foreign = MTPDrafterContinuation(
            state: first.continuation.copy().state,
            targetBoundaryHidden: first.continuation.targetBoundaryHidden,
            targetProcessedTokenCount: first.continuation.targetProcessedTokenCount,
            drafterID: foreignDrafterID)

        let second = try secondCall(fixture, after: first, carrying: foreign)

        #expect(!second.resumed)
        #expect(second.passthroughReason == nil)
        #expect(second.tokens == plain)
    }

    @Test("a stream ignores the drafter caches of a continuation of a differently built drafter")
    func continuationOfAnotherDrafterLayoutIsIgnored() async throws {
        let fixture = try Qwen35MTPFixture()
        let first = try await firstCall(fixture)
        let plain = try await plainSecondCallTokens(fixture, after: first)
        let foreignDrafter = try foreignConfigurationDrafter()
        let foreign = MTPDrafterContinuation(
            state: foreignDrafter.makeState(parameters: nil),
            targetBoundaryHidden: first.continuation.targetBoundaryHidden,
            targetProcessedTokenCount: first.continuation.targetProcessedTokenCount,
            drafterID: foreignDrafter.drafterID)

        let second = try secondCall(fixture, after: first, carrying: foreign)

        #expect(second.passthroughReason == nil)
        #expect(second.tokens == plain)
    }

    @Test("the drafter identity follows the configuration, not the weights")
    func drafterIDFollowsTheConfiguration() throws {
        let copying = try Qwen35MTPFixture(drafter: .copiesTarget)
        let independent = try Qwen35MTPFixture(drafter: .independent)

        #expect(copying.drafter.drafterID == independent.drafter.drafterID)
        #expect(copying.drafter.drafterID != (try foreignConfigurationDrafter()).drafterID)
    }

    @Test("the text and the vision-language drafters of one configuration differ in identity")
    func drafterIDNamesTheDrafterType() throws {
        let text = try Qwen35MTPFixture()
        let visionLanguage = try Qwen35VLMMTPFixture()

        #expect(text.drafter.drafterID != visionLanguage.drafter.drafterID)
        #expect(visionLanguage.drafter.drafterID == (try Qwen35VLMMTPFixture()).drafter.drafterID)
    }

    @Test("a restore into a drafter of another configuration throws")
    func restoreIntoAnotherDrafterThrows() async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        let arrays = try continuation.arrays()
        let metadata = try continuation.metadata()
        let foreign = try foreignConfigurationDrafter()

        #expect(throws: KVCacheError.self) {
            try MTPDrafterContinuation(arrays: arrays, metadata: metadata, drafter: foreign)
        }
    }

    @Test(
        "a restore of tampered arrays or metadata throws",
        arguments: MTPDrafterContinuationTamper.allCases)
    func restoreOfTamperedValuesThrows(tamper: MTPDrafterContinuationTamper) async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        var arrays = try continuation.arrays()
        var metadata = try continuation.metadata()
        try tamper.apply(arrays: &arrays, metadata: &metadata)

        #expect(throws: KVCacheError.self) {
            try MTPDrafterContinuation(arrays: arrays, metadata: metadata, drafter: fixture.drafter)
        }
    }

    @Test("a restore keeps a seed proposal of the boundary hidden row's batch and width")
    func restoreKeepsAWellFormedSeedProposal() async throws {
        let fixture = try Qwen35MTPFixture()
        let continuation = try await firstCall(fixture).continuation
        var arrays = try continuation.arrays()
        try MTPDrafterContinuationTamper.writeSeed(into: &arrays) { batch, width in
            ([batch, 1], [batch, 1, width])
        }

        let restored = try MTPDrafterContinuation(
            arrays: arrays, metadata: try continuation.metadata(), drafter: fixture.drafter)

        typealias Key = MTPDrafterContinuation.SerializedKey
        #expect(restored.state.seedToken?.shape == arrays[Key.seedToken]?.shape)
        #expect(restored.state.seedHidden?.shape == arrays[Key.seedHidden]?.shape)
    }
}

/// One change to the serialized form of a continuation that its restore must refuse.
enum MTPDrafterContinuationTamper: String, CaseIterable, CustomTestStringConvertible {

    /// A format version this reader does not know.
    case formatVersion

    /// The identity of another drafter.
    case drafterID

    /// A next position one past the entries the drafter caches hold.
    case nextPosition

    /// A first cache position after the next position.
    case cacheStartPosition

    /// A rotating cache where the drafter makes a simple one.
    case cacheKind

    /// No boundary hidden row.
    case boundaryHidden

    /// A boundary hidden row one column wider than the drafter reads.
    case boundaryHiddenWidth

    /// A seed proposal whose hidden row is one column wider than the drafter makes.
    case seedHiddenWidth

    /// A seed proposal whose hidden row has one more batch row than the boundary hidden row.
    case seedHiddenBatch

    /// A seed proposal whose token has one more batch row than the boundary hidden row.
    case seedTokenBatch

    /// A seed proposal of more than one token for each batch row.
    case seedTokenCount

    /// Drafter caches with one more batch row than the boundary hidden row.
    case cacheBatch

    var testDescription: String { rawValue }

    /// The error of a tamper whose target is missing from the serialized form.
    struct MissingTarget: Error {}

    /// The shapes of a seed proposal, from the batch and the width of the boundary hidden row.
    typealias SeedShapes = (_ batch: Int, _ width: Int) -> (token: [Int], hidden: [Int])

    /// The axis of the hidden columns of a hidden row.
    private static let hiddenAxis = -1

    /// A token count for each batch row that a seed proposal never has.
    private static let excessSeedTokenCount = 2

    /// Applies the change to the serialized form.
    ///
    /// - Parameters:
    ///   - arrays: The serialized arrays.
    ///   - metadata: The serialized metadata.
    /// - Throws: ``MissingTarget`` when the serialized form holds no value to change.
    func apply(arrays: inout [String: MLXArray], metadata: inout [String: String]) throws {
        typealias Key = MTPDrafterContinuation.SerializedKey
        switch self {
        case .formatVersion:
            metadata[Key.formatVersion] = "0"
        case .drafterID:
            metadata[Key.drafterID] = "another drafter"
        case .nextPosition:
            try offset(Key.nextPosition, in: &metadata)
        case .cacheStartPosition:
            let next = try integer(Key.nextPosition, in: metadata)
            metadata[Key.cacheStartPosition] = String(next + 1)
        case .cacheKind:
            let simpleClassName = try #require(KVCacheSimple.serializedClassName)
            let rotatingClassName = try #require(RotatingKVCache.serializedClassName)
            let simpleCacheKeys = metadata.filter { $0.value == simpleClassName }.keys
            guard !simpleCacheKeys.isEmpty else { throw MissingTarget() }
            for key in simpleCacheKeys {
                metadata[key] = rotatingClassName
            }
        case .boundaryHidden:
            guard arrays.removeValue(forKey: Key.boundaryHidden) != nil else {
                throw MissingTarget()
            }
        case .boundaryHiddenWidth:
            let row = try Self.boundaryRow(of: arrays)
            arrays[Key.boundaryHidden] = MLXArray.zeros(
                [row.dim(MTPDrafterContinuation.batchAxis), 1, row.dim(Self.hiddenAxis) + 1],
                dtype: row.dtype)
        case .seedHiddenWidth:
            try Self.writeSeed(into: &arrays) { batch, width in
                ([batch, 1], [batch, 1, width + 1])
            }
        case .seedHiddenBatch:
            try Self.writeSeed(into: &arrays) { batch, width in
                ([batch, 1], [batch + 1, 1, width])
            }
        case .seedTokenBatch:
            try Self.writeSeed(into: &arrays) { batch, width in
                ([batch + 1, 1], [batch, 1, width])
            }
        case .seedTokenCount:
            try Self.writeSeed(into: &arrays) { batch, width in
                ([batch, Self.excessSeedTokenCount], [batch, 1, width])
            }
        case .cacheBatch:
            let cacheKeys = arrays.keys.filter { $0.hasPrefix(Key.cachePrefix) }
            guard !cacheKeys.isEmpty else { throw MissingTarget() }
            for key in cacheKeys {
                arrays[key] = arrays[key].map {
                    concatenated([$0, $0], axis: MTPDrafterContinuation.batchAxis)
                }
            }
        }
    }

    /// Writes a zero seed proposal of the shapes `shapes` gives for the boundary hidden row.
    ///
    /// - Parameters:
    ///   - arrays: The serialized arrays.
    ///   - shapes: The shapes of the seed token and the seed hidden row.
    /// - Throws: ``MissingTarget`` when the serialized form holds no boundary hidden row.
    static func writeSeed(into arrays: inout [String: MLXArray], shapes: SeedShapes) throws {
        typealias Key = MTPDrafterContinuation.SerializedKey
        let row = try boundaryRow(of: arrays)
        let (token, hidden) = shapes(
            row.dim(MTPDrafterContinuation.batchAxis), row.dim(hiddenAxis))
        arrays[Key.seedToken] = MLXArray.zeros(token, type: Int32.self)
        arrays[Key.seedHidden] = MLXArray.zeros(hidden, dtype: row.dtype)
    }

    /// The boundary hidden row of the serialized arrays.
    private static func boundaryRow(of arrays: [String: MLXArray]) throws -> MLXArray {
        guard let row = arrays[MTPDrafterContinuation.SerializedKey.boundaryHidden] else {
            throw MissingTarget()
        }
        return row
    }

    /// Adds one to the integer metadata value of `key`.
    private func offset(_ key: String, in metadata: inout [String: String]) throws {
        metadata[key] = String(try integer(key, in: metadata) + 1)
    }

    /// The integer metadata value of `key`.
    private func integer(_ key: String, in metadata: [String: String]) throws -> Int {
        guard let value = metadata[key].flatMap({ Int($0) }) else { throw MissingTarget() }
        return value
    }
}
