// Copyright © 2025 Apple Inc.

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration
import MLXGuidedGeneration
#endif

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("MLXLanguageModel initialization")
struct MLXLanguageModelInitTests {

    @Test("modelID returns configuration.name")
    func identifier() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = MLXLanguageModel(
            configuration: ModelConfiguration(id: "mlx-community/Qwen3-4B-4bit"),
            capabilities: [.reasoning],
            weightsLocation: { _ in URL(fileURLWithPath: "/tmp") },
            load: { configuration, progress in
                try await loadModelContainer(
                    from: StubDownloader(), using: StubTokenizerLoader(),
                    configuration: configuration, progressHandler: progress)
            }
        )
        #expect(model.modelID == "mlx-community/Qwen3-4B-4bit")
    }
}

// MARK: - Revision identity

/// Counts the loads a stub container loader serves, so a test can tell a
/// cache hit from a fresh load.
private actor LoadCounter {
    /// The number of loads served so far.
    private(set) var count = 0

    /// Records one load.
    func increment() { count += 1 }
}

// Nested under the serialized `FoundationModelsCacheTests` parent (declared in
// ModelCacheEvictionTests.swift): `MLXLanguageModel` holds one process-global
// `static let cache`, and these tests read and evict entries in it.
extension FoundationModelsCacheTests {

    @Suite("MLXLanguageModel revision identity")
    struct RevisionIdentity {

        /// The one repository the revision tests share.
        private static let repositoryID = "org/repo"

        /// The first of the two revisions the tests load.
        private static let revisionA = "a"

        /// The second of the two revisions the tests load.
        private static let revisionB = "b"

        /// A weights location that holds no `config.json`.
        private static let missingWeights = URL(fileURLWithPath: "/no/such/path")

        /// The number of loads two distinct revisions cost.
        private static let loadsForTwoRevisions = 2

        /// The number of loads after one evicted revision loads again.
        private static let loadsAfterReload = 3

        /// Makes a model for ``repositoryID`` at `revision` whose loader
        /// counts on `counter` and serves a scripted container: no weights.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private static func makeModel(
            revision: String, counter: LoadCounter
        ) -> MLXLanguageModel {
            MLXLanguageModel(
                configuration: ModelConfiguration(id: repositoryID, revision: revision),
                capabilities: [],
                weightsLocation: { _ in missingWeights },
                load: { configuration, _ in
                    await counter.increment()
                    return makeScriptedContainer(modelID: configuration.name, rounds: [])
                })
        }

        /// Makes a model over `configuration` that never loads: a test reads
        /// only its identity and its on-disk check. `weightsLocation` gets
        /// ``missingWeights`` when the test does not care about the disk.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private static func makeUnloadedModel(
            configuration: ModelConfiguration,
            weightsLocation: @escaping @Sendable (String) -> URL = { _ in missingWeights }
        ) -> MLXLanguageModel {
            MLXLanguageModel(
                configuration: configuration,
                capabilities: [],
                weightsLocation: weightsLocation,
                load: stubLoad())
        }

        /// Removes both revisions from the shared cache, so a test starts
        /// and ends with no entry of its own.
        @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
        private static func evictBoth(_ first: MLXLanguageModel, _ second: MLXLanguageModel) async {
            await first.evict()
            await second.evict()
        }

        @Test("modelID carries the revision when it is not main")
        func modelIDCarriesRevision() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let counter = LoadCounter()
            let modelA = Self.makeModel(revision: Self.revisionA, counter: counter)
            let modelB = Self.makeModel(revision: Self.revisionB, counter: counter)

            #expect(modelA.modelID == "org/repo@a")
            #expect(modelB.modelID == "org/repo@b")
            #expect(modelA.modelID != modelB.modelID)
            #expect(modelA.configuration.name == Self.repositoryID)
            #expect(modelB.configuration.name == Self.repositoryID)
        }

        @Test("modelID is configuration.name at revision main")
        func modelIDAtMainIsName() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let model = Self.makeUnloadedModel(
                configuration: ModelConfiguration(id: Self.repositoryID, revision: "main"))
            #expect(model.modelID == Self.repositoryID)
        }

        @Test("modelID is configuration.name for a directory")
        func modelIDForDirectoryIsName() {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let directory = URL(fileURLWithPath: "/models/org/repo")
            let model = Self.makeUnloadedModel(
                configuration: ModelConfiguration(directory: directory))
            #expect(model.modelID == model.configuration.name)
            #expect(model.modelID == "org/repo")
        }

        @Test("two revisions of one id load two containers")
        func twoRevisionsLoadTwice() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let counter = LoadCounter()
            let modelA = Self.makeModel(revision: Self.revisionA, counter: counter)
            let modelB = Self.makeModel(revision: Self.revisionB, counter: counter)
            await Self.evictBoth(modelA, modelB)

            _ = try await modelA.loadContainer()
            _ = try await modelB.loadContainer()
            #expect(await counter.count == Self.loadsForTwoRevisions)

            // A second call for either revision is a cache hit.
            _ = try await modelA.loadContainer()
            _ = try await modelB.loadContainer()
            #expect(await counter.count == Self.loadsForTwoRevisions)

            await Self.evictBoth(modelA, modelB)
        }

        @Test("evict() on one revision leaves the other revision cached")
        func evictIsPerRevision() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let counter = LoadCounter()
            let modelA = Self.makeModel(revision: Self.revisionA, counter: counter)
            let modelB = Self.makeModel(revision: Self.revisionB, counter: counter)
            await Self.evictBoth(modelA, modelB)
            _ = try await modelA.loadContainer()
            _ = try await modelB.loadContainer()

            await modelA.evict()

            _ = try await modelB.loadContainer()
            #expect(
                await counter.count == Self.loadsForTwoRevisions,
                "evict() on revision a must leave revision b cached")
            _ = try await modelA.loadContainer()
            #expect(
                await counter.count == Self.loadsAfterReload,
                "evict() on revision a must make revision a load again")

            await Self.evictBoth(modelA, modelB)
        }

        @Test("modelExistsOnDisk() resolves through configuration.name")
        func modelExistsOnDiskUsesName() throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

            let present = try makeScriptedWeightsDirectory()
            defer { try? FileManager.default.removeItem(at: present) }
            let model = Self.makeUnloadedModel(
                configuration: ModelConfiguration(id: Self.repositoryID, revision: Self.revisionA),
                weightsLocation: { id in
                    id == Self.repositoryID ? present : Self.missingWeights
                })

            #expect(model.modelID != model.configuration.name)
            #expect(model.modelExistsOnDisk())
        }
    }
}

// MARK: - Test Stubs

/// Minimal `Downloader` conformance. The tests in this suite only verify
/// MLXLanguageModel's construction surface; no download is actually invoked.
private struct StubDownloader: Downloader {
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        URL(fileURLWithPath: "/tmp/\(id)")
    }
}

/// Minimal `TokenizerLoader` conformance. As above, never invoked here.
private struct StubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        StubTokenizer()
    }
}

/// Empty `Tokenizer` conformance returned by `StubTokenizerLoader.load`.
/// All operations no-op or return empty results -- this exists only so the
/// loader has something to hand back.
private struct StubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

// MARK: - Temperature plumbing

/// Pure-function tests for the `Double?` (FoundationModels) →
/// `Float?` (MLXLMCommon `GenerateParameters.temperature`) translation
/// done by the unconstrained-generation path. Verifies the clamp
/// semantics that prevent negative sampling temperatures from landing
/// in `CategoricalSampler` and producing inverted distributions.
@Suite("Temperature plumbing")
struct TemperaturePlumbingTests {

    @Test("nil temperature returns nil so the sampler default is used")
    func nilPassesThrough() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.clampedTemperature(nil) == nil)
    }

    @Test("zero passes through unchanged — greedy via ArgMaxSampler")
    func zeroPassesThrough() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.clampedTemperature(0) == 0)
    }

    @Test("positive value passes through unchanged")
    func positivePassesThrough() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.clampedTemperature(0.7) == Float(0.7))
    }

    @Test("negative value clamps to zero")
    func negativeClampsToZero() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(MLXLanguageModel.Executor.clampedTemperature(-0.5) == 0)
    }

    @Test("Double precision narrows to Float without surprise")
    func doubleNarrowsToFloat() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // Sanity check: 0.1 in Double rounds slightly differently than 0.1 in
        // Float. The helper's contract is `Float(max(0, value))`, so we assert
        // exactly that, not arbitrary equality.
        #expect(MLXLanguageModel.Executor.clampedTemperature(0.1) == Float(0.1))
    }
}

// MARK: - Typed error mapping

/// Pure-function tests for the `GrammarError → Error` translation in
/// `Executor.mapGrammarError(_:)`. Verifies that the one xgrammar case where
/// user-fault is provable (`invalidJSONSchema`) maps to the typed
/// `LanguageModelError.unsupportedGenerationGuide`, and everything else
/// passes through untyped so internal-shim failures don't masquerade as
/// developer mistakes.
@Suite("GrammarError typed mapping")
struct GrammarErrorMappingTests {

    @Test("invalidJSONSchema maps to LanguageModelError.unsupportedGenerationGuide")
    func invalidJSONSchemaMapsToTypedError() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let mapped = MLXLanguageModel.Executor.mapGrammarError(
            .invalidJSONSchema(
                "xgrammar rejected the schema: top-level type must be a string")
        )

        guard case LanguageModelError.unsupportedGenerationGuide(let payload) = mapped
        else {
            Issue.record(
                "Expected LanguageModelError.unsupportedGenerationGuide, got \(type(of: mapped)): \(mapped)"
            )
            return
        }
        #expect(
            payload.schemaName == nil,
            "We can't recover the schema name from the xgrammar error path")
        #expect(
            payload.debugDescription
                == "xgrammar rejected the schema: top-level type must be a string",
            "Provider's raw error message should pass through verbatim into debugDescription"
        )
    }

    @Test("constraintCompilationFailed passes through unchanged (origin is ambiguous)")
    func constraintCompilationFailedPassesThrough() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let original = GrammarError.constraintCompilationFailed("matcher init failed")
        let mapped = MLXLanguageModel.Executor.mapGrammarError(original)

        guard case GrammarError.constraintCompilationFailed(let msg) = mapped else {
            Issue.record(
                "Expected GrammarError.constraintCompilationFailed unchanged, got \(type(of: mapped)): \(mapped)"
            )
            return
        }
        #expect(msg == "matcher init failed")
    }

    @Test("tokenizerCreationFailed passes through unchanged (internal shim failure)")
    func tokenizerCreationFailedPassesThrough() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let original = GrammarError.tokenizerCreationFailed("vocab extraction failed")
        let mapped = MLXLanguageModel.Executor.mapGrammarError(original)

        guard case GrammarError.tokenizerCreationFailed(let msg) = mapped else {
            Issue.record(
                "Expected GrammarError.tokenizerCreationFailed unchanged, got \(type(of: mapped)): \(mapped)"
            )
            return
        }
        #expect(msg == "vocab extraction failed")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
