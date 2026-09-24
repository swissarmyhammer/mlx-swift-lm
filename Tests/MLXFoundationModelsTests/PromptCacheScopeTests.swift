// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Proves that `MLXLanguageModel.promptCacheScope` sets or disables the prompt
/// cache key of an executor pass.
///
/// Each test calls `Executor.respond` directly, on the task that binds the
/// scope and its own store. The scripted model holds one KV cache layer, thus
/// a pass checks a cache in, and the next pass of the same session reuses it.
///
/// The executor is available from iOS 27, macOS 27 and visionOS 27. On an
/// earlier system each test records an issue, thus it fails and does not pass
/// with no assertion.
@Suite("A host sets or disables the prompt cache key of a pass")
struct PromptCacheScopeTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    @Test("a bound session shares one cache across passes with different first entries")
    func aBoundSessionSharesOneCacheAcrossFirstEntries() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectABoundSessionSharesOneCacheAcrossFirstEntries()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("two bound sessions over the same transcript do not share an entry")
    func twoBoundSessionsOverOneTranscriptDoNotShareAnEntry() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectTwoBoundSessionsOverOneTranscriptDoNotShareAnEntry()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("an uncached pass takes no cache and leaves none")
    func anUncachedPassTakesNoCacheAndLeavesNone() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAnUncachedPassTakesNoCacheAndLeavesNone()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("binding nil inside an uncached scope gives the first-entry rule")
    func bindingNilInsideAnUncachedScopeGivesTheFirstEntryRule() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectBindingNilInsideAnUncachedScopeGivesTheFirstEntryRule()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("with nothing bound, the first entry names the session")
    func withNothingBoundTheFirstEntryNamesTheSession() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectWithNothingBoundTheFirstEntryNamesTheSession()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    // MARK: - The checks

    /// Two passes of session A, with different first entries, share one cache:
    /// the second pass starts warm, and the store holds session A alone.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectABoundSessionSharesOneCacheAcrossFirstEntries() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(weights: weights)
        let store = makeStore()

        let first = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1"), scope: .session("A"),
            model: model, inside: store)
        let second = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-2", turns: 2),
            scope: .session("A"),
            model: model, inside: store)

        #expect(first == 0)
        #expect(second > 0, "The second pass of session A must start warm.")
        #expect(await store.peek(key("A", of: model)) != nil)
        #expect(await store.peek(key("entry-1", of: model)) == nil)
        #expect(await store.peek(key("entry-2", of: model)) == nil)
        #expect(await store.retainedSessionCount == 1)
    }

    /// A fork of session A into session B copies the transcript of A, thus it
    /// also copies the first entry. The fork starts cold, although its render
    /// extends the render of A, and the store holds one entry for each session.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectTwoBoundSessionsOverOneTranscriptDoNotShareAnEntry() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(weights: weights)
        let store = makeStore()

        try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1"), scope: .session("A"),
            model: model, inside: store)
        let forked = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1", turns: 2),
            scope: .session("B"),
            model: model, inside: store)

        #expect(forked == 0, "Session B must not reuse the cache of session A.")
        let entryOfA = try #require(await store.peek(key("A", of: model)))
        let entryOfB = try #require(await store.peek(key("B", of: model)))
        #expect(entryOfA !== entryOfB)
        #expect(await store.retainedSessionCount == 2)
    }

    /// A pass with the scope `.uncached` reuses nothing, although the store
    /// holds a cache of its first entry that its render extends, and it leaves
    /// the session count and the byte count of the store as they were.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAnUncachedPassTakesNoCacheAndLeavesNone() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(weights: weights)
        let store = makeStore()
        try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1"), scope: nil,
            model: model, inside: store)
        let sessionsBefore = await store.retainedSessionCount
        let bytesBefore = await store.retainedByteCount

        let reused = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1", turns: 2),
            scope: .uncached,
            model: model, inside: store)

        #expect(reused == 0, "An uncached pass must reuse nothing.")
        #expect(sessionsBefore == 1)
        #expect(await store.retainedSessionCount == sessionsBefore)
        #expect(await store.retainedByteCount == bytesBefore)
        #expect(await store.peek(key("entry-1", of: model)) != nil)
    }

    /// A pass that binds `nil` inside an outer `.uncached` binding has no
    /// scope, thus the first entry names its session: it reuses the cache of
    /// its first entry, and it checks its cache in under that first entry.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectBindingNilInsideAnUncachedScopeGivesTheFirstEntryRule() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(weights: weights)
        let store = makeStore()
        try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1"), scope: nil,
            model: model, inside: store)
        let entryBefore = try #require(await store.peek(key("entry-1", of: model)))

        let reused = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
            try await respond(
                over: ScriptedSessionModel.transcript(firstEntryID: "entry-1", turns: 2),
                scope: nil,
                model: model, inside: store)
        }

        #expect(reused > 0, "A pass that binds nil must reuse the cache of its first entry.")
        let entryAfter = try #require(await store.peek(key("entry-1", of: model)))
        #expect(entryAfter !== entryBefore, "The pass must check its cache in.")
        #expect(await store.retainedSessionCount == 1)
    }

    /// With no scope bound, the first entry names the session: a second pass
    /// with the same first entry starts warm, and another first entry starts
    /// cold.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectWithNothingBoundTheFirstEntryNamesTheSession() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }
        let model = ScriptedSessionModel.make(weights: weights)
        let store = makeStore()

        let first = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1"), scope: nil,
            model: model, inside: store)
        let second = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-1", turns: 2), scope: nil,
            model: model, inside: store)
        let other = try await respond(
            over: ScriptedSessionModel.transcript(firstEntryID: "entry-2", turns: 2), scope: nil,
            model: model, inside: store)

        #expect(first == 0)
        #expect(second > 0, "The second pass of one first entry must start warm.")
        #expect(other == 0, "Another first entry names another session.")
        #expect(await store.peek(key("entry-1", of: model)) != nil)
        #expect(await store.peek(key("entry-2", of: model)) != nil)
    }

    // MARK: - Fixtures

    /// A store in a folder of its own, whose writer writes nothing.
    ///
    /// These tests read the memory tier alone. A writer that writes nothing
    /// thus leaves no file, and two stores never share a file name.
    private func makeStore() -> ExecutorPromptCacheStore {
        ExecutorPromptCacheStore(
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("PromptCacheScopeTests-\(UUID().uuidString)"),
            writer: { _, _ in })
    }

    /// Runs one pass of `model` over `transcript` inside `store`, with `scope`
    /// bound on the task that calls the executor.
    ///
    /// - Parameters:
    ///   - transcript: the transcript of the request.
    ///   - scope: the prompt cache scope the pass binds, or nil to bind no
    ///     scope.
    ///   - model: the model of the pass.
    ///   - store: the prompt cache store the pass binds.
    /// - Returns: the prompt tokens the pass reused.
    /// - Throws: the error of the executor.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @discardableResult
    private func respond(
        over transcript: Transcript, scope: MLXLanguageModel.PromptCacheScope?,
        model: MLXLanguageModel, inside store: ExecutorPromptCacheStore
    ) async throws -> Int {
        try await MLXLanguageModel.$promptCacheScope.withValue(scope) {
            try await ScriptedExecutorPass.run(over: transcript, model: model, inside: store)
        }
    }

    /// The key of `sessionID` for `model`.
    ///
    /// - Parameters:
    ///   - sessionID: the session the key names.
    ///   - model: the model the key names.
    /// - Returns: the key.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func key(_ sessionID: String, of model: MLXLanguageModel) -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: model.modelID, sessionID: sessionID)
    }
}

#endif  // FoundationModelsIntegration
