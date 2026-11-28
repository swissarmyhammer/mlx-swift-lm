// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A guided pass -- a schema pass, or a required tool pass -- takes the prompt cache its session
/// carries for the common prefix, as a plain pass does, and checks its ledger in.
///
/// These passes run the executor over a scripted model with one KV cache layer. No weights are
/// needed. The first turn of each session is a plain turn that checks its cache in.
@Suite("A guided pass takes the carried prompt cache and checks its ledger in")
struct GuidedPassPromptCacheTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    @Test("a schema pass after a plain turn reuses the cache of that turn")
    func aSchemaPassAfterAPlainTurnReusesItsCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.jsonString])
            defer { session.remove() }

            let guided = try await session.respondWithSchema(
                turns: GuidedScriptedSession.secondTurnCount, includesSchemaInPrompt: false)

            #expect(guided.reusedTokenCount > 0)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a schema pass checks its ledger in")
    func aSchemaPassChecksItsLedgerIn() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.jsonString])
            defer { session.remove() }

            _ = try await session.respondWithSchema(turns: GuidedScriptedSession.secondTurnCount)

            #expect(await session.store.peek(session.key) != nil)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a plain turn after a schema pass reuses the cache of the schema pass")
    func aPlainTurnAfterASchemaPassReusesItsCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [
                    GuidedScriptedSession.jsonString, ScriptedSessionModel.scriptedResponse,
                ])
            defer { session.remove() }
            let guided = try await session.respondWithSchema(
                turns: GuidedScriptedSession.secondTurnCount, includesSchemaInPrompt: false)

            let plain = try await session.respond(turns: GuidedScriptedSession.thirdTurnCount)

            #expect(plain.reusedTokenCount >= guided.promptTokenCount)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a second schema pass reuses the cache of the first")
    func aSecondSchemaPassReusesTheCacheOfTheFirst() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.jsonString, GuidedScriptedSession.jsonString])
            defer { session.remove() }
            let first = try await session.respondWithSchema(
                turns: GuidedScriptedSession.secondTurnCount)

            let second = try await session.respondWithSchema(
                turns: GuidedScriptedSession.thirdTurnCount)

            #expect(second.reusedTokenCount >= first.promptTokenCount)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a required tool pass after a plain turn reuses the cache of that turn")
    func aRequiredToolPassAfterAPlainTurnReusesItsCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.toolCallScript])
            defer { session.remove() }

            let guided = try await session.respondRequiringTool(
                turns: GuidedScriptedSession.secondTurnCount)

            #expect(guided.toolCallNames == [GuidedScriptedSession.toolName])
            #expect(guided.reusedTokenCount > 0)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a required tool pass reads the cache its session spilled to disk")
    func aRequiredToolPassReadsASpilledCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.toolCallScript], budget: 0)
            defer { session.remove() }

            let guided = try await session.respondRequiringTool(
                turns: GuidedScriptedSession.secondTurnCount)

            #expect(guided.reusedTokenCount > 0)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("the copy of a guided session reuses the copied prefix on its next turn")
    func theCopyOfAGuidedSessionReusesTheCopiedPrefix() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let session = try await GuidedScriptedSession.make(
                laterScripts: [GuidedScriptedSession.jsonString, GuidedScriptedSession.jsonString])
            defer { session.remove() }
            let source = try await session.respondWithSchema(
                turns: GuidedScriptedSession.secondTurnCount)
            _ = await session.copy(to: GuidedScriptedSession.copySessionID)

            let copy = try await session.respondWithSchema(
                turns: GuidedScriptedSession.thirdTurnCount,
                sessionID: GuidedScriptedSession.copySessionID)

            #expect(copy.reusedTokenCount >= source.promptTokenCount)
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }
}

// MARK: - A scripted session whose later passes are guided

/// A scripted session whose first turn checked a prompt cache in. Its later passes are schema
/// passes, required tool passes, or plain turns.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct GuidedScriptedSession {

    /// The folders, the store and the model of the session.
    let session: ScriptedSpoolSession

    /// A memory budget that keeps every cache of the session in memory.
    static let memoryBudgetBytes = 1 << 30

    /// The first entry of the session.
    static let sessionID = "guided-pass-prompt-cache"

    /// The session a copy of the session's cache makes.
    static let copySessionID = "guided-pass-prompt-cache-copy"

    /// The number of prompts of the pass after the first turn.
    static let secondTurnCount = 2

    /// The number of prompts of the pass after that one.
    static let thirdTurnCount = 3

    /// The name of the tool of the required tool pass.
    static let toolName = "lookup"

    /// The tool of the required tool pass. Its argument is one string.
    static let tool = Transcript.ToolDefinition(
        name: toolName, description: "Looks a word up.", parameters: String.generationSchema)

    /// A JSON string, which a string schema accepts. It is the text of a schema pass and the
    /// argument of the call of a required tool pass.
    static let jsonString = "\"A\""

    /// A call of ``tool`` with ``jsonString`` as its argument, in the bare form that the grammar
    /// of the required tool mode accepts.
    static let toolCallScript = "{\"name\": \"\(toolName)\", \"arguments\": \(jsonString)}"

    /// The store of the session.
    var store: ExecutorPromptCacheStore { session.store }

    /// The key of the session.
    var key: ExecutorPromptCacheKey { session.key }

    /// Runs the plain first turn of a new session, and waits until its cache is in the store.
    ///
    /// - Parameters:
    ///   - laterScripts: The text of each pass after the first turn, in order.
    ///   - budget: The memory budget of the store in bytes. Zero spills the cache of the first
    ///     turn to disk.
    /// - Returns: The session.
    /// - Throws: The error of the first turn.
    static func make(
        laterScripts: [String], budget: Int = memoryBudgetBytes
    ) async throws -> GuidedScriptedSession {
        GuidedScriptedSession(
            session: try await ScriptedSpoolSession.make(sessionID: sessionID, budget: budget) {
                ScriptedSessionModel.make(
                    weights: $0, scripts: [ScriptedSessionModel.scriptedResponse] + laterScripts)
            })
    }

    /// Runs one plain pass of the session.
    ///
    /// - Parameter turns: The number of prompts of the transcript of the pass.
    /// - Returns: What the pass streamed.
    /// - Throws: The error of the executor.
    func respond(turns: Int) async throws -> ScriptedPassResult {
        try await respond(to: makeExecutorRequest(transcript: transcript(turns: turns)))
    }

    /// Runs one schema pass of a string schema.
    ///
    /// - Parameters:
    ///   - turns: The number of prompts of the transcript of the pass.
    ///   - includesSchemaInPrompt: Whether the prompt carries the schema. A prompt without it
    ///     renders as a plain turn of the same transcript renders.
    ///   - sessionID: The first entry of the transcript, which names the session.
    /// - Returns: What the pass streamed.
    /// - Throws: The error of the executor.
    func respondWithSchema(
        turns: Int, includesSchemaInPrompt: Bool = true, sessionID: String = sessionID
    ) async throws -> ScriptedPassResult {
        try await respond(
            to: makeExecutorRequest(
                transcript: transcript(turns: turns, sessionID: sessionID),
                schema: String.generationSchema,
                contextOptions: ContextOptions(includeSchemaInPrompt: includesSchemaInPrompt)))
    }

    /// Runs one required tool pass of ``tool``.
    ///
    /// - Parameter turns: The number of prompts of the transcript of the pass.
    /// - Returns: What the pass streamed.
    /// - Throws: The error of the executor.
    func respondRequiringTool(turns: Int) async throws -> ScriptedPassResult {
        try await respond(
            to: makeExecutorRequest(
                transcript: transcript(turns: turns), enabledTools: [Self.tool],
                generationOptions: GenerationOptions(toolCallingMode: .required)))
    }

    /// Copies the cache of the session to `destination` through the public API.
    ///
    /// - Parameter destination: The session the copy makes.
    /// - Returns: What the copy did.
    func copy(to destination: String) async -> PromptCacheCopyResult {
        await ExecutorPromptCacheStore.$current.withValue(store) {
            await session.model.copyPromptCache(
                fromSession: Self.sessionID, toSession: destination)
        }
    }

    /// Runs `request` on the model of the session, inside its store.
    private func respond(
        to request: LanguageModelExecutorGenerationRequest
    ) async throws -> ScriptedPassResult {
        let result = try await ScriptedExecutorPass.respond(
            to: request, model: session.model, inside: store)
        await store.waitForSpills()
        return result
    }

    /// The transcript of `turns` prompts whose first entry is `sessionID`.
    private func transcript(turns: Int, sessionID: String = sessionID) -> Transcript {
        ScriptedSessionModel.transcript(firstEntryID: sessionID, turns: turns)
    }

    /// Removes the spool folder and the weights folder.
    func remove() {
        session.remove()
    }
}
#endif  // FoundationModelsIntegration
