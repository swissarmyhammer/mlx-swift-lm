// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// An executor pass that cannot leave a prompt cache behind checks in
/// nothing, thus the next turn of its session starts cold.
///
/// These passes run the executor over scripted models. No weights are needed.
@Suite("A pass that checks in nothing leaves its session cold")
struct ExecutorPromptCacheCheckInNothingTests {

    /// The issue a test records on a system that has no executor.
    private static let unsupportedSystem: Comment =
        "The executor needs iOS 27, macOS 27 or visionOS 27."

    @Test("a turn whose stream body throws checks in nothing, and the next turn starts cold")
    func aTurnWhoseStreamBodyThrowsChecksInNothing() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAFailedStreamBodyChecksInNothing()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a schema pass reports no reuse and leaves no cache of its session in the store")
    func aSchemaPassReportsNoReuseAndLeavesNoCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASchemaPassLeavesNoCache()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a required tool pass reports no reuse and deletes the spilled file of its session")
    func aRequiredToolPassDeletesTheSpilledFile() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectARequiredToolPassDeletesTheSpilledFile()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    @Test("a required tool pass reports no reuse and leaves no cache of its session in the store")
    func aRequiredToolPassReportsNoReuseAndLeavesNoCache() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectARequiredToolPassLeavesNoCache()
        } else {
            Issue.record(Self.unsupportedSystem)
        }
    }

    /// A turn whose stream body throws stops its generation and checks in nothing. The next
    /// turn of the session then reuses no prompt token.
    ///
    /// The second turn writes a tool call that never closes. The tool-call processor of the
    /// plain path rejects it at the end of the stream, thus the body throws after the plan of
    /// the turn took the cache of the first turn.
    ///
    /// - Throws: the error of a turn that must succeed.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAFailedStreamBodyChecksInNothing() async throws {
        let session = try await CheckedInNothingSession.make(
            laterScripts: [
                CheckedInNothingSession.unclosedToolCallScript,
                ScriptedSessionModel.scriptedResponse,
            ],
            budget: CheckedInNothingSession.memoryBudgetBytes)
        defer { session.remove() }
        await session.expectFirstTurnCheckedIn()

        await #expect(throws: RejectedToolCallError.self) {
            try await session.respond(turns: CheckedInNothingSession.laterTurnCount)
        }
        #expect(await session.store.peek(session.key) == nil)

        let next = try await session.respond(turns: CheckedInNothingSession.nextTurnCount)
        #expect(next.reusedTokenCount == 0, "The turn after a failed turn must start cold.")
        #expect(next.responseText == ScriptedSessionModel.scriptedResponse)
    }

    /// A schema pass over a session whose cache is in memory reports no reused prompt token,
    /// and the store holds nothing for the session after the pass.
    ///
    /// - Throws: the error of a turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASchemaPassLeavesNoCache() async throws {
        let session = try await CheckedInNothingSession.make(
            laterScripts: [CheckedInNothingSession.jsonString],
            budget: CheckedInNothingSession.memoryBudgetBytes)
        defer { session.remove() }
        await session.expectFirstTurnCheckedIn()

        let guided = try await session.respond(
            turns: CheckedInNothingSession.laterTurnCount, schema: String.generationSchema)

        try await session.expectCheckedInNothing(after: guided)
    }

    /// A required tool pass over a session whose cache went to disk reports no reused prompt
    /// token, deletes the spilled file, and leaves nothing in the store for the session.
    ///
    /// - Throws: the error of a turn, or an issue when the first turn spilled no file.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectARequiredToolPassDeletesTheSpilledFile() async throws {
        let session = try await CheckedInNothingSession.make(
            laterScripts: [CheckedInNothingSession.toolCallScript], budget: 0)
        defer { session.remove() }
        #expect(
            try CheckedInNothingSession.fileNames(in: session.directory).count == 1,
            "The first turn must spill one file.")

        try await session.expectARequiredToolPassChecksInNothing()
    }

    /// A required tool pass over a session whose cache is in memory reports no reused prompt
    /// token, and the store holds nothing for the session after the pass.
    ///
    /// - Throws: the error of a turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectARequiredToolPassLeavesNoCache() async throws {
        let session = try await CheckedInNothingSession.make(
            laterScripts: [CheckedInNothingSession.toolCallScript],
            budget: CheckedInNothingSession.memoryBudgetBytes)
        defer { session.remove() }
        await session.expectFirstTurnCheckedIn()

        try await session.expectARequiredToolPassChecksInNothing()
    }
}

// MARK: - A session whose later pass checks in nothing

/// A scripted session whose first turn checked a prompt cache in, in memory or on disk. A
/// later pass of the session must check in nothing.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct CheckedInNothingSession: PromptCacheSpoolFixtures {

    /// The folders, the store and the model of the session.
    let session: ScriptedSpoolSession

    /// A memory budget that keeps the cache of the first turn in memory.
    static let memoryBudgetBytes = 1 << 30

    /// The first entry of the session.
    static let sessionID = "checked-in-nothing"

    /// The number of prompts of the pass after the first turn.
    static let laterTurnCount = 2

    /// The number of prompts of the turn after that pass.
    static let nextTurnCount = 3

    /// The name of the tool of the required tool pass.
    static let toolName = "lookup"

    /// The tool of the required tool pass. Its argument is one string.
    static let tool = Transcript.ToolDefinition(
        name: toolName, description: "Looks a word up.", parameters: String.generationSchema)

    /// A tool call that never closes. The tool-call processor of the plain path rejects it
    /// at the end of the stream.
    static let unclosedToolCallScript = "<tool_call>{\"name\": \"\(toolName)\""

    /// A JSON string, which a string schema accepts. It is the text of the schema pass and
    /// the arguments of the call of the required tool pass.
    static let jsonString = "\"A\""

    /// A call of ``tool`` with ``jsonString`` as its argument, in the bare form that the
    /// grammar of the required tool mode accepts.
    static let toolCallScript = "{\"name\": \"\(toolName)\", \"arguments\": \(jsonString)}"

    /// The spool folder of the store.
    var directory: URL { session.directory }

    /// The store of the session.
    var store: ExecutorPromptCacheStore { session.store }

    /// The scripted model of the session.
    var model: MLXLanguageModel { session.model }

    /// The key of the session.
    var key: ExecutorPromptCacheKey { session.key }

    /// Runs the first turn of a new session, and waits until its cache is in the store.
    ///
    /// - Parameters:
    ///   - laterScripts: the text of each pass after the first turn, in order.
    ///   - budget: the memory budget of the store in bytes. Zero spills the cache of the
    ///     first turn to disk.
    /// - Returns: the session.
    /// - Throws: the error of the first turn.
    static func make(laterScripts: [String], budget: Int) async throws -> CheckedInNothingSession {
        CheckedInNothingSession(
            session: try await ScriptedSpoolSession.make(sessionID: sessionID, budget: budget) {
                ScriptedSessionModel.make(
                    weights: $0, scripts: [ScriptedSessionModel.scriptedResponse] + laterScripts)
            })
    }

    /// Records an issue unless the first turn of the session checked its cache in.
    func expectFirstTurnCheckedIn() async {
        #expect(await store.peek(key) != nil, "The first turn must check in.")
    }

    /// Runs one later pass of the session.
    ///
    /// - Parameters:
    ///   - turns: the number of prompts of the transcript of the pass.
    ///   - schema: the schema of the response, or nil for a response of plain text.
    ///   - tools: the tools that the pass enables.
    ///   - options: the generation options of the pass.
    /// - Returns: what the pass streamed.
    /// - Throws: the error of the executor.
    func respond(
        turns: Int, schema: GenerationSchema? = nil, tools: [Transcript.ToolDefinition] = [],
        options: GenerationOptions = GenerationOptions()
    ) async throws -> ScriptedPassResult {
        try await ScriptedExecutorPass.respond(
            to: makeExecutorRequest(
                transcript: ScriptedSessionModel.transcript(
                    firstEntryID: Self.sessionID, turns: turns),
                enabledTools: tools, schema: schema, generationOptions: options),
            model: model, inside: store)
    }

    /// Runs one required tool pass of ``tool`` over ``laterTurnCount`` prompts, and records an
    /// issue unless the pass called the tool and checked in nothing.
    ///
    /// - Throws: the error of the pass, or of the check of the store.
    func expectARequiredToolPassChecksInNothing() async throws {
        let guided = try await respond(
            turns: Self.laterTurnCount, tools: [Self.tool],
            options: GenerationOptions(toolCallingMode: .required))

        #expect(guided.toolCallNames == [Self.toolName])
        try await expectCheckedInNothing(after: guided)
    }

    /// Records an issue unless `guided` reused no prompt token and the store holds nothing
    /// for the session: no entry in memory, no spill, and no file.
    ///
    /// - Parameter guided: what a guided pass of the session streamed.
    /// - Throws: the error of the read of the spool folder.
    func expectCheckedInNothing(after guided: ScriptedPassResult) async throws {
        await store.waitForSpills()
        #expect(guided.reusedTokenCount == 0)
        try await Self.expectNothingStored(for: key, in: store, directory: directory)
    }

    /// Removes the spool folder and the weights folder.
    func remove() {
        session.remove()
    }
}
#endif  // FoundationModelsIntegration
