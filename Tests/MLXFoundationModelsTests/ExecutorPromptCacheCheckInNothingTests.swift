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
}

// MARK: - A session whose later pass checks in nothing

/// A scripted session whose first turn checked a prompt cache in. A later pass of the session
/// must check in nothing.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct CheckedInNothingSession {

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

    /// A tool call that never closes. The tool-call processor of the plain path rejects it
    /// at the end of the stream.
    static let unclosedToolCallScript = "<tool_call>{\"name\": \"lookup\""

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

    /// Runs one later plain pass of the session.
    ///
    /// - Parameter turns: the number of prompts of the transcript of the pass.
    /// - Returns: what the pass streamed.
    /// - Throws: the error of the executor.
    func respond(turns: Int) async throws -> ScriptedPassResult {
        try await ScriptedExecutorPass.respond(
            over: ScriptedSessionModel.transcript(firstEntryID: Self.sessionID, turns: turns),
            model: model, inside: store)
    }

    /// Removes the spool folder and the weights folder.
    func remove() {
        session.remove()
    }
}
#endif  // FoundationModelsIntegration
