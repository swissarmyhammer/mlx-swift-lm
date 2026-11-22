// Copyright © 2026 Apple Inc.

import IntegrationTestHelpers
import MLXLMCommon
import Testing

/// `CollectedGeneration` drains the generation stream of a chat turn into its text, its
/// reasoning, its tool calls and its completion info, as one turn or as one pass for each
/// completion info.
struct CollectedGenerationTests {

    private static let firstPassTokens = 3
    private static let secondPassTokens = 5

    private static func info(generated: Int) -> GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: 0, generationTokenCount: generated, promptTime: 0,
            generationTime: 0, stopReason: .stop)
    }

    /// A tool loop of two passes: reasoning and a tool call, then the answer.
    private static func toolLoopStream() -> AsyncThrowingStream<Generation, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(.reasoning("look it up"))
            continuation.yield(.toolCall(ToolCall(function: .init(name: "find", arguments: [:]))))
            continuation.yield(.info(info(generated: firstPassTokens)))
            continuation.yield(.chunk("the answer"))
            continuation.yield(.info(info(generated: secondPassTokens)))
            continuation.finish()
        }
    }

    @Test func collectKeepsTheTextOfEveryPass() async throws {
        let turn = try await CollectedGeneration.collect(Self.toolLoopStream())
        #expect(turn.text == "the answer")
    }

    @Test func collectKeepsTheReasoningApartFromTheText() async throws {
        let turn = try await CollectedGeneration.collect(Self.toolLoopStream())
        #expect(turn.reasoning == "look it up")
    }

    @Test func collectKeepsTheToolCalls() async throws {
        let turn = try await CollectedGeneration.collect(Self.toolLoopStream())
        #expect(turn.toolCalls.map(\.function.name) == ["find"])
    }

    @Test func collectKeepsTheLastCompletionInfo() async throws {
        let turn = try await CollectedGeneration.collect(Self.toolLoopStream())
        #expect(turn.info?.generationTokenCount == Self.secondPassTokens)
    }

    @Test func collectPassesEndsAPassAtEachCompletionInfo() async throws {
        let passes = try await CollectedGeneration.collectPasses(Self.toolLoopStream())
        #expect(
            passes.map(\.info?.generationTokenCount) == [
                Self.firstPassTokens, Self.secondPassTokens,
            ])
    }

    @Test func collectPassesGivesEachPassItsOwnText() async throws {
        let passes = try await CollectedGeneration.collectPasses(Self.toolLoopStream())
        #expect(passes.map(\.text) == ["", "the answer"])
    }

    /// One pass whose tool call the session rejected.
    private static func rejectedCallStream() -> AsyncThrowingStream<Generation, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(
                .rejectedToolCall(
                    RejectedToolCall(
                        reason: .undeclaredTool, format: .inkling, toolName: "missing",
                        rawText: "missing()")))
            continuation.yield(.info(info(generated: firstPassTokens)))
            continuation.finish()
        }
    }

    @Test func collectKeepsTheRejectedToolCallsApartFromTheToolCalls() async throws {
        let turn = try await CollectedGeneration.collect(Self.rejectedCallStream())
        #expect(turn.rejectedToolCalls.map(\.toolName) == ["missing"])
        #expect(turn.toolCalls.isEmpty)
    }

    @Test func summaryNamesTheRejectedToolCalls() async throws {
        let turn = try await CollectedGeneration.collect(Self.rejectedCallStream())
        #expect(turn.summary.hasSuffix("1 rejected tool call(s)"))
    }
}
