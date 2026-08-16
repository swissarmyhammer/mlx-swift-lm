// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import Testing

@testable import MLXFoundationModels

// MARK: - Session property

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension SessionPropertyValues {

    /// The number of tool calls the re-entrancy turn made.
    ///
    /// ``RealModelReentryProfile`` reads it to select the tool-calling mode of
    /// each round.
    @SessionPropertyEntry
    var realModelReentryToolCallCount: Int = 0
}

// MARK: - Tool

/// What the tool body produced, and how many times the body ran.
///
/// The session runs the tool body on a task of its own, thus an actor holds the
/// shared state.
private actor RealModelReentryLog {

    /// The text the tool body generated, or `nil` when the body never ran.
    private(set) var generatedText: String?

    /// The number of times the tool body ran.
    private(set) var callCount = 0

    /// Keeps the text of one nested generation and counts the call.
    func record(_ text: String) {
        generatedText = text
        callCount += 1
    }
}

/// The one argument of the probe tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct RealModelReentryArguments {
    @Guide(description: "Any topic to look up.")
    var topic: String
}

/// A tool whose body starts a second generation on the model that its own turn
/// runs on, which is what a host does when a tool ranks candidates with a model.
///
/// The body drives a second session over the same model, thus over the one
/// resident container. If the executor held that container across the tool call,
/// the body could not take it and both sides would park forever.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RealModelReentryTool: FoundationModels.Tool {
    let name = "probe"
    let description = "Looks up a topic with a second generation on the same model."

    /// Upper bound on the tokens of the nested generation. The test measures
    /// COMPLETION, not the quality of the answer, thus a short answer is enough.
    private static let maximumNestedTokens = 48

    let model: MLXLanguageModel
    let log: RealModelReentryLog

    func call(arguments: RealModelReentryArguments) async throws -> String {
        let nested = LanguageModelSession(model: model, tools: [], instructions: nil)
        let response = try await nested.respond(
            to:
                "Name one colour that you relate to \(arguments.topic). Answer with the colour only.",
            options: GenerationOptions(maximumResponseTokens: Self.maximumNestedTokens))
        await log.record(response.content)
        return response.content
    }
}

// MARK: - Profile

/// Makes the outer turn call the tool one time and then stop.
///
/// A model of one billion parameters does not reliably decide to call a tool,
/// thus round one uses `.required`, which constrains generation to a real tool
/// call. Each round after the tool output uses `.disallowed`, thus the turn
/// always ends instead of calling the tool again.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
private struct RealModelReentryProfile: LanguageModelSession.DynamicProfile {

    /// Upper bound on the tokens of each round of the outer turn.
    private static let maximumRoundTokens = 64

    let model: MLXLanguageModel
    let log: RealModelReentryLog

    @SessionProperty(\.realModelReentryToolCallCount)
    var toolCallCount

    var body: some LanguageModelSession.DynamicProfile {
        if toolCallCount == 0 {
            Profile {
                Instructions {
                    "Call the probe tool one time. After it returns, answer with its result."
                }
                RealModelReentryTool(model: model, log: log)
            }
            .model(model)
            .maximumResponseTokens(Self.maximumRoundTokens)
            .toolCallingMode(.required)
            .onToolCall {
                toolCallCount += 1
            }
        } else {
            Profile {
                Instructions {
                    "Answer from the latest tool output in one short sentence."
                }
            }
            .model(model)
            .maximumResponseTokens(Self.maximumRoundTokens)
            .toolCallingMode(.disallowed)
        }
    }
}

// MARK: - Turn outcome

/// What one session turn produced, carried out of the turn's own task.
private enum RealModelTurnOutcome: Sendable {
    case finished(String)
    case failed(String)
}

// MARK: - Tests

/// The executor must not hold the model container across the tool call it emits.
///
/// `ToolBodyContainerReentryTests`, in the package test target, asserts the same
/// behavior over a scripted tokenizer and a stub model, thus it does NO MLX
/// evaluation. This suite closes that gap: every generation of the turn — the
/// tool-call round, the nested round the tool body starts, and the round that
/// concludes the turn — is a real forward pass over real weights.
@Suite(.serialized, .timeLimit(.minutes(10)))
struct ToolBodyContainerReentryRealModelTests {

    /// Llama 3.2 (1B) is the smallest instruction-tuned model in the local
    /// cache, and its chat template renders the tool block, an assistant tool
    /// call and a tool result, thus each round of the turn renders correctly.
    private static let modelID = TestFixtures.llamaModelID

    /// Upper bound for the complete turn.
    ///
    /// The turn loads one 680 MB checkpoint and then makes three bounded
    /// generations, thus it completes in seconds on any supported host. Five
    /// minutes is well beyond the first-use Metal shader compilation and beyond
    /// a cold read of the weights from disk, thus only a true park reaches it.
    /// The bound exists so this test FAILS instead of hanging: a hanging test
    /// blocks the whole suite.
    private static let turnTimeout = Duration.seconds(300)

    @Test("Setup: release GPU state from prior suites")
    func clearGPUBeforeContainerReentry() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        await releaseAllGPUMemory()
    }

    @Test("A tool body may generate on the same real model while its turn is in flight")
    func toolBodyGeneratesOnTheSameRealModelDuringItsTurn() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let model = makeTestModel(Self.modelID)
        let log = RealModelReentryLog()
        let session = LanguageModelSession(
            profile: RealModelReentryProfile(model: model, log: log))

        // The turn runs in a task of its own and reports through a stream. A
        // task parked on the container lock cannot be cancelled, thus the
        // timeout must never await that task. Structured concurrency always
        // awaits its children, thus it cannot hold this turn: an unstructured
        // task is the only shape that lets the timeout win.
        let (outcomes, report) = AsyncStream<RealModelTurnOutcome>.makeStream()
        Task {
            do {
                let response = try await session.respond(
                    to: "Look up the topic ocean with the probe tool.")
                report.yield(.finished(response.content))
            } catch {
                report.yield(.failed(String(describing: error)))
            }
            report.finish()
        }

        guard let outcome = await Self.firstOutcome(from: outcomes, within: Self.turnTimeout)
        else {
            Issue.record(
                """
                The turn did not complete within \(Self.turnTimeout) on \(Self.modelID). \
                The tool body generates on the same container the turn holds, thus both \
                sides are parked: the executor emits its tool call while it still holds \
                the container, and the tool body cannot take it.
                """)
            return
        }

        switch outcome {
        case .finished(let content):
            let callCount = await log.callCount
            #expect(
                callCount == 1,
                "The turn must run the tool body exactly one time.")
            let nested = try #require(
                await log.generatedText,
                "The tool body did not generate on the same model.")
            #expect(
                !nested.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "The nested generation produced no text, thus it did no MLX evaluation.")
            #expect(
                !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "The outer turn completed with no answer.")
            #expect(
                session.transcript.contains(where: Self.isToolOutput),
                "The session transcript must hold the output of the tool it ran.")
        case .failed(let description):
            Issue.record("The turn failed instead of completing: \(description)")
        }
    }

    /// Reports whether a transcript entry is the output of a tool.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func isToolOutput(_ entry: Transcript.Entry) -> Bool {
        guard case .toolOutput = entry else { return false }
        return true
    }

    /// The first outcome the turn reports, or `nil` when `timeout` wins.
    ///
    /// Both children are cancellable, thus the group always finishes even when
    /// the turn itself never does.
    ///
    /// `ToolBodyContainerReentryTests` holds the same helper. The two test
    /// targets link no common module, thus neither can call the other's copy.
    private static func firstOutcome(
        from outcomes: AsyncStream<RealModelTurnOutcome>, within timeout: Duration
    ) async -> RealModelTurnOutcome? {
        await withTaskGroup(of: RealModelTurnOutcome?.self) { group in
            group.addTask {
                for await outcome in outcomes { return outcome }
                return nil
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }
}

#endif  // FoundationModelsIntegration
