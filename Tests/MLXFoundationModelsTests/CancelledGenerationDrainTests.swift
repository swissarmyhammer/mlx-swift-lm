// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A cancelled generation must drain the GPU before the throw returns.
///
/// When a caller cancels `respond`, the throw must not return while
/// generation work is still in flight on the GPU. A process that exits
/// inside that window aborts on signal 6 (a Metal command-buffer
/// assertion) or signal 11 (a forward pass over freed process state).
/// A Swift Testing time limit always creates that window, because
/// nothing runs after the throw.
@Suite("Cancelled generation drain")
struct CancelledGenerationDrainTests {

    /// Token budget of the cancelled turn.
    ///
    /// The budget bounds how long an undrained generation can run after
    /// a failed drain, so a failing run ends in a few seconds instead of
    /// running to the executor's own default budget.
    private static let responseTokenBudget = 64

    /// Length of the scripted text, in single-byte tokens.
    ///
    /// The script is longer than ``responseTokenBudget``, thus the model
    /// never emits its end-of-text token and only the budget or the
    /// cancellation stops the generation.
    private static let scriptLength = 128

    /// Duration of one scripted forward pass, in seconds.
    ///
    /// The delay reproduces the decode speed of a real model — a 1B
    /// model runs a forward pass in tens of milliseconds — so the drain
    /// after the cancellation takes real time, as it does in the
    /// reported crash.
    private static let forwardPassDuration: TimeInterval = 0.05

    /// Forward passes that prove the generation is mid-decode.
    ///
    /// The cancel fires only after the model ran this many forward
    /// passes, thus the cancellation always races an active generation.
    private static let stepsBeforeCancel = 3

    /// How long the test watches for forward passes after the throw returns.
    ///
    /// A drained generation runs zero forward passes in this window. An
    /// undrained one runs several, because each forward pass takes
    /// ``forwardPassDuration``.
    private static let quiescenceWindow = Duration.milliseconds(500)

    /// Upper bound on the wait for the generation to start.
    ///
    /// The scripted model loads instantly, thus only a broken load path
    /// reaches this bound. The bound exists so this test FAILS instead
    /// of polling forever.
    private static let generationStartTimeout = Duration.seconds(10)

    /// Pause between two reads of the forward-pass count while polling.
    private static let pollInterval = Duration.milliseconds(10)

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("A cancelled session respond returns only after the GPU drain completes")
    func cancelledSessionRespondDrainsBeforeReturn() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let forwardSteps = ForwardStepCounter()
        let model = Self.makeDrainProbeModel(
            weights: weights, forwardSteps: forwardSteps)
        let session = LanguageModelSession(model: model, tools: [], instructions: nil)

        let turn = Task {
            _ = try await session.respond(
                to: "Generate.",
                options: GenerationOptions(
                    maximumResponseTokens: Self.responseTokenBudget))
        }
        try await Self.cancelMidDecodeAndExpectDrain(turn, forwardSteps: forwardSteps)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("A cancelled executor respond returns only after the GPU drain completes")
    func cancelledExecutorRespondDrainsBeforeReturn() async throws {
        let weights = try makeScriptedWeightsDirectory()
        defer { try? FileManager.default.removeItem(at: weights) }

        let forwardSteps = ForwardStepCounter()
        let model = Self.makeDrainProbeModel(
            weights: weights, forwardSteps: forwardSteps)
        let executor = try makeMLXExecutor(for: model)
        let request = makeExecutorRequest(
            transcript: Transcript(entries: [
                .prompt(
                    Transcript.Prompt(
                        segments: [.text(Transcript.TextSegment(content: "Generate."))],
                        responseFormat: nil))
            ]),
            generationOptions: GenerationOptions(
                maximumResponseTokens: Self.responseTokenBudget))
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside
        // the executor or every send parks it.
        let consumer = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { consumer.cancel() }

        let turn = Task {
            try await executor.respond(to: request, model: model, streamingInto: channel)
        }
        try await Self.cancelMidDecodeAndExpectDrain(turn, forwardSteps: forwardSteps)
    }

    /// Builds the model the drain probes cancel.
    ///
    /// The model replays one script that outruns ``responseTokenBudget``
    /// and slows each forward pass to ``forwardPassDuration``, and it
    /// records every pass into `forwardSteps`. A fresh identity for each
    /// call keeps the process-wide model cache out of every other test.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func makeDrainProbeModel(
        weights: URL, forwardSteps: ForwardStepCounter
    ) -> MLXLanguageModel {
        let modelID = "probe/cancelled-drain-\(UUID().uuidString)"
        let script = String(repeating: "A", count: scriptLength)
        return MLXLanguageModel(
            configuration: ModelConfiguration(id: modelID),
            capabilities: [],
            weightsLocation: { _ in weights },
            load: { _, _ in
                makeScriptedContainer(
                    modelID: modelID, rounds: [script], forwardSteps: forwardSteps,
                    forwardDelay: forwardPassDuration)
            })
    }

    /// Cancels `turn` mid-decode and asserts the drain beat the return.
    ///
    /// Waits for the generation to reach ``stepsBeforeCancel`` forward
    /// passes, cancels the turn, awaits its result, and then asserts
    /// that no forward pass was still running when the turn returned
    /// and that none starts inside ``quiescenceWindow``.
    private static func cancelMidDecodeAndExpectDrain(
        _ turn: Task<Void, any Error>, forwardSteps: ForwardStepCounter
    ) async throws {
        let started = await waitForForwardSteps(stepsBeforeCancel, of: forwardSteps)
        guard started else {
            turn.cancel()
            _ = await turn.result
            Issue.record(
                """
                The generation ran no forward pass within \
                \(generationStartTimeout), so there was no mid-decode \
                generation to cancel.
                """)
            return
        }

        turn.cancel()
        _ = await turn.result

        let inFlightAtReturn = forwardSteps.inFlight
        let begunAtReturn = forwardSteps.begun
        #expect(
            inFlightAtReturn == 0,
            """
            The cancelled respond returned while \(inFlightAtReturn) forward \
            pass(es) were still running on the GPU. A process exit in that \
            window aborts on signal 6 or 11.
            """)
        try await Task.sleep(for: quiescenceWindow)
        let begunAfterWindow = forwardSteps.begun
        #expect(
            begunAfterWindow == begunAtReturn,
            """
            The cancelled respond returned while generation work was still \
            in flight: the model started \(begunAfterWindow - begunAtReturn) \
            forward pass(es) after the throw returned to the caller. A \
            process exit in that window aborts on signal 6 or 11.
            """)
    }

    /// Waits until the model started `steps` forward passes.
    ///
    /// - Parameters:
    ///   - steps: the count to wait for.
    ///   - counter: the counter the model writes.
    /// - Returns: true when the count was reached, false when
    ///   ``generationStartTimeout`` won.
    private static func waitForForwardSteps(
        _ steps: Int, of counter: ForwardStepCounter
    ) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: generationStartTimeout)
        while ContinuousClock.now < deadline {
            if counter.begun >= steps { return true }
            try? await Task.sleep(for: pollInterval)
        }
        return counter.begun >= steps
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
