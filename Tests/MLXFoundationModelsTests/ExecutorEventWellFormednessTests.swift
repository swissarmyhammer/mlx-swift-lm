// Copyright © 2026 Apple Inc.

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Exercises `MLXLanguageModel.Executor`'s channel-emission helpers --
/// `sendDelta`, `sendUsageUpdate`, `sendIncompleteOutputMetadata`, and the
/// shared `textDeltaTokenCount`/`clampedUsageCounts` seam -- to assert the
/// well-formedness invariants documented in `MLXLanguageModel.swift`'s
/// "Channel & Error Surface Conformance" section: every emitted text/
/// arguments fragment carries a nonzero token count, the channel delivers
/// every fragment of a round without drops, and usage totals are
/// clamped/monotone.
///
/// `LanguageModelExecutorGenerationChannel.Event` (and every nested `Action`
/// type -- `Response.Action`, `Reasoning.Action`, `ToolCalls.Action`, etc.)
/// is opaque once constructed: the macOS 27 FoundationModels swiftinterface
/// exposes only static factory functions (`.appendText(...)`,
/// `.updateUsage(...)`, ...) and no public accessors to read a value back
/// out of an already-built `Event`. A test cannot drain a channel and
/// inspect what arrived -- every other test in this target that drains a
/// channel (e.g. `ContextSizeValidationTests`) only discards what it reads,
/// for exactly this reason. These tests instead assert well-formedness at
/// the true emission seam: the values this adapter computes immediately
/// before handing them to `channel.send(...)`, calling the SAME internal
/// helpers `respond()` itself calls (widened from `private` to
/// package-internal for this purpose), proven against a real channel to
/// confirm every send actually completes without dropping or hanging.
///
/// This opacity also bounds what these tests can prove about entryID
/// stability specifically: no test -- including a full end-to-end
/// `respond()` run -- can drain a channel and confirm the entryID *value*
/// carried by each of a round's fragments actually matches. `respond()`'s
/// real guarantee there is structural (see
/// `deltasSharingOneEntryIDAreAllDelivered`'s doc comment for the detail),
/// not something re-verified at runtime here.
@Suite("Executor channel-emission well-formedness")
struct ExecutorEventWellFormednessTests {

    @Test("Every text/arguments fragment this adapter emits carries a nonzero tokenCount")
    func textDeltaTokenCountIsNonzero() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // `textDeltaTokenCount` is the single source of truth every
        // `.appendText`/`.appendArguments` call site (`sendDelta`,
        // `handleRealTool`) embeds -- the SDK's `TextFragment`/
        // `ArgumentsFragment.tokenCount` is a non-optional `Int`, so this
        // must never be zero.
        #expect(MLXLanguageModel.Executor.textDeltaTokenCount > 0)
    }

    @Test(
        "The channel delivers every fragment of a round without drops when they share one entryID"
    )
    func deltasSharingOneEntryIDAreAllDelivered() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // NOTE on what this test does and doesn't prove: `entryID`/
        // `reasoningEntryID` below are literals this test chooses, equal by
        // construction -- not values threaded through `respond()`'s real
        // call chain (`dispatchGeneration` -> `runTextGeneration` ->
        // `runUnconstrained`/`runReasoning` -> `handleGenerationEvent`/
        // `sendDelta`), which this test does not exercise. `respond()`'s
        // actual entryID-stability guarantee is structural instead: it
        // mints `entryID`/`toolCallsEntryID`/`reasoningEntryID` via exactly
        // three `UUID().uuidString` calls (see `respond()`'s locals) and
        // threads each as a required, non-defaulted `String` parameter
        // through every nested helper -- none of `sendDelta`/
        // `sendUsageUpdate`/`sendIncompleteOutputMetadata`/`handleRealTool`
        // has any way to substitute a fresh id of its own. What this test
        // verifies instead is the other half of "well-formed": that the
        // channel itself accepts repeated sends under one entryID, back to
        // back, without dropping or hanging on any of them --
        // `LanguageModelExecutorGenerationChannel.Event`'s opacity (see the
        // suite doc comment) means no test, including a full end-to-end
        // `respond()` run, can additionally assert on the actual entryID
        // *value* carried by a drained `Event`.
        let channel = LanguageModelExecutorGenerationChannel()

        let entryID = "round-1-response"
        let reasoningEntryID = "round-1-reasoning"
        let expectedEventCount = 4

        async let seenCount: Int = {
            var seen = 0
            do {
                for try await _ in channel {
                    seen += 1
                    if seen == expectedEventCount { break }
                }
            } catch {
                // Draining only; a well-formed run never throws here.
            }
            return seen
        }()

        await MLXLanguageModel.Executor.sendDelta(
            "Hel", entryID: entryID, channel: channel, isReasoning: false)
        await MLXLanguageModel.Executor.sendDelta(
            "lo", entryID: entryID, channel: channel, isReasoning: false)
        await MLXLanguageModel.Executor.sendDelta(
            "thinking...", entryID: reasoningEntryID, channel: channel, isReasoning: true)
        await MLXLanguageModel.Executor.sendIncompleteOutputMetadata(
            entryID: entryID, channel: channel)

        let received = await seenCount
        #expect(
            received == expectedEventCount,
            "expected all \(expectedEventCount) emitted events to be delivered without drops")
    }

    @Test("Usage totals are clamped so cached/reasoning counts never exceed their totals")
    func usageCountsAreClampedToTheirTotals() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let overstated = MLXLanguageModel.Executor.clampedUsageCounts(
            promptTokenCount: 10, cachedTokenCount: 999,
            outputTokenCount: 5, reasoningTokenCount: 999)
        #expect(overstated.cachedTokenCount == 10)
        #expect(overstated.reasoningTokenCount == 5)

        let honest = MLXLanguageModel.Executor.clampedUsageCounts(
            promptTokenCount: 10, cachedTokenCount: 3,
            outputTokenCount: 5, reasoningTokenCount: 2)
        #expect(honest.cachedTokenCount == 3)
        #expect(honest.reasoningTokenCount == 2)
    }

    @Test("Usage totals reported across a growing multi-round transcript never decrease")
    func usageTotalsAreMonotoneAcrossRounds() async {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let channel = LanguageModelExecutorGenerationChannel()
        let expectedEventCount = 2

        async let seenCount: Int = {
            var seen = 0
            do {
                for try await _ in channel {
                    seen += 1
                    if seen == expectedEventCount { break }
                }
            } catch {
                // Draining only; a well-formed run never throws here.
            }
            return seen
        }()

        // Round 2's transcript strictly extends round 1's (the prior
        // round's prompt+response becomes part of round 2's prompt), so its
        // reported totals must never be smaller once clamped.
        let round1 = MLXLanguageModel.Executor.clampedUsageCounts(
            promptTokenCount: 10, cachedTokenCount: 0, outputTokenCount: 8, reasoningTokenCount: 0)
        let round2 = MLXLanguageModel.Executor.clampedUsageCounts(
            promptTokenCount: 18, cachedTokenCount: 10, outputTokenCount: 12,
            reasoningTokenCount: 4)

        await MLXLanguageModel.Executor.sendUsageUpdate(
            entryID: "round-1", promptTokenCount: 10, cachedTokenCount: round1.cachedTokenCount,
            outputTokenCount: 8, reasoningTokenCount: round1.reasoningTokenCount, channel: channel)
        await MLXLanguageModel.Executor.sendUsageUpdate(
            entryID: "round-2", promptTokenCount: 18, cachedTokenCount: round2.cachedTokenCount,
            outputTokenCount: 12, reasoningTokenCount: round2.reasoningTokenCount, channel: channel)

        let received = await seenCount
        #expect(received == expectedEventCount)
        #expect(round2.cachedTokenCount >= round1.cachedTokenCount)
        #expect(round2.reasoningTokenCount >= round1.reasoningTokenCount)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
