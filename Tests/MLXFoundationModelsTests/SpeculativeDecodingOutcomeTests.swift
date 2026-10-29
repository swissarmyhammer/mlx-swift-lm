// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// The MTP outcome of one call: what each pass of the call reported, and the response metadata
/// that says it.
///
/// These tests feed the outcome by hand. `SpeculativeDecodingOutcomeMetadataTests` runs the same
/// rules through the executor on the shared Qwen 3.5 + MTP fixture.
@Suite
struct SpeculativeDecodingOutcomeTests {

    /// The drafts the first pass of a call proposed.
    private static let firstPassDrafts = 6

    /// The drafts of the first pass that the target accepted.
    private static let firstPassAccepted = 4

    /// The drafts the second pass of a call proposed.
    private static let secondPassDrafts = 9

    /// The drafts of the second pass that the target accepted.
    private static let secondPassAccepted = 2

    /// The drafts a guided pass proposed.
    private static let guidedPassDrafts = 5

    /// The drafts of the guided pass that the target accepted.
    private static let guidedPassAccepted = 3

    /// The drafts an outer sink held before the guided pass of the call began.
    private static let earlierDrafts = 11

    /// The drafts of ``earlierDrafts`` that the target accepted.
    private static let earlierAccepted = 7

    /// The rounds each speculating pass of these tests ran.
    private static let roundsPerPass = 2

    /// The completion report of one pass that proposed `drafted` drafts, of which the target
    /// accepted `accepted`, and that fell back for `fallback`.
    private func completion(
        drafted: Int = 0, accepted: Int = 0, fallback: SpeculativeDecodingFallbackReason? = nil
    ) -> GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: 1, generationTokenCount: 1, promptTime: 0, generationTime: 0,
            speculativeDecodingTelemetry: drafted > 0
                ? SpeculativeDecodingTelemetry(
                    roundCount: Self.roundsPerPass, draftTokenCount: drafted,
                    acceptedDraftTokenCount: accepted)
                : nil,
            speculativeDecodingFallbackReason: fallback)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a pass that drafted reports speculation and its counts, and no fallback reason")
    func aPassThatDraftedReportsSpeculation() {
        let outcome = SpeculativeDecodingOutcome()

        outcome.record(completion(drafted: Self.firstPassDrafts, accepted: Self.firstPassAccepted))

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding] as? Bool == true)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
                == Self.firstPassDrafts)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int
                == Self.firstPassAccepted)
        #expect(metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a call that never drafted reports no speculation and zero counts")
    func aCallThatNeverDraftedReportsNoSpeculation() {
        let outcome = SpeculativeDecodingOutcome()

        outcome.record(completion(fallback: .incompatibleDrafter))

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding] as? Bool == false)
        #expect(metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int == 0)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int == 0)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] as? String
                == "incompatibleDrafter")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the counts sum over every pass of the call, guided passes included")
    func theCountsSumOverEveryPass() {
        let outcome = SpeculativeDecodingOutcome()

        outcome.record(completion(drafted: Self.firstPassDrafts, accepted: Self.firstPassAccepted))
        outcome.record(
            completion(drafted: Self.secondPassDrafts, accepted: Self.secondPassAccepted))
        outcome.recordGuidedPass {
            GuidedGenerationDiagnosticSink.current?.recordSpeculativeRound(
                draftedTokenCount: Self.guidedPassDrafts,
                acceptedDraftTokenCount: Self.guidedPassAccepted)
        }

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
                == Self.firstPassDrafts + Self.secondPassDrafts + Self.guidedPassDrafts)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int
                == Self.firstPassAccepted + Self.secondPassAccepted + Self.guidedPassAccepted)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("the first fallback a pass of the call reported is the reason, after a pass that drafted")
    func theFirstFallbackIsTheReason() {
        let outcome = SpeculativeDecodingOutcome()

        outcome.record(completion(drafted: Self.firstPassDrafts, accepted: Self.firstPassAccepted))
        outcome.record(completion(fallback: .unsupportedCache))
        outcome.record(completion(fallback: .incompatibleDrafter))

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding] as? Bool == true)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] as? String
                == "unsupportedCache")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided pass that falls back reports the reason of its sink")
    func aGuidedFallbackReportsItsReason() {
        let outcome = SpeculativeDecodingOutcome()

        outcome.recordGuidedPass {
            GuidedGenerationDiagnosticSink.current?.recordSpeculationFallback(
                "the drafter cannot draft for the target", reason: .incompatibleDrafter)
        }

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.speculativeDecoding] as? Bool == false)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] as? String
                == "incompatibleDrafter")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided pass under a bound sink counts only its own rounds and fallbacks")
    func aGuidedPassUnderABoundSinkCountsOnlyItsOwn() {
        let sink = GuidedGenerationDiagnosticSink()
        sink.recordSpeculativeRound(
            draftedTokenCount: Self.earlierDrafts, acceptedDraftTokenCount: Self.earlierAccepted)
        sink.recordSpeculationFallback("an earlier call fell back", reason: .incompatibleDrafter)
        let outcome = SpeculativeDecodingOutcome()

        GuidedGenerationDiagnosticSink.$current.withValue(sink) {
            outcome.recordGuidedPass {
                GuidedGenerationDiagnosticSink.current?.recordSpeculativeRound(
                    draftedTokenCount: Self.guidedPassDrafts,
                    acceptedDraftTokenCount: Self.guidedPassAccepted)
            }
        }

        let metadata = outcome.metadata
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
                == Self.guidedPassDrafts)
        #expect(
            metadata[SpeculativeDecodingOutcome.MetadataKey.acceptedDraftTokenCount] as? Int
                == Self.guidedPassAccepted)
        #expect(metadata[SpeculativeDecodingOutcome.MetadataKey.fallbackReason] == nil)
        #expect(sink.draftedTokenCount == Self.earlierDrafts + Self.guidedPassDrafts)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided pass with no bound sink binds one that records no decoding")
    func aGuidedPassBindsASpeculationOnlySink() {
        let outcome = SpeculativeDecodingOutcome()

        let recordsDecoding = outcome.recordGuidedPass {
            GuidedGenerationDiagnosticSink.current?.recordsDecoding
        }

        #expect(recordsDecoding == false)
        #expect(GuidedGenerationDiagnosticSink.current == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided pass outside a call that speculates binds no sink")
    func aGuidedPassOutsideACallBindsNoSink() {
        let bound = SpeculativeDecodingOutcome.recordingGuidedPass {
            GuidedGenerationDiagnosticSink.current
        }

        #expect(bound == nil)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test("a guided pass inside a call that speculates records into the outcome of the call")
    func aGuidedPassInsideACallRecordsIntoItsOutcome() {
        let outcome = SpeculativeDecodingOutcome()

        SpeculativeDecodingOutcome.$current.withValue(outcome) {
            SpeculativeDecodingOutcome.recordingGuidedPass {
                GuidedGenerationDiagnosticSink.current?.recordSpeculativeRound(
                    draftedTokenCount: Self.guidedPassDrafts,
                    acceptedDraftTokenCount: Self.guidedPassAccepted)
            }
        }

        #expect(
            outcome.metadata[SpeculativeDecodingOutcome.MetadataKey.draftedTokenCount] as? Int
                == Self.guidedPassDrafts)
    }
}

#endif
