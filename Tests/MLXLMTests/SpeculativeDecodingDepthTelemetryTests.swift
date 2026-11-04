import Testing

@testable import MLXLMCommon

/// The acceptance of each draft depth that speculative decoding telemetry counts: of the rounds
/// that drafted a token at a depth, the share whose draft at that depth the target accepted.
@Suite
struct SpeculativeDecodingDepthTelemetryTests {

    /// A round that drafted three tokens and accepted the first.
    private static let firstRound = (drafted: 3, accepted: 1)

    /// A round that drafted two tokens and accepted both.
    private static let secondRound = (drafted: 2, accepted: 2)

    /// The telemetry of ``firstRound`` and ``secondRound``.
    private static func recordedTelemetry() -> SpeculativeDecodingTelemetry {
        var telemetry = SpeculativeDecodingTelemetry()
        for round in [firstRound, secondRound] {
            telemetry.recordRound(
                drafted: round.drafted, accepted: round.accepted,
                targetVerified: round.drafted + 1)
        }
        return telemetry
    }

    @Test("each depth counts the rounds that drafted a token there and the rounds that kept it")
    func countsRoundsByDepth() {
        let telemetry = Self.recordedTelemetry()

        #expect(telemetry.draftedRoundCountsByDepth == [2, 2, 1])
        #expect(telemetry.acceptedRoundCountsByDepth == [2, 1, 0])
    }

    @Test("the acceptance of a depth is its accepted rounds over its drafted rounds")
    func acceptanceRateOfEachDepth() {
        let telemetry = Self.recordedTelemetry()

        #expect(telemetry.acceptanceRate(atDepth: 1) == 1)
        #expect(telemetry.acceptanceRate(atDepth: 2) == 0.5)
        #expect(telemetry.acceptanceRate(atDepth: 3) == 0)
    }

    @Test("a depth no round drafted has no acceptance rate")
    func undraftedDepthHasNoAcceptanceRate() {
        let telemetry = Self.recordedTelemetry()
        let undraftedDepth = telemetry.draftedRoundCountsByDepth.count + 1

        #expect(telemetry.acceptanceRate(atDepth: undraftedDepth) == nil)
        #expect(telemetry.acceptanceRate(atDepth: 0) == nil)
    }
}
