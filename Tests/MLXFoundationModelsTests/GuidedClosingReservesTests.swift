// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing

@testable import MLXFoundationModels

/// The closing reserves of a guided pass: each scales with the shortest valid output, and the
/// budget caps it, thus the start of the output decodes free of the closing bias.
///
/// The numbers are the ones Inkling 2 Mini measured: its shortest output of the city schema
/// takes 32 tokens, and the call allowed 256. Without the caps the hard zone covered the whole
/// budget, and every string of the output closed at once or filled with `}`.
@Suite("The closing reserves of a guided pass")
struct GuidedClosingReservesTests {

    /// The tokens of the shortest output of the city schema on Inkling 2 Mini.
    private static let cityStructuralReserve = 32

    /// The budget of the Inkling 2 city call.
    private static let shortBudget = 256

    /// A shortest output longer than a quarter of ``shortBudget``.
    private static let largeStructuralReserve = 100

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func reserves(_ structural: Int, _ maxTokens: Int) -> (
        completion: Int, hard: Int
    ) {
        MLXLanguageModel.Executor.closingReserves(
            structuralReserve: structural, maxTokens: maxTokens)
    }

    @Test("the hard zone of a short budget takes at most a quarter of it")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theHardZoneOfAShortBudgetTakesAQuarter() {
        let reserves = Self.reserves(Self.cityStructuralReserve, Self.shortBudget)

        #expect(
            reserves.hard
                == Self.shortBudget / MLXLanguageModel.Executor.hardReserveBudgetDivisor)
    }

    @Test("the soft zone of a short budget takes at most half of it")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theSoftZoneOfAShortBudgetTakesHalf() {
        let reserves = Self.reserves(Self.largeStructuralReserve, Self.shortBudget)

        #expect(
            reserves.completion
                == Self.shortBudget
                / MLXLanguageModel.Executor.completionReserveBudgetCeilingDivisor)
    }

    @Test("a cap keeps the hard zone at the shortest valid output")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCapKeepsTheShortestValidOutput() {
        let reserves = Self.reserves(Self.largeStructuralReserve, Self.shortBudget)

        #expect(reserves.hard == Self.largeStructuralReserve)
        #expect(reserves.completion >= reserves.hard)
    }

    @Test("the default budget keeps the reserves of the structural multipliers")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theDefaultBudgetKeepsTheMultipliers() {
        let budget = MLXLanguageModel.Executor.defaultMaxTokens
        let reserves = Self.reserves(Self.cityStructuralReserve, budget)

        #expect(
            reserves.hard
                == Self.cityStructuralReserve
                * MLXLanguageModel.Executor.hardReserveStructuralMultiplier)
        #expect(
            reserves.completion
                == budget / MLXLanguageModel.Executor.completionReserveBudgetDivisor)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
