// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing

@testable import MLXFoundationModels

/// The closing reserves of a guided pass: each scales with the shortest valid output, and the
/// budget caps it, thus the start of the output decodes free of the closing bias. The cap stops
/// at a floor of three times the shortest valid output: a budget of that floor or less runs
/// wholly in the hard zone, the accepted trade against incomplete output.
///
/// The numbers come from the Inkling 2 Mini city call of `Inkling2GuidedGenerationTests`: the
/// call allowed 256 tokens, and the test takes 32 tokens as the shortest output of its schema,
/// thus eight times that reserve, the hard zone before the caps, covers the whole budget. With
/// that hard zone every string of the output closed at once or filled with `}`.
@Suite("The closing reserves of a guided pass")
struct GuidedClosingReservesTests {

    /// A shortest valid output whose hard zone before the caps (× 8) covers ``shortBudget``.
    private static let cityStructuralReserve = 32

    /// The budget of the Inkling 2 city call.
    private static let shortBudget = 256

    /// A shortest output longer than a quarter of ``shortBudget``.
    private static let largeStructuralReserve = 100

    /// The structural reserves the invariant sweep covers.
    private static let sweptStructuralReserves = 0 ... 200

    /// The budgets the invariant sweep covers.
    private static let sweptBudgets = 0 ... 4096

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func reserves(_ structural: Int, _ maxTokens: Int)
        -> MLXLanguageModel.Executor.ClosingReserves
    {
        MLXLanguageModel.Executor.closingReserves(
            structuralReserve: structural, maxTokens: maxTokens)
    }

    /// The smallest hard zone the budget cap leaves for `structural`.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func hardFloor(_ structural: Int) -> Int {
        structural * MLXLanguageModel.Executor.hardReserveStructuralFloorMultiplier
    }

    @Test("the hard zone of a short budget takes its floor or a quarter, not the whole budget")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theHardZoneOfAShortBudgetLeavesTheStartFree() {
        let reserves = Self.reserves(Self.cityStructuralReserve, Self.shortBudget)

        #expect(
            reserves.hard
                == Swift.max(
                    Self.hardFloor(Self.cityStructuralReserve),
                    Self.shortBudget
                        / MLXLanguageModel.Executor.hardReserveBudgetCeilingDivisor))
        #expect(reserves.hard < Self.shortBudget)
    }

    @Test("the soft zone of a short budget takes at most half of it")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theSoftZoneOfAShortBudgetTakesHalf() {
        let reserves = Self.reserves(Self.largeStructuralReserve, Self.shortBudget)

        #expect(
            reserves.completion
                == Swift.max(
                    reserves.hard,
                    Self.shortBudget
                        / MLXLanguageModel.Executor.completionReserveBudgetCeilingDivisor))
    }

    @Test("a cap keeps the hard zone at its structural floor")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCapKeepsTheStructuralFloor() {
        let reserves = Self.reserves(Self.largeStructuralReserve, Self.shortBudget)

        #expect(reserves.hard == Self.hardFloor(Self.largeStructuralReserve))
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
                == budget / MLXLanguageModel.Executor.completionReserveBudgetFloorDivisor)
    }

    @Test("a budget below the shortest valid output keeps the structural floor")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aBudgetBelowTheShortestOutput() {
        let structural = Self.cityStructuralReserve
        let reserves = Self.reserves(structural, structural - 1)

        #expect(reserves.hard == Self.hardFloor(structural))
        #expect(reserves.completion == reserves.hard)
    }

    @Test("a budget of zero to three tokens keeps the structural floor", arguments: 0 ... 3)
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aTinyBudget(maxTokens: Int) {
        let structural = Self.cityStructuralReserve
        let reserves = Self.reserves(structural, maxTokens)

        #expect(reserves.hard == Self.hardFloor(structural))
        #expect(reserves.completion == reserves.hard)
    }

    @Test("a schema whose shortest output takes no token reserves no hard zone")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aZeroStructuralReserve() {
        for budget in [0, 3, Self.shortBudget, MLXLanguageModel.Executor.defaultMaxTokens] {
            let reserves = Self.reserves(0, budget)

            #expect(reserves.hard == 0, "budget \(budget)")
            #expect(
                reserves.completion
                    == Swift.min(
                        budget
                            / MLXLanguageModel.Executor.completionReserveBudgetFloorDivisor,
                        budget
                            / MLXLanguageModel.Executor.completionReserveBudgetCeilingDivisor),
                "budget \(budget)")
        }
    }

    @Test("every reserve keeps its bounds over the swept structural reserves and budgets")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theInvariantsHoldOverTheSweep() {
        let executor = MLXLanguageModel.Executor.self
        var violations: [String] = []
        for structural in Self.sweptStructuralReserves {
            for budget in Self.sweptBudgets {
                let reserves = Self.reserves(structural, budget)
                let hardCeiling = structural * executor.hardReserveStructuralMultiplier
                let hardFloor = Swift.min(Self.hardFloor(structural), hardCeiling)
                let budgetCap = Swift.max(
                    Self.hardFloor(structural), budget / executor.hardReserveBudgetCeilingDivisor)
                let completionCeiling = Swift.max(
                    reserves.hard, budget / executor.completionReserveBudgetCeilingDivisor)
                let holds =
                    reserves.hard >= 0
                    && reserves.hard >= hardFloor
                    && reserves.hard <= hardCeiling
                    && reserves.hard <= budgetCap
                    && reserves.completion >= reserves.hard
                    && reserves.completion <= completionCeiling
                if !holds {
                    violations.append("s=\(structural) M=\(budget): \(reserves)")
                }
            }
        }

        #expect(violations.isEmpty, "\(violations.prefix(10))")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels, _version: 2)
