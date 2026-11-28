// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// A hybrid model keeps a second checkpoint at the end of the instructions of its prompt, beside
/// the checkpoint at the end of the last message. A next render that keeps the instructions and
/// parts from the ledger after them goes back to it, thus a session seeded with a copy of a
/// template whose prompt parts right after the instructions reads them from the copy.
///
/// These plans run over a scripted model of one attention layer and one recurrent layer. No
/// weights are needed.
@Suite("A hybrid model restores its instructions checkpoint when a render parts after them")
struct ExecutorPromptCacheInstructionsCheckpointTests {

    /// The tokens of the instructions.
    private static let instructions = [1, 2]

    /// The tokens at the checkpoint at the end of the last message: the instructions and the
    /// first prompt.
    private static let transcript = instructions + [3, 4]

    /// The ledger of the entry: the transcript, then the generated turn of the pass.
    private static let ledger = transcript + [5, 6]

    /// A token that no ledger here holds, thus a render parts from the ledger at it.
    private static let otherToken = 9

    /// An entry whose hybrid caches hold ``ledger``, with a checkpoint at the end of
    /// ``transcript`` and an instructions checkpoint at the end of ``instructions``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func entryWithBothCheckpoints() throws -> ExecutorPromptCacheEntry {
        let caches = try hybridScriptedModel().newCache(parameters: nil)
        feed(caches, tokenCount: Self.instructions.count)
        let instructionsCheckpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: Self.instructions, state: nil))
        feed(caches, tokenCount: Self.transcript.count - Self.instructions.count)
        let checkpoint = try #require(
            PromptCacheCheckpoint(caches: caches, tokens: Self.transcript, state: nil))
        feed(caches, tokenCount: Self.ledger.count - Self.transcript.count)
        return ExecutorPromptCacheEntry(
            caches: caches, tokens: Self.ledger, checkpoint: checkpoint,
            instructionsCheckpoint: instructionsCheckpoint)
    }

    /// Plans a pass over `render` that reuses ``entryWithBothCheckpoints()``.
    ///
    /// - Parameter render: the tokens of the next render.
    /// - Returns: the plan.
    /// - Throws: the error of the plan or of the fixture, or when no plan is made.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func planRender(_ render: [Int]) throws -> ExecutorPromptCachePlan {
        try #require(
            try planPromptCache(
                render: render, reusing: entryWithBothCheckpoints(), model: hybridScriptedModel()))
    }

    /// Plans a pass of a slot that carries nothing over ``ledger``, with the instructions
    /// boundary at the end of ``instructions`` and the transcript boundary at the end of
    /// ``transcript``.
    ///
    /// - Parameter slot: the slot of the pass.
    /// - Returns: the plan.
    /// - Throws: the error of the plan, or when no plan is made.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func planWithBothBoundaries(in slot: ExecutorPromptCacheSlot) throws
        -> ExecutorPromptCachePlan
    {
        try #require(
            try slot.plan(
                input: LMInput(tokens: MLXArray(Self.ledger)), model: hybridScriptedModel(),
                parameters: GenerateParameters(),
                checkpointBoundaries: ExecutorCheckpointBoundaries(
                    instructionsEnd: Self.instructions.count,
                    transcriptEnd: Self.transcript.count),
                decodeTokens: { _ in "" }))
    }

    @Test("a render that parts after the instructions goes back to the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatPartsAfterTheInstructionsGoesBackToTheInstructionsCheckpoint() throws {
        let planned = try planRender(Self.instructions + [Self.otherToken])

        #expect(planned.reusedTokenCount == Self.instructions.count)
    }

    @Test("a restore of the instructions checkpoint feeds the render past the instructions")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRestoreOfTheInstructionsCheckpointFeedsTheRenderPastTheInstructions() throws {
        let planned = try planRender(Self.instructions + [Self.otherToken])

        #expect(planned.input.text.tokens.asArray(Int.self) == [Self.otherToken])
    }

    @Test("a restore of the instructions checkpoint takes the caches back to the instructions")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRestoreOfTheInstructionsCheckpointTakesTheCachesBackToTheInstructions() throws {
        let planned = try planRender(Self.instructions + [Self.otherToken])

        #expect(planned.caches.allSatisfy { $0.offset == Self.instructions.count })
    }

    @Test("a render that both checkpoints serve goes back to the longer one")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatBothCheckpointsServeGoesBackToTheLongerOne() throws {
        let planned = try planRender(Self.transcript + [Self.otherToken])

        #expect(planned.reusedTokenCount == Self.transcript.count)
    }

    @Test("a restore keeps the instructions checkpoint for the next turn")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRestoreKeepsTheInstructionsCheckpointForTheNextTurn() throws {
        let planned = try planRender(Self.transcript + [Self.otherToken])

        #expect(planned.instructionsCheckpoint?.tokens == Self.instructions)
    }

    @Test("a render that extends the ledger keeps the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatExtendsTheLedgerKeepsTheInstructionsCheckpoint() throws {
        let planned = try planRender(Self.ledger + [Self.otherToken])

        #expect(planned.instructionsCheckpoint?.tokens == Self.instructions)
    }

    @Test("a render that parts inside the instructions drops the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aRenderThatPartsInsideTheInstructionsDropsTheInstructionsCheckpoint() throws {
        let planned = try planRender([Self.instructions[0], Self.otherToken])

        #expect(planned.instructionsCheckpoint == nil)
    }

    @Test("a pass with an instructions boundary takes a checkpoint at the end of the instructions")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassWithAnInstructionsBoundaryTakesACheckpointThere() throws {
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })

        let planned = try planWithBothBoundaries(in: slot)

        #expect(planned.instructionsCheckpoint?.tokens == Self.instructions)
    }

    @Test("a pass with both boundaries still takes the checkpoint at the end of the transcript")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassWithBothBoundariesStillTakesTheTranscriptCheckpoint() throws {
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })

        let planned = try planWithBothBoundaries(in: slot)

        #expect(planned.checkpoint?.tokens == Self.transcript)
    }

    @Test("a pass with both boundaries leaves the tokens after the transcript to the generation")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassWithBothBoundariesLeavesTheRestToTheGeneration() throws {
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })

        let planned = try planWithBothBoundaries(in: slot)

        #expect(
            planned.input.text.tokens.asArray(Int.self)
                == Array(Self.ledger.dropFirst(Self.transcript.count)))
    }

    @Test("a pass with both boundaries counts every token it fed before the generation")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aPassWithBothBoundariesCountsEveryTokenItFed() throws {
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })

        _ = try planWithBothBoundaries(in: slot)

        #expect(slot.prefilledTokenCount == Self.transcript.count)
    }

    @Test("the entry a pass commits carries its instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theEntryAPassCommitsCarriesItsInstructionsCheckpoint() throws {
        let slot = ExecutorPromptCacheSlot(nil, report: { _ in })
        let planned = try planWithBothBoundaries(in: slot)
        feed(planned.caches, tokenCount: Self.ledger.count - Self.transcript.count)

        slot.commit(planned, generatedTokens: [])

        #expect(slot.entry?.instructionsCheckpoint?.tokens == Self.instructions)
    }

    @Test("a copy of an entry carries its instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func aCopyOfAnEntryCarriesItsInstructionsCheckpoint() throws {
        let copy = try entryWithBothCheckpoints().copy()

        #expect(copy.instructionsCheckpoint?.tokens == Self.instructions)
    }

    @Test("an entry counts the bytes of both checkpoints")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func anEntryCountsTheBytesOfBothCheckpoints() throws {
        let entry = try entryWithBothCheckpoints()
        let checkpoint = try #require(entry.checkpoint)
        let instructionsCheckpoint = try #require(entry.instructionsCheckpoint)

        #expect(
            entry.byteCount
                == entry.caches.reduce(0) { $0 + $1.residentByteCount } + checkpoint.byteCount
                + instructionsCheckpoint.byteCount)
    }

    @Test("the commit line names the position of the instructions checkpoint")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func theCommitLineNamesThePositionOfTheInstructionsCheckpoint() throws {
        let line = ExecutorPromptCacheReport.commitLine(
            key: PromptCacheTestKey.make(modelID: .init("test/checkpoint"), sessionID: .init("s")),
            outcome: .checkedIn(try entryWithBothCheckpoints()))

        #expect(
            line
                == "prompt cache commit model=test/checkpoint session=s ledger=6 checkpoint=4 "
                + "instructions=2")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
