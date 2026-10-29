// Copyright © 2026 Apple Inc.
//
// A two-turn greedy FoundationModels session on Qwen3.8-27B-4bit, with its 4-bit MTP head and
// without it, that reuses its prompt cache. Run it in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPPromptCacheTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Synchronization
import os
import Testing

@testable import MLXFoundationModels

/// The second turn of a FoundationModels session with MTP reuses the prompt cache of the first
/// turn and the drafter state beside it, recomputes what the same session recomputes without
/// MTP, and writes the text that session writes, or parts from it only where it writes the best
/// token of an uncached forward pass or a near tie.
///
/// Each session keeps the reasoning protocol of the target, as an app that declares `.reasoning`
/// does, thus the render of the second turn writes the reasoning of the first turn inside its
/// think block and reuses the whole ledger of the first turn: the second turn feeds only the
/// close of that turn and the new prompt.
@Suite(.serialized, .timeLimit(.minutes(30)))
struct MTPPromptCacheTests {

    /// The tokens each turn generates.
    private static let generatedTokenCount = 128

    /// The instructions of each session.
    private static let instructions = "You are a concise math tutor. Answer in plain text."

    /// The prompt of each turn, in order.
    private static let prompts = [
        "What is 17 times 23? Show the steps.",
        "Now divide that result by 7 and round it to two decimals.",
    ]

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP PROMPT CACHE:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPPromptCache")

    /// What one turn wrote, the prompt tokens its usage reported, the completion of its
    /// generation, and the token ledger of the cache it left: its prompt and the tokens it fed.
    private struct Turn {
        let text: String
        let promptTokenCount: Int
        let cachedTokenCount: Int
        let completion: GenerateCompletionInfo
        let ledger: [Int]

        /// The prompt tokens the turn fed to the model: the part the cache did not serve.
        var recomputedTokenCount: Int { promptTokenCount - cachedTokenCount }
    }

    /// The turns of one session, and the start of the drafter state its cache keeps after the
    /// last turn.
    private struct Conversation {
        let turns: [Turn]
        let drafterStartPosition: Int?
    }

    /// The usage and the completion events of one turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private final class TurnLog: Sendable {

        /// The prompt tokens of a usage event: every one, and the ones the cache served.
        typealias Usage = (total: Int, cached: Int)

        /// The last usage and the last completion so far.
        private let recorded = Mutex<(usage: Usage?, completion: GenerateCompletionInfo?)>(
            (nil, nil))

        /// Records `event` when it is a usage event or a completion event.
        func record(_ event: MLXLanguageModel.Executor.GenerationEvent) {
            switch event {
            case .updateUsage(let input, _, _):
                recorded.withLock {
                    $0.usage = (input.totalTokenCount, input.cachedTokenCount)
                }
            case .completion(let info):
                recorded.withLock { $0.completion = info }
            default:
                break
            }
        }

        /// The turn these events give, with `text` and `ledger`.
        ///
        /// - Parameters:
        ///   - text: The response text of the turn.
        ///   - ledger: The token ledger of the cache the turn left.
        /// - Returns: The turn.
        /// - Throws: When the turn reported no usage or no completion.
        func turn(text: String, ledger: [Int]) throws -> Turn {
            let (usage, completion) = recorded.withLock { ($0.usage, $0.completion) }
            let reported = try #require(usage, "The turn must report its usage.")
            return Turn(
                text: text, promptTokenCount: reported.total, cachedTokenCount: reported.cached,
                completion: try #require(completion, "The turn must report its completion."),
                ledger: ledger)
        }
    }

    /// Runs ``prompts`` as the turns of one greedy session of `model`.
    ///
    /// - Parameter model: The model of the session.
    /// - Returns: The turns, and the drafter state the cache of the session keeps after them.
    /// - Throws: The error of a turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func converse(_ model: MLXLanguageModel) async throws -> Conversation {
        let session = LanguageModelSession(
            model: model, tools: [], instructions: Self.instructions)
        var turns: [Turn] = []
        var entry: ExecutorPromptCacheEntry?
        for prompt in Self.prompts {
            let log = TurnLog()
            let observer: @Sendable (MLXLanguageModel.Executor.GenerationEvent) -> Void = {
                log.record($0)
            }
            let text = try await MLXLanguageModel.Executor.$generationObserver.withValue(observer) {
                try await session.respond(
                    to: prompt,
                    options: GenerationOptions(
                        samplingMode: .greedy, maximumResponseTokens: Self.generatedTokenCount)
                ).content
            }
            // The first transcript entry names the session of the cache.
            let sessionID = try #require(session.transcript.first?.id)
            entry = await ExecutorPromptCacheStore.shared.peek(
                ExecutorPromptCacheKey(modelID: model.modelID, sessionID: sessionID))
            let ledger = try #require(entry?.tokens, "The turn must check its cache in.")
            turns.append(try log.turn(text: text, ledger: ledger))
        }
        return Conversation(
            turns: turns,
            drafterStartPosition: try entry?.drafterState?.metadata()[Self.cacheStartPositionKey]
                .flatMap { Int($0) })
    }

    /// The metadata key under which a drafter state records the position of its first entry.
    private static let cacheStartPositionKey =
        MTPDrafterContinuation.reservedKeyPrefix + "cache_start_position"

    /// Writes one measurement line for each turn of the two sessions.
    private func logMeasurements(mtp: Conversation, plain: Conversation) {
        for (index, (mtpTurn, plainTurn)) in zip(mtp.turns, plain.turns).enumerated() {
            let measurement = """
                \(Self.measurementPrefix) turn \(index + 1): MTP recomputed \
                \(mtpTurn.recomputedTokenCount) of \(mtpTurn.promptTokenCount) prompt tokens, \
                \(mtpTurn.completion.tokensPerSecond) tok/s (accepted \
                \(mtpTurn.completion.acceptedDraftTokens ?? 0) of \
                \(mtpTurn.completion.proposedDraftTokens ?? 0)); plain recomputed \
                \(plainTurn.recomputedTokenCount) of \(plainTurn.promptTokenCount), \
                \(plainTurn.completion.tokensPerSecond) tok/s
                """
            Self.measurementLog.info("\(measurement, privacy: .public)")
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: Qwen38FourBitMTPModel.isLocal))
    func aSecondTurnReusesThePromptCacheAndTheDrafterState() async throws {
        let plainModel = try Qwen38FourBitMTPModel.make(speculates: false, reasons: true)
        let mtpModel = try Qwen38FourBitMTPModel.make(speculates: true, reasons: true)

        let plain = try await converse(plainModel)
        let mtp = try await converse(mtpModel)
        logMeasurements(mtp: mtp, plain: plain)

        for turn in mtp.turns {
            #expect((turn.completion.proposedDraftTokens ?? 0) > 0)
            #expect(turn.completion.passthroughReason == nil)
        }
        // The two sessions render the same prompts until a turn writes another text, and the
        // later turns then render other prompts. That turn may part from plain decode only where
        // MTP wrote the best token of an uncached forward pass, which plain decode's cached
        // passes rounded the other way, or at a near tie.
        let divergentTurn = mtp.turns.indices.first { mtp.turns[$0].text != plain.turns[$0].text }
        for index in 0 ... (divergentTurn ?? mtp.turns.count - 1) {
            #expect(mtp.turns[index].recomputedTokenCount == plain.turns[index].recomputedTokenCount)
        }
        if let divergentTurn,
            let divergence = try await Qwen38FourBitMTPModel.divergence(
                mtp: mtp.turns[divergentTurn].ledger, plain: plain.turns[divergentTurn].ledger,
                context: [])
        {
            #expect(
                divergence.mtpWroteTheArgmax || divergence.mtpWroteANearTie,
                "turn \(divergentTurn + 1): \(divergence)")
        }
        // The second turn of each session reuses the whole cache of the first: it feeds only
        // the tokens past the commit that closes the first turn, never the chat again. A splice
        // counts the reused tokens up to that commit in the render, which a turn that stopped
        // on its token budget reaches one token past the ledger.
        for conversation in [mtp, plain] {
            let first = try #require(conversation.turns.first)
            let second = try #require(conversation.turns.last)
            #expect(second.cachedTokenCount >= first.ledger.count)
        }
        // The drafter state began with the first turn and was resumed, not started cold.
        #expect(mtp.drafterStartPosition == 0)
        await plainModel.evict()
        await mtpModel.evict()
    }
}

#endif
