// Copyright © 2026 Apple Inc.
//
// Real-weights proof that a `LanguageModelSession` of an `MLXLanguageModel` whose
// `kvCache` asks for affine 8-bit compression still carries its prompt cache across
// turns: the second turn reuses the first, a turn that drops the generated turn
// restores the checkpoint, and a cache that went to disk comes back warm.
//
// The checkpoints under measurement are the Qwen 3.8 hybrid (16 full-attention
// layers beside its linear layers) and Flint (full-attention layers beside
// sliding-window layers). Generation compresses each full-attention cache after the
// prefill, in a new cache object, thus the executor must carry the compressed
// caches, and not the full-precision caches it handed over.
//
// Every measurement line goes to the unified log, in the subsystem
// `com.apple.FoundationModels-MLX` under the category `KVCacheSessionPromptCache`,
// with the `KV8 SESSION:` prefix. Read the numbers after a run with:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/KVCacheSessionPromptCacheTests`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// Prefix that makes every measurement line greppable in the log.
private let measurementPrefix = "KV8 SESSION:"

/// The log every measurement line of this suite goes to.
private let measurementLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "KVCacheSessionPromptCache")

/// The Qwen 3.8 hybrid checkpoint under measurement.
private let qwenModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// The Flint checkpoint under measurement.
private let flintModelID = "mlx-community/flint-oQ4e"

/// The per-test time limit, in minutes.
private let suiteTimeLimitMinutes = 30

/// Tokens each turn may generate. Thinking mode reasons before it answers.
private let generatedTokenBudget = 1_024

/// The affine group size the Router asks for.
private let affineGroupSize = 64

/// The KV-cache bit width under measurement.
private let affineBits = 8

/// The instructions of the session, the same words the Router test uses.
private let sessionInstructions = "You are a terse, literal assistant."

/// The two turns, the same words the Router test uses.
private let firstPrompt = "My favorite color is teal. Reply with just \"OK\"."
private let secondPrompt = "What is my favorite color? Answer with just the color, lowercase."

/// The color the second turn must name.
private let expectedColor = "teal"

/// Measures framework sessions under affine 8-bit KV-cache compression.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct KVCacheSessionPromptCacheTests {

    /// The second turn of an 8-bit Qwen 3.8 session reuses the whole first turn and names
    /// the color.
    @Test func aSecondEightBitTurnOfQwenReusesTheFirstTurn() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASecondEightBitTurnReusesTheFirstTurn(qwenModelID)
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// The second turn of an 8-bit Flint session reuses the whole first turn and names the
    /// color.
    @Test func aSecondEightBitTurnOfFlintReusesTheFirstTurn() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectASecondEightBitTurnReusesTheFirstTurn(flintModelID)
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// An 8-bit Qwen 3.8 turn that drops the generated turn reuses the first turn up to the
    /// checkpoint the executor took at the end of its last message.
    @Test func anEightBitQwenTurnThatDropsTheGeneratedTurnRestoresTheCheckpoint() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAnEightBitTurnThatDropsTheGeneratedTurnRestoresTheCheckpoint()
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// An 8-bit Qwen 3.8 session whose cache went to disk between its turns comes back
    /// warm: the compressed caches read back from the file.
    @Test func anEightBitQwenSessionComesBackWarmFromDisk() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await expectAnEightBitSessionComesBackWarmFromDisk()
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    // MARK: - Scenarios

    /// Runs two turns of one 8-bit framework session of `modelID`, and records an issue
    /// unless the second turn reuses the whole render of the first, carries compressed
    /// caches, and names the color.
    ///
    /// - Parameter modelID: the checkpoint under measurement.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectASecondEightBitTurnReusesTheFirstTurn(_ modelID: String) async throws {
        await releaseAllGPUMemory()
        let model = try makeEightBitModel(modelID)
        let session = LanguageModelSession(
            model: model, tools: [], instructions: sessionInstructions)

        let first = try await session.respond(to: firstPrompt, options: Self.options)
        let second = try await session.respond(to: secondPrompt, options: Self.options)
        let entry = await ExecutorPromptCacheStore.shared.peek(try key(of: session, model: model))
        let compressed = entry?.caches.count { $0 is QuantizedKVCache } ?? 0
        let line =
            "\(measurementPrefix) \(modelID) turn 1 rendered \(first.usage.input.totalTokenCount); "
            + "turn 2 cached \(second.usage.input.cachedTokenCount) of "
            + "\(second.usage.input.totalTokenCount); compressed caches \(compressed); "
            + "turn 1 <<<\(first.content)>>> turn 2 <<<\(second.content)>>>"
        measurementLog.info("\(line, privacy: .public)")

        #expect(first.usage.input.cachedTokenCount == 0, "\(line)")
        #expect(
            second.usage.input.cachedTokenCount >= first.usage.input.totalTokenCount, "\(line)")
        #expect(compressed > 0, "\(line)")
        #expect(second.content.lowercased().contains(expectedColor), "\(line)")
        await releaseAllGPUMemory()
    }

    /// Runs one 8-bit Qwen 3.8 turn, then a turn over the same transcript without the entries
    /// the model wrote, and records an issue unless the second turn reuses exactly the
    /// checkpoint of the first and names the color.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAnEightBitTurnThatDropsTheGeneratedTurnRestoresTheCheckpoint()
        async throws
    {
        await releaseAllGPUMemory()
        let model = try makeEightBitModel(qwenModelID)
        let session = LanguageModelSession(
            model: model, tools: [], instructions: sessionInstructions)

        let first = try await session.respond(to: firstPrompt, options: Self.options)
        let checkpoint = try #require(
            await ExecutorPromptCacheStore.shared.peek(try key(of: session, model: model))?
                .checkpointPosition)
        let entries = Array(session.transcript)
        let lastPrompt = try #require(
            entries.lastIndex { entry in
                if case .prompt = entry { return true }
                return false
            })
        let dropped = LanguageModelSession(
            model: model, tools: [], transcript: Transcript(entries: Array(entries[...lastPrompt])))
        let second = try await dropped.respond(to: secondPrompt, options: Self.options)
        let line =
            "\(measurementPrefix) dropped turn cached \(second.usage.input.cachedTokenCount) of "
            + "\(second.usage.input.totalTokenCount); checkpoint \(checkpoint); turn 1 rendered "
            + "\(first.usage.input.totalTokenCount); turn 2 <<<\(second.content)>>>"
        measurementLog.info("\(line, privacy: .public)")

        #expect(second.usage.input.cachedTokenCount == checkpoint, "\(line)")
        #expect(second.content.lowercased().contains(expectedColor), "\(line)")
        await releaseAllGPUMemory()
    }

    /// Runs two 8-bit Qwen 3.8 turns of one session inside a store that spills each check-in,
    /// and records an issue unless the second turn restores the first turn from disk and names
    /// the color.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectAnEightBitSessionComesBackWarmFromDisk() async throws {
        await releaseAllGPUMemory()
        let model = try makeEightBitModel(qwenModelID)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("KVCacheSessionPromptCacheTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        await store.configure(memoryBudgetBytes: 0)

        try await ExecutorPromptCacheStore.$current.withValue(store) {
            let session = LanguageModelSession(
                model: model, tools: [], instructions: sessionInstructions)
            let first = try await session.respond(to: firstPrompt, options: Self.options)
            await store.waitForSpills()
            let diskBytes = await store.diskByteCount
            let second = try await session.respond(to: secondPrompt, options: Self.options)
            let line =
                "\(measurementPrefix) disk restore: disk bytes after turn 1 = \(diskBytes), "
                + "turn 2 cached \(second.usage.input.cachedTokenCount) of "
                + "\(second.usage.input.totalTokenCount), turn 1 rendered "
                + "\(first.usage.input.totalTokenCount); turn 2 <<<\(second.content)>>>"
            measurementLog.info("\(line, privacy: .public)")

            #expect(diskBytes > 0, "\(line)")
            #expect(
                second.usage.input.cachedTokenCount >= first.usage.input.totalTokenCount,
                "\(line)")
            #expect(second.content.lowercased().contains(expectedColor), "\(line)")
        }
        await releaseAllGPUMemory()
    }

    // MARK: - Fixtures

    /// Greedy options with the turn budget.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static var options: GenerationOptions {
        GenerationOptions(
            samplingMode: .greedy, temperature: 0, maximumResponseTokens: generatedTokenBudget)
    }

    /// A reasoning model of `modelID` whose generations run under affine 8-bit KV-cache
    /// compression, as the Router asks for it.
    ///
    /// - Parameter modelID: the checkpoint.
    /// - Returns: the model.
    /// - Throws: when the weights are not in the local Hugging Face cache, or the
    ///   configuration is not valid.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func makeEightBitModel(_ modelID: String) throws -> MLXLanguageModel {
        let kvCache = KVCacheConfiguration(
            strategy: .affine(
                try AffineKVCacheConfiguration(bits: affineBits, groupSize: affineGroupSize)))
        let model = makeTestModel(
            modelID, capabilities: [.reasoning, .guidedGeneration, .toolCalling],
            kvCache: kvCache)
        try model.requireLocalWeights()
        return model
    }

    /// The prompt cache key of `session` on `model`.
    ///
    /// - Parameters:
    ///   - session: the session, whose first transcript entry names its cache.
    ///   - model: the model of the session.
    /// - Returns: the key.
    /// - Throws: when the transcript is empty.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func key(
        of session: LanguageModelSession, model: MLXLanguageModel
    ) throws -> ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(
            modelID: .init(model.modelID),
            sessionID: .init(try #require(session.transcript.first?.id)),
            kvCache: model.kvCache)
    }
}

#endif  // FoundationModelsIntegration
