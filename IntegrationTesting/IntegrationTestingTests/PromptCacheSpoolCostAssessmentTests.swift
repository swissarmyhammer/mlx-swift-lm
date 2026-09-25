// Copyright © 2026 Apple Inc.
//
// Real-weights measurement of the cost of the prompt cache disk spool. The spool is useful only
// when a file write and a later file read cost less than the prefill that they save, and when
// neither stalls other models for too long.
//
// `save(arrays:)` holds the process-wide `evalLock` of MLX for the whole write
// (`.build/checkouts/mlx-swift/Source/MLX/IO.swift`). The read evaluates the lazy load, which
// also takes the lock. Thus the seconds of the write and the seconds of the read are the two
// lock holds.
//
// For each model and for each context of 4 096 and 32 768 tokens, the suite does these steps:
//
//   1. It prefills the context cold, and records the seconds.
//   2. It records `residentByteCount` of the caches.
//   3. It saves the caches with `savePromptCache` to a temporary folder, and records the seconds
//      and the file size.
//   4. It loads the file with `loadPromptCacheSnapshot(url:into:)` into fresh caches of
//      `model.newCache(parameters:)`, which evaluates the arrays, and records the seconds.
//   5. It runs one greedy decode step on the restored caches and one on the original caches.
//      The two tokens must be equal.
//
// The models are `mlx-community/Qwen3-4B-4bit` (pure attention) and
// `mlx-community/Qwen3.8-27B-mxfp4` (hybrid). The suite downloads nothing. When a model is not
// in the local Hugging Face cache, the suite fails with a message that names the model.
//
// Every measurement line goes to the unified log, in the subsystem
// `com.apple.FoundationModels-MLX` under the category `PromptCacheSpoolCostAssessment`, with the
// `PROMPT CACHE SPOOL COST:` prefix. Read the lines after a run with:
// `log show --info --last 1h --predicate 'subsystem == "com.apple.FoundationModels-MLX"'`
//
// Run explicitly via:
// `xcodebuild test -skipPackagePluginValidation -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/PromptCacheSpoolCostAssessmentTests`
//
// `swift test` does not see this file. No SwiftPM target holds `IntegrationTesting/`. Use
// `xcodebuild build-for-testing` as the compile evidence for a change.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

// MARK: - Constants

/// Prefix that makes every measurement line greppable in the log.
private let measurementPrefix = "PROMPT CACHE SPOOL COST:"

/// The log every measurement line of this suite goes to.
private let measurementLog = Logger(
    subsystem: "com.apple.FoundationModels-MLX", category: "PromptCacheSpoolCostAssessment")

/// The hybrid checkpoint, which `Qwen35AgenticPromptCacheAssessmentTests` also measures.
private let hybridModelID = "mlx-community/Qwen3.8-27B-mxfp4"

/// The short context, in tokens.
private let shortContextTokenCount = 4_096

/// The long context, in tokens.
private let longContextTokenCount = 32_768

/// The contexts that the suite measures, in tokens.
private let contextTokenCounts = [shortContextTokenCount, longContextTokenCount]

/// The time limit of each test, in minutes. One test prefills 36 864 tokens of one model and
/// writes and reads two cache files.
private let suiteTimeLimitMinutes = 60

/// The smallest number of tokens that one row of the context text encodes to. The text has
/// sufficient rows for the largest context when each row gives this number of tokens or more.
private let minimumTokensPerContextRow = 8

/// A model identifier that no Hugging Face cache holds.
private let absentModelID = "mlx-community/PromptCacheSpoolCostAssessment-absent-model"

// MARK: - The measurement

/// The numbers of one model at one context.
private struct SpoolCostMeasurement: Sendable {
    /// Tokens in the context.
    let contextTokenCount: Int
    /// Seconds of the cold prefill of the context.
    let prefillSeconds: TimeInterval
    /// Bytes that the caches and the model state hold after the prefill.
    let residentBytes: Int
    /// Bytes of the prompt cache file.
    let fileBytes: Int
    /// Seconds of the file write, which is the lock hold of the write.
    let writeSeconds: TimeInterval
    /// Seconds of the file read, which is the lock hold of the read.
    let readSeconds: TimeInterval
    /// The greedy token of the decode step on the restored caches.
    let restoredToken: Int
    /// The greedy token of the decode step on the original caches.
    let originalToken: Int

    /// The longer of the two lock holds.
    var longestLockHoldSeconds: TimeInterval { max(writeSeconds, readSeconds) }

    /// Whether a write and a read together cost less than the prefill that they save.
    var spoolIsCheaperThanPrefill: Bool { writeSeconds + readSeconds < prefillSeconds }

    /// The measurement line, with every number of the measurement.
    func line(label: String) -> String {
        "\(measurementPrefix) \(label) context=\(contextTokenCount) "
            + "prefillSeconds=\(prefillSeconds) residentBytes=\(residentBytes) "
            + "fileBytes=\(fileBytes) writeSeconds=\(writeSeconds) readSeconds=\(readSeconds) "
            + "longestLockHoldSeconds=\(longestLockHoldSeconds) "
            + "spoolCheaperThanPrefill=\(spoolIsCheaperThanPrefill) "
            + "restoredToken=\(restoredToken) originalToken=\(originalToken)"
    }
}

/// The caches of one cold prefill, and the greedy token that the prefill gives.
private struct PrefilledContext {
    /// The caches that hold the context.
    let caches: [KVCache]
    /// The model state that the prefill gave, if the model gives one.
    let state: LMOutput.State?
    /// The greedy token after the context.
    let nextToken: Int
    /// Seconds of the prefill.
    let seconds: TimeInterval
}

/// Runs the five steps of one measurement inside `ModelContainer.perform`.
private struct SpoolCostProbe {
    /// The loaded model and its tokenizer.
    let context: ModelContext
    /// The folder of the prompt cache files.
    let directory: URL

    /// Runs the five steps at one context.
    ///
    /// - Parameter contextTokenCount: the tokens in the context.
    /// - Returns: the numbers of the measurement.
    func measure(contextTokenCount: Int) throws -> SpoolCostMeasurement {
        let prefilled = try prefill(tokens: try contextTokens(count: contextTokenCount))
        let residentBytes =
            prefilled.caches.reduce(0) { $0 + $1.residentByteCount }
            + (prefilled.state?.residentByteCount ?? 0)

        let url = directory.appendingPathComponent("context-\(contextTokenCount).safetensors")
        let writeSeconds = try Self.seconds {
            try savePromptCache(url: url, cache: prefilled.caches, state: prefilled.state)
        }.seconds
        let fileBytes = try #require(
            try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)

        let templates = try context.model.newCache(parameters: nil)
        let (snapshot, readSeconds) = try Self.seconds {
            try loadPromptCacheSnapshot(url: url, into: templates)
        }

        let restoredToken = greedyToken(
            after: prefilled.nextToken, caches: snapshot.cache, state: snapshot.state)
        let originalToken = greedyToken(
            after: prefilled.nextToken, caches: prefilled.caches, state: prefilled.state)
        return SpoolCostMeasurement(
            contextTokenCount: contextTokenCount, prefillSeconds: prefilled.seconds,
            residentBytes: residentBytes, fileBytes: fileBytes, writeSeconds: writeSeconds,
            readSeconds: readSeconds, restoredToken: restoredToken, originalToken: originalToken)
    }

    /// The first `count` tokens of a long plain text.
    ///
    /// - Parameter count: the tokens to give.
    /// - Returns: exactly `count` tokens.
    private func contextTokens(count: Int) throws -> [Int] {
        let rows = (1 ... count / minimumTokensPerContextRow).map { index in
            "Entry \(index): crate \(index) moved from the north dock to the south dock, "
                + "and the scanner read its label without error."
        }
        let tokens = context.tokenizer.encode(
            text: rows.joined(separator: "\n"), addSpecialTokens: false)
        try #require(
            tokens.count >= count, "The context text gave \(tokens.count) tokens, not \(count).")
        return Array(tokens.prefix(count))
    }

    /// Prefills `tokens` into fresh caches, and evaluates the caches.
    ///
    /// - Parameter tokens: the context.
    /// - Returns: the caches, the model state, the greedy token and the seconds.
    private func prefill(tokens: [Int]) throws -> PrefilledContext {
        let caches = try context.model.newCache(parameters: nil)
        let (output, seconds) = try Self.seconds {
            let output: LMOutput
            switch try context.model.prepare(
                LMInput(tokens: MLXArray(tokens)), cache: caches, state: nil,
                prefill: PrefillParameters())
            {
            case .tokens(let remainder):
                output = context.model(remainder[text: .newAxis], cache: caches, state: nil)
            case .logits(let logits):
                output = logits
            }
            eval(output.logits, caches)
            return output
        }
        return PrefilledContext(
            caches: caches, state: output.state, nextToken: Self.greedyToken(of: output),
            seconds: seconds)
    }

    /// Runs one decode step of `token` and gives the greedy token after it.
    ///
    /// - Parameters:
    ///   - token: the token to feed.
    ///   - caches: the caches that hold the context.
    ///   - state: the model state that belongs to the caches.
    /// - Returns: the greedy token.
    private func greedyToken(after token: Int, caches: [KVCache], state: LMOutput.State?) -> Int {
        let input = LMInput.Text(tokens: MLXArray([token]))
        return Self.greedyToken(
            of: context.model(input[text: .newAxis], cache: caches, state: state))
    }

    /// The greedy token of the last position of `output`.
    private static func greedyToken(of output: LMOutput) -> Int {
        output.logits[0, -1].argMax().item(Int.self)
    }

    /// Runs `body` and measures its wall-clock seconds.
    ///
    /// - Parameter body: the work to measure.
    /// - Returns: the value of `body` and the seconds.
    private static func seconds<Value>(of body: () throws -> Value) rethrows -> (
        value: Value, seconds: TimeInterval
    ) {
        let start = Date()
        let value = try body()
        return (value, Date().timeIntervalSince(start))
    }
}

// MARK: - The suite

/// Measures a prompt cache file write and read against a cold prefill, on real weights.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct PromptCacheSpoolCostAssessmentTests {

    /// The pure-attention model.
    @Test("a pure-attention model restores a spilled cache and decodes the same token")
    func pureAttentionModelSpoolCost() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await measure(modelID: TestFixtures.qwen3ModelID, label: "qwen3-4b")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// The hybrid model.
    @Test("a hybrid model restores a spilled cache and decodes the same token")
    func hybridModelSpoolCost() async throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            try await measure(modelID: hybridModelID, label: "qwen3.8-27b")
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    /// A model that is not in the local cache stops the measurement with an error that names it.
    @Test("a model that is not in the local cache fails with its name")
    func absentModelFailsWithItsName() throws {
        if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
            let error = #expect(throws: MissingLocalModelError.self) {
                try makeTestModel(absentModelID).requireLocalWeights()
            }
            #expect(error?.description.contains(absentModelID) == true)
        } else {
            Issue.record(.unsupportedSystem)
        }
    }

    // MARK: - The check

    /// Measures every context on one model, and logs one line for each context.
    ///
    /// - Parameters:
    ///   - modelID: the model to measure.
    ///   - label: the greppable label of every line.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func measure(modelID: String, label: String) async throws {
        await releaseAllGPUMemory()
        let model = makeTestModel(modelID)
        try model.requireLocalWeights()
        let container = try await model.loadContainer()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("PromptCacheSpoolCostAssessmentTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        for contextTokenCount in contextTokenCounts {
            let measurement = try await container.perform { context in
                try SpoolCostProbe(context: context, directory: directory)
                    .measure(contextTokenCount: contextTokenCount)
            }
            let line = measurement.line(label: label)
            measurementLog.info("\(line, privacy: .public)")
            #expect(
                measurement.restoredToken == measurement.originalToken,
                "The restored caches must decode the token of the original caches. \(line)")
            Memory.clearCache()
        }
        await releaseAllGPUMemory()
    }
}

#endif  // FoundationModelsIntegration
