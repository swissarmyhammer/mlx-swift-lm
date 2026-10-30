// Copyright © 2026 Apple Inc.
//
// The divergence the FoundationModels router saw on Qwen3.8-27B-4bit with its 4-bit MTP head
// (148f227e): greedy MTP on the router's load path reasoned about 30 tokens longer than plain
// greedy decode on one short prompt, and at a 512-token cap it ended inside its reasoning. This
// suite decodes that prompt with MTP and without it, on the router's load path and on the text
// path, finds the first token where the two part, and holds that token to the near-tie rule of
// `MTPChatTokenMatch`. Run it in Release:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPNearTieDivergenceTests`
//
// Every measurement line goes to the unified log under the `MTP NEAR TIE:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPNearTie"'`

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import BenchmarkHelpers
import Foundation
import FoundationModels
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import Testing
import os

@testable import MLXFoundationModels

/// The time limit of each test, in minutes: one model load, two decodes of 512 tokens, and one
/// token-by-token replay of the shared prefix.
private let nearTieTimeLimitMinutes = 30

/// The first token where greedy MTP parts from plain greedy decode on the router's prompt is a
/// near tie: plain decode's two best logits there lie within
/// ``MTPChatTokenMatch/nearTieLogitTolerance`` of each other, and MTP took the second.
///
/// Neither decode reports its logits through the executor, thus the logits at the divergence come
/// from a replay of plain decode over the shared prefix: the render in one forward pass, then one
/// token at a time, as plain decode feeds it. From that cache, a one-token forward of the last
/// shared token gives plain decode's row, and a two-token forward of the last shared token and
/// MTP's token gives the row a verify pass reads for that position: row 0 of a two-row forward,
/// which the batched kernels round apart from a one-row forward. A first divergence that is no
/// near tie is a bug, and the failure names its token index and both gaps.
@Suite(.serialized, .timeLimit(.minutes(nearTieTimeLimitMinutes)))
struct MTPNearTieDivergenceTests {

    /// The prompt the router sent, with no instructions.
    private static let prompt =
        "Write the numbers from 1 to 80 in words, separated by commas. Output only the list."

    /// The token cap the router decoded under.
    private static let tokenBudget = 512

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP NEAR TIE:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPNearTie")

    /// The rows of the replay at the first divergence.
    private struct DivergenceRows {

        /// The two best tokens of plain decode's one-token row.
        let plain: [GenerateTokenLogProbability]

        /// The two best tokens of the verify pass's row: row 0 of a two-token forward.
        let verify: [GenerateTokenLogProbability]

        /// The gap of the two best logits of `candidates`.
        static func gap(_ candidates: [GenerateTokenLogProbability]) -> Float {
            candidates[0].logProbability - candidates[1].logProbability
        }
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: Qwen38FourBitMTPModel.LoadPath.allCases)
    func firstDivergenceFromPlainDecodeIsANearTie(
        path: Qwen38FourBitMTPModel.LoadPath
    ) async throws {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        let plainRunner = try SessionRunner(
            model: try Qwen38FourBitMTPModel.make(speculates: false, reasons: true, path: path),
            label: "\(path) plain")
        let mtpRunner = try SessionRunner(
            model: try Qwen38FourBitMTPModel.make(speculates: true, reasons: true, path: path),
            label: "\(path) MTP")
        let container = try await plainRunner.model.loadContainer()
        try await plainRunner.warmUp()
        try await mtpRunner.warmUp()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MTPNearTieDivergenceTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        // No entry spills, thus the ledger of each decode stays readable.
        await store.configure(memoryBudgetBytes: Int.max)
        let (plain, mtp) = try await ExecutorPromptCacheStore.$current.withValue(store) {
            let plain = try await plainRunner.run(
                [SWEBenchTranscript.prompt(Self.prompt)], tokenBudget: Self.tokenBudget,
                declaresTool: false)
            let mtp = try await mtpRunner.run(
                [SWEBenchTranscript.prompt(Self.prompt)], tokenBudget: Self.tokenBudget,
                declaresTool: false)
            return (plain, mtp)
        }
        for (name, pass) in [("plain", plain), ("MTP", mtp)] {
            Self.log(
                """
                \(path) \(name): generated \(pass.completion.generationTokenCount) tokens \
                (stop \(pass.completion.stopReason)), reasoning \(pass.events.reasoning.count) \
                characters, response \(pass.events.text.count) characters, accepted \
                \(pass.completion.acceptedDraftTokens ?? 0) of \
                \(pass.completion.proposedDraftTokens ?? 0) drafts
                """)
        }
        #expect((mtp.completion.proposedDraftTokens ?? 0) > 0, "\(path): MTP drafted nothing")
        #expect(mtp.completion.passthroughReason == nil, "\(path)")

        if plain.output == mtp.output {
            Self.log("\(path): MTP wrote the tokens of plain decode")
        } else {
            try await expectNearTie(path: path, plain: plain, mtp: mtp, container: container)
        }
        await releaseAllGPUMemory()
    }

    /// Finds where `mtp` first parts from `plain`, records both gaps there, and holds the
    /// near-tie rule.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectNearTie(
        path: Qwen38FourBitMTPModel.LoadPath, plain: Pass, mtp: Pass, container: ModelContainer
    ) async throws {
        let step = commonPrefixLength(plain.ledger, mtp.ledger)
        let renderCount = plain.renderTokens.count
        #expect(step >= renderCount, "\(path): the renders part at token \(step)")
        let plainToken = try #require(
            plain.ledger.indices.contains(step) ? plain.ledger[step] : nil,
            "\(path): plain decode ended at the divergence, token \(step - renderCount)")
        let mtpToken = try #require(
            mtp.ledger.indices.contains(step) ? mtp.ledger[step] : nil,
            "\(path): MTP ended at the divergence, token \(step - renderCount)")
        let rows = try await Self.replayRows(
            container, ledger: plain.ledger, renderCount: renderCount, step: step,
            mtpToken: mtpToken)
        let tokens = { (ledger: [Int], candidates: [GenerateTokenLogProbability]) in
            ledger.enumerated().map { index, token in
                MTPChatToken(token: token, candidates: index == step ? candidates : [])
            }
        }
        let match = MTPChatTokenMatch(
            mtp: tokens(mtp.ledger, []), plain: tokens(plain.ledger, rows.plain))
        let report = """
            \(path): first divergence at generated token \(step - renderCount) (ledger \
            \(step)): plain wrote \(plainToken), MTP wrote \(mtpToken); plain row top-2 \
            \(rows.plain), gap \(DivergenceRows.gap(rows.plain)); verify row top-2 \
            \(rows.verify), gap \(DivergenceRows.gap(rows.verify)); replay reproduces plain \
            \(rows.plain[0].token == plainToken); \(match.summary)
            """
        Self.log(report)
        #expect(match.holds, "Bug, no near tie: \(report)")
    }

    /// Replays plain decode over `ledger` up to `step` and reads plain decode's row and the
    /// verify row there.
    ///
    /// - Parameters:
    ///   - container: The loaded target.
    ///   - ledger: The tokens of plain decode: its render, then what it decoded.
    ///   - renderCount: The tokens of the render.
    ///   - step: The first position where MTP parts from plain decode.
    ///   - mtpToken: The token MTP wrote at `step`.
    /// - Returns: The two rows.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func replayRows(
        _ container: ModelContainer, ledger: [Int], renderCount: Int, step: Int, mtpToken: Int
    ) async throws -> DivergenceRows {
        try await container.perform { modelContext in
            let model = modelContext.model
            let cache = try model.newCache(parameters: nil)
            let forward = { (tokens: [Int], cache: [KVCache], state: LMOutput.State?) in
                model(
                    LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis, 0...]),
                    cache: cache, state: state)
            }
            // Plain decode prefills the render but its last token, then feeds one token at a
            // time from there; the last shared token is fed apart, from a copy of the cache.
            let lastShared = step - 1
            let prefillEnd = min(renderCount - 1, lastShared)
            try #require(prefillEnd > 0, "the render must hold more than one token")
            let prefilled = forward(Array(ledger[0 ..< prefillEnd]), cache, nil)
            eval(cache.flatMap(\.state))
            let state = ledger[prefillEnd ..< lastShared].reduce(prefilled.state) { state, token in
                let output = forward([token], cache, state)
                eval(cache.flatMap(\.state))
                return output.state
            }
            let plainRow = forward([ledger[lastShared]], cache.map { $0.copy() }, state)
            let verifyRow = forward(
                [ledger[lastShared], mtpToken], cache.map { $0.copy() }, state)
            return DivergenceRows(
                plain: Qwen38FourBitMTPModel.topCandidates(of: plainRow.logits[0, -1]),
                verify: Qwen38FourBitMTPModel.topCandidates(of: verifyRow.logits[0, 0]))
        }
    }

    /// Writes `measurement` under the measurement prefix.
    private static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
