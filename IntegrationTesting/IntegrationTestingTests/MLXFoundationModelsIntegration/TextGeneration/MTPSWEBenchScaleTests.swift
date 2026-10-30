// Copyright © 2026 Apple Inc.
//
// Qwen3.8-27B-4bit with its 4-bit MTP head at the scale of the ACP agent's SWE-bench run: an
// agent transcript of about 32k tokens with a `runCode` tool, follow-up turns that reuse the
// prompt cache, and one decode of at least 4096 tokens; on the load path of the
// FoundationModels router (`loadModelContainer`, thus the vision-language `Qwen35`, and the
// drafter `.matchingTarget(.qwen35Text, .qwen35VLM)`) and on the text path (`LLMModelFactory`
// and `.qwen35Text`). Run it in Release, or the throughput measures the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/MTPSWEBenchScaleTests`
//
// Every measurement line goes to the unified log under the `MTP SWE-BENCH SCALE:` prefix:
// `log show --info --start <time> --predicate 'subsystem == "com.apple.FoundationModels-MLX" AND category == "MTPSWEBenchScale"'`

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

/// How the target loads.
private typealias LoadPath = Qwen38FourBitMTPModel.LoadPath

/// The time limit of each test, in minutes: the session test runs four sessions of about 13k
/// decoded tokens over a 32k-token prompt, and the sweep of one load path prefills about 208k
/// tokens and decodes 1024 tokens ten times: plain and MTP.
private let sweBenchScaleTimeLimitMinutes = 150

/// The tool call one pass emitted.
struct ExecutorToolCall {
    let id: String
    let name: String
    let arguments: String
}

/// The decode rate of a pass at each position, read from the times its text chunks streamed.
///
/// The executor streams at most one chunk for each decoded token: the detokenizer holds a token
/// that ends inside a UTF-8 sequence or a stop string. A turn with tools streams only its
/// reasoning, because its response may turn out to be a tool call, and sends the response as one
/// chunk at the end. A rate is thus a lower bound on the token rate over the chunks that streamed.
private struct DecodeRates {

    /// The chunks of one position bucket.
    static let bucketChunkCount = 1024

    /// When each streamed chunk arrived, in order.
    let chunkTimes: [Date]

    /// The chunks per second over `range`, or `nil` when it spans no time.
    func rate(over range: Range<Int>) -> Double? {
        guard range.count > 1 else { return nil }
        let seconds = chunkTimes[range.upperBound - 1].timeIntervalSince(
            chunkTimes[range.lowerBound])
        return seconds > 0 ? Double(range.count - 1) / seconds : nil
    }

    /// The rate over the first bucket of the decode.
    var first: Double? { rate(over: 0 ..< min(Self.bucketChunkCount, chunkTimes.count)) }

    /// The rate over the last bucket of the decode.
    var last: Double? {
        rate(over: max(0, chunkTimes.count - Self.bucketChunkCount) ..< chunkTimes.count)
    }

    /// The rate over each bucket, in decode order.
    var buckets: [Double] {
        stride(from: 0, to: chunkTimes.count, by: Self.bucketChunkCount).compactMap {
            rate(over: $0 ..< min($0 + Self.bucketChunkCount, chunkTimes.count))
        }
    }

    /// The rates in a measurement line.
    var summary: String {
        let format = { (rate: Double?) in rate.map { String(format: "%.1f", $0) } ?? "n/a" }
        return """
            \(chunkTimes.count) streamed chunks, first \(Self.bucketChunkCount) \
            \(format(first)) chunks/s, last \(Self.bucketChunkCount) \(format(last)) chunks/s, \
            buckets [\(buckets.map { format($0) }.joined(separator: ", "))]
            """
    }
}

/// The events of one executor pass, as they streamed.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct PassEvents {
    var reasoning = ""
    var text = ""
    var toolCall: ExecutorToolCall?
    var renderedTokenCount = 0
    var cachedTokenCount = 0
    var completion: GenerateCompletionInfo?
    var chunkTimes: [Date] = []

    /// Reads `event`, which arrived at `time`.
    mutating func consume(_ event: MLXLanguageModel.Executor.GenerationEvent, at time: Date) {
        switch event {
        case .appendText(let chunk, _, .reasoning):
            reasoning += chunk
            chunkTimes.append(time)
        case .appendText(let chunk, _, .response):
            text += chunk
            chunkTimes.append(time)
        case .toolCall(let id, let name, let arguments):
            toolCall = toolCall ?? ExecutorToolCall(id: id, name: name, arguments: arguments)
        case .updateUsage(let input, _, _):
            renderedTokenCount = input.totalTokenCount
            cachedTokenCount = input.cachedTokenCount
        case .completion(let info):
            completion = info
        case .updateMetadata:
            break
        }
    }
}

/// What one executor pass wrote and reported, and the cache it left.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct Pass {
    let events: PassEvents
    let completion: GenerateCompletionInfo

    /// The tokens the cache holds after the pass: the render and the tokens decode fed.
    let ledger: [Int]

    /// The prompt the pass rendered.
    let renderTokens: [Int]

    /// The position of the first entry of the drafter state the cache keeps, or `nil`.
    let drafterStartPosition: Int?

    /// The peak of MLX active memory over the pass, in bytes.
    let peakMemoryBytes: Int

    /// The prompt tokens the pass fed: the part the cache did not serve.
    var fedTokenCount: Int { events.renderedTokenCount - events.cachedTokenCount }

    /// The tokens decode wrote after the render.
    var generatedTokens: [Int] { Array(ledger.dropFirst(renderTokens.count)) }

    /// Everything the pass wrote, for the comparison with another session.
    var output: String {
        events.reasoning + "\n" + events.text + "\n" + (events.toolCall?.arguments ?? "")
    }
}

/// One turn of the script: the user prompt and its token budget.
private struct ScriptedTurn {
    let prompt: String
    let tokenBudget: Int
}

/// Runs the turns of one agent session through the executor of one model, the way the
/// FoundationModels framework hands each turn to it.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct SessionRunner {
    let model: MLXLanguageModel
    let executor: MLXLanguageModel.Executor
    let label: String

    /// The tokens of the warm-up pass that compiles the kernels of the model.
    private static let warmUpTokenCount = 8

    /// The key in the drafter state metadata of the position of its first entry.
    private static let cacheStartPositionKey =
        MTPDrafterContinuation.reservedKeyPrefix + "cache_start_position"

    init(model: MLXLanguageModel, label: String) throws {
        self.model = model
        self.executor = try makeMLXExecutor(for: model)
        self.label = label
    }

    /// Runs one greedy pass over `entries` and collects its events.
    func stream(
        _ entries: [Transcript.Entry], tokenBudget: Int, declaresTool: Bool
    ) async throws -> PassEvents {
        let request = makeExecutorRequest(
            transcript: Transcript(entries: entries),
            enabledTools: declaresTool ? [SWEBenchTranscript.tool] : [],
            generationOptions: GenerationOptions(
                samplingMode: .greedy, maximumResponseTokens: tokenBudget))
        return try await executeResponse(executor, request: request, model: model)
            .reduce(into: PassEvents()) { $0.consume($1, at: Date()) }
    }

    /// Compiles the kernels of the model with one short pass that leaves no cache.
    func warmUp() async throws {
        _ = try await MLXLanguageModel.$promptCacheScope.withValue(.uncached) {
            try await stream(
                [SWEBenchTranscript.instructions(), SWEBenchTranscript.prompt("Say OK.")],
                tokenBudget: Self.warmUpTokenCount, declaresTool: true)
        }
    }

    /// Runs one pass of the session `entries` name, and reads the cache it left.
    func run(
        _ entries: [Transcript.Entry], tokenBudget: Int, declaresTool: Bool = true
    ) async throws -> Pass {
        let key = try #require(
            MLXLanguageModel.Executor.sessionCacheKey(
                for: makeExecutorRequest(transcript: Transcript(entries: entries)),
                modelID: model.modelID))
        Memory.peakMemory = 0
        let events = try await stream(
            entries, tokenBudget: tokenBudget, declaresTool: declaresTool)
        let peakMemoryBytes = Memory.peakMemory
        let checkedIn = await ExecutorPromptCacheStore.current.peek(key)
        let entry = try #require(checkedIn, "\(label): the pass must check its cache in")
        return Pass(
            events: events,
            completion: try #require(events.completion, "\(label): the pass must complete"),
            ledger: entry.tokens, renderTokens: entry.renderTokens,
            drafterStartPosition: try entry.drafterState?.metadata()[Self.cacheStartPositionKey]
                .flatMap { Int($0) },
            peakMemoryBytes: peakMemoryBytes)
    }

    /// The entries `pass` adds to the transcript: its reasoning, then its tool call and the
    /// canned output of the tool, or its response.
    func entries(after pass: Pass, turnIndex: Int) throws -> [Transcript.Entry] {
        let reasoning: [Transcript.Entry] =
            pass.events.reasoning.isEmpty
            ? []
            : [
                .reasoning(
                    Transcript.Reasoning(
                        segments: [SWEBenchTranscript.segment(pass.events.reasoning)]))
            ]
        guard let call = pass.events.toolCall else {
            let text = pass.events.text.isEmpty ? "(no answer)" : pass.events.text
            return reasoning + [
                .response(
                    Transcript.Response(
                        assetIDs: [], segments: [SWEBenchTranscript.segment(text)]))
            ]
        }
        let toolCall = Transcript.ToolCall(
            id: call.id, toolName: call.name, arguments: try GeneratedContent(json: call.arguments))
        let output = SWEBenchTranscript.followUpToolOutput(turnIndex: turnIndex)
        return reasoning + [
            .toolCalls(Transcript.ToolCalls(id: "toolcalls_turn_\(turnIndex)", [toolCall])),
            .toolOutput(
                Transcript.ToolOutput(
                    id: call.id, toolName: call.name,
                    segments: [SWEBenchTranscript.segment(output)])),
        ]
    }

    /// Runs `turns` as one session over `history`, each turn on the transcript the turns before
    /// it left, as an agent session grows.
    fileprivate func converse(
        history: [Transcript.Entry], turns: [ScriptedTurn]
    ) async throws -> [Pass] {
        var entries = [SWEBenchTranscript.instructions()] + history
        var passes: [Pass] = []
        for (turnIndex, turn) in turns.enumerated() {
            entries.append(SWEBenchTranscript.prompt(turn.prompt))
            let pass = try await run(entries, tokenBudget: turn.tokenBudget)
            passes.append(pass)
            entries += try self.entries(after: pass, turnIndex: turnIndex)
        }
        return passes
    }
}

/// Qwen3.8-27B-4bit with its MTP head speculates through an agent session of SWE-bench scale on
/// the router's load path and on the text path: every turn drafts, each follow-up feeds only its
/// new messages, the long decode is faster than plain decode, and MTP writes what plain decode
/// writes up to the first near tie.
@Suite(.serialized, .timeLimit(.minutes(sweBenchScaleTimeLimitMinutes)))
struct MTPSWEBenchScaleTests {

    /// The content tokens of the agent history: with the markup of the chat template, the first
    /// turn renders about 32k tokens.
    private static let historyTokenCount = 31_000

    /// The tokens the render of the first turn must reach.
    private static let minimumPromptTokenCount = 30_000

    /// The most history rounds any history may hold.
    private static let maximumHistoryRoundCount = 512

    /// The token budget of a short turn, which holds its reasoning and its answer or tool call.
    private static let shortTurnTokenBudget = 1_536

    /// The token budget of the long decode.
    private static let longDecodeTokenBudget = 8_192

    /// The tokens the long decode must write.
    private static let minimumLongDecodeTokenCount = 4_096

    /// The tokens of the ledger before a follow-up that the follow-up may feed again: the chat
    /// template writes a tool call in its own key order and spacing, thus the tokens from the
    /// start of a tool call the model wrote may be fed again.
    private static let ledgerSeamSlack = 128

    /// The tokens of each chunk of the reference prefill that reads the logits at a divergence.
    private static let referenceChunkTokenCount = 2_048

    /// The context lengths of the decode-rate sweep, in tokens of content: the ACP agent's
    /// SWE-bench sessions reach about 103k.
    private static let sweepContextTokenCounts = [2_048, 8_192, 32_768, 65_536, 100_000]

    /// The token budget of each pass of the sweep, which each pass must write.
    private static let sweepDecodeTokenCount = 1_024

    /// The milliseconds of one second.
    private static let millisecondsPerSecond = 1_000.0

    /// The bytes of one gibibyte.
    private static let bytesPerGibibyte = Double(1 << 30)

    /// Prefix that makes every measurement line greppable in the log.
    private static let measurementPrefix = "MTP SWE-BENCH SCALE:"

    /// The log every measurement line of this suite goes to.
    private static let measurementLog = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "MTPSWEBenchScale")

    /// The prompt of the long decode.
    static let longDecodePrompt = """
        Now write, directly in your reply and without calling any tool, the complete corrected \
        version of the module that the failing test covers: every import, constant, class and \
        function of the file in full, each function with a detailed docstring and inline \
        comments that explain the rounding. Follow it with a complete pytest module that tests \
        every function with at least five cases each, and finish with a migration note for every \
        other module of the repository that shares the rounding bug, naming each module and the \
        exact change. Do not abbreviate, elide or summarize any part; write everything out.
        """

    /// The turns of the session after the history, the last one the long decode.
    private static let turns = [
        ScriptedTurn(
            prompt: """
                Read the last test run above. In one short paragraph, name the function whose \
                rounding makes the test fail and say what the fix is.
                """,
            tokenBudget: shortTurnTokenBudget),
        ScriptedTurn(
            prompt: "Call runCode to print the module that test covers, so we see every case.",
            tokenBudget: shortTurnTokenBudget),
        ScriptedTurn(
            prompt: """
                In three or four sentences, say which functions of that module still round half \
                to even, and why that breaks the totals.
                """,
            tokenBudget: shortTurnTokenBudget),
        ScriptedTurn(prompt: longDecodePrompt, tokenBudget: longDecodeTokenBudget),
    ]

    /// The passes of the two sessions of one load path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct Sessions {
        let path: LoadPath
        let plain: [Pass]
        let mtp: [Pass]
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(.enabled(if: Qwen38FourBitMTPModel.isLocal))
    func agentSessionSpeculatesAtSWEBenchScale() async throws {
        let router = try await measureSessions(on: .router)
        let text = try await measureSessions(on: .text)
        let routerLong = try #require(router.mtp.last).completion.tokensPerSecond
        let textLong = try #require(text.mtp.last).completion.tokensPerSecond
        Self.log("long decode MTP: router path \(routerLong) tok/s, text path \(textLong) tok/s")
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(
        .enabled(if: Qwen38FourBitMTPModel.isLocal),
        arguments: Qwen38FourBitMTPModel.LoadPath.allCases)
    func decodeRateFromTwoToOneHundredThousandContextTokens(
        path: Qwen38FourBitMTPModel.LoadPath
    ) async throws {
        let (runners, countTokens) = try await makeSweepRunners(on: path)
        let baseline = try await measureSweepPoint(
            try #require(Self.sweepContextTokenCounts.first), runners: runners,
            countTokens: countTokens)
        for contextTokenCount in Self.sweepContextTokenCounts.dropFirst() {
            let point = try await measureSweepPoint(
                contextTokenCount, runners: runners, countTokens: countTokens)
            Self.log(
                """
                \(path) sweep \(contextTokenCount): MTP at \(point.mtp / baseline.mtp) and \
                plain at \(point.plain / baseline.plain) of their rates at \
                \(Self.sweepContextTokenCounts[0]) tokens
                """)
        }
        await releaseAllGPUMemory()
    }

    // MARK: - The session benchmark

    /// Runs a plain session and an MTP session on `path`, logs every pass, and holds every
    /// requirement of the two.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func measureSessions(on path: LoadPath) async throws -> Sessions {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        let plainRunner = try SessionRunner(
            model: try Qwen38FourBitMTPModel.make(speculates: false, reasons: true, path: path),
            label: "\(path) plain")
        let mtpRunner = try SessionRunner(
            model: try Qwen38FourBitMTPModel.make(speculates: true, reasons: true, path: path),
            label: "\(path) MTP")
        let container = try await plainRunner.model.loadContainer()
        let tokenizer = await container.tokenizer
        let history = try SWEBenchTranscript.history(
            minimumTokenCount: Self.historyTokenCount,
            maximumRoundCount: Self.maximumHistoryRoundCount,
            countTokens: { tokenizer.encode(text: $0, addSpecialTokens: false).count })
        try await plainRunner.warmUp()
        try await mtpRunner.warmUp()

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MTPSWEBenchScaleTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        // No entry spills, thus the ledger of every pass stays readable.
        await store.configure(memoryBudgetBytes: Int.max)
        let sessions = try await ExecutorPromptCacheStore.$current.withValue(store) {
            let plain = try await plainRunner.converse(history: history, turns: Self.turns)
            let mtp = try await mtpRunner.converse(history: history, turns: Self.turns)
            return Sessions(path: path, plain: plain, mtp: mtp)
        }
        logSessions(sessions)
        try expectSessions(sessions)
        try await expectPlainOutput(sessions, container: container)
        await releaseAllGPUMemory()
        return sessions
    }

    /// Writes one measurement line for each pass of the two sessions, and one comparison line
    /// for each turn.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func logSessions(_ sessions: Sessions) {
        for (index, (plain, mtp)) in zip(sessions.plain, sessions.mtp).enumerated() {
            let turn = "\(sessions.path) turn \(index + 1)"
            Self.log("\(turn) plain: \(Self.describe(plain))")
            Self.log("\(turn) MTP: \(Self.describe(mtp))")
            Self.log(
                """
                \(turn): MTP \(mtp.completion.tokensPerSecond) tok/s vs plain \
                \(plain.completion.tokensPerSecond) tok/s, \
                \(mtp.completion.tokensPerSecond / plain.completion.tokensPerSecond)x
                """)
        }
    }

    /// The numbers of one pass in a measurement line.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func describe(_ pass: Pass) -> String {
        let completion = pass.completion
        let proposed = completion.proposedDraftTokens ?? 0
        let accepted = completion.acceptedDraftTokens ?? 0
        let fallback =
            completion.passthroughReason
            ?? completion.speculativeDecodingFallbackReason.map { "\($0)" } ?? "none"
        let tokensPerVerify =
            completion.speculativeDecodingTelemetry?.meanEmittedTokensPerTargetCall ?? 0
        return """
            prompt \(pass.events.renderedTokenCount) tokens, prefilled \(pass.fedTokenCount) \
            (cached \(pass.events.cachedTokenCount)) at \(completion.promptTokensPerSecond) tok/s; \
            generated \(completion.generationTokenCount) at \(completion.tokensPerSecond) tok/s \
            (stop \(completion.stopReason)); accepted \(accepted) of \(proposed) drafts \
            (\(Double(accepted) / Double(max(proposed, 1)))), \(tokensPerVerify) tokens per \
            verify; drafter start \(pass.drafterStartPosition.map(String.init) ?? "none"); \
            fallback \(fallback); tool call \(pass.events.toolCall?.name ?? "none"); peak GPU \
            \(Double(pass.peakMemoryBytes) / bytesPerGibibyte) GiB; \
            \(DecodeRates(chunkTimes: pass.events.chunkTimes).summary)
            """
    }

    /// Holds the scale, the speculation, the prompt-cache reuse and the speed of the sessions.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectSessions(_ sessions: Sessions) throws {
        let label = "\(sessions.path)"
        for passes in [sessions.plain, sessions.mtp] {
            let first = try #require(passes.first)
            #expect(first.events.renderedTokenCount >= Self.minimumPromptTokenCount, "\(label)")
            // Each follow-up reuses the whole ledger of the turn before it, up to the seam of a
            // tool call, thus it feeds only its new messages.
            for index in passes.indices.dropFirst() {
                let (previous, next) = (passes[index - 1], passes[index])
                #expect(
                    next.events.cachedTokenCount >= previous.ledger.count - Self.ledgerSeamSlack,
                    """
                    \(label) turn \(index + 1) reused \(next.events.cachedTokenCount) tokens of \
                    a \(previous.ledger.count)-token ledger, and fed \(next.fedTokenCount)
                    """)
            }
            let long = try #require(passes.last)
            #expect(
                long.completion.generationTokenCount >= Self.minimumLongDecodeTokenCount,
                "\(label): the long decode wrote \(long.completion.generationTokenCount) tokens")
        }
        for (index, pass) in sessions.mtp.enumerated() {
            let turn = "\(label) turn \(index + 1)"
            #expect((pass.completion.proposedDraftTokens ?? 0) > 0, "\(turn) drafted nothing")
            #expect(pass.completion.passthroughReason == nil, "\(turn)")
            #expect(pass.completion.speculativeDecodingFallbackReason == nil, "\(turn)")
            // The drafter state began with the first turn and was resumed, never started cold.
            #expect(pass.drafterStartPosition == 0, "\(turn)")
        }
        let plainLong = try #require(sessions.plain.last).completion.tokensPerSecond
        let mtpLong = try #require(sessions.mtp.last).completion.tokensPerSecond
        #expect(mtpLong > plainLong, "\(label): MTP \(mtpLong) tok/s, plain \(plainLong) tok/s")
    }

    /// Holds that the two sessions write the same turns, or part only at a near tie.
    ///
    /// Up to the first turn where they part, the two sessions render the same prompts and must
    /// feed the same tokens. That turn may part only where plain decode's two best tokens lie
    /// within ``MTPChatTokenMatch/nearTieLogitTolerance`` of each other in an independent
    /// reference prefill, and the two decodes took one each. After a near tie the greedy
    /// decodes follow different prefixes, so a long decode is compared up to its first near tie
    /// alone, and no later turn is compared.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func expectPlainOutput(_ sessions: Sessions, container: ModelContainer) async throws {
        let label = "\(sessions.path)"
        let pairs = Array(zip(sessions.plain, sessions.mtp))
        let divergentTurn = pairs.firstIndex { $0.output != $1.output }
        for (index, (plain, mtp)) in pairs.prefix(through: divergentTurn ?? pairs.count - 1)
            .enumerated()
        {
            #expect(mtp.fedTokenCount == plain.fedTokenCount, "\(label) turn \(index + 1)")
        }
        if let divergentTurn {
            let (plain, mtp) = pairs[divergentTurn]
            let step = commonPrefixLength(plain.ledger, mtp.ledger)
            #expect(step >= plain.renderTokens.count, "\(label): the renders part at \(step)")
            let candidates = try await Self.referenceCandidates(
                container, context: Array(plain.ledger.prefix(step)))
            let match = Self.tokenMatch(
                mtp: mtp.generatedTokens, plain: plain.generatedTokens,
                step: step - plain.renderTokens.count, candidates: candidates)
            Self.log(
                "\(label) turn \(divergentTurn + 1): \(match.summary), reference \(candidates)")
            #expect(match.holds, "\(label) turn \(divergentTurn + 1): \(match.summary)")
        }
    }

    /// How `mtp` compares with `plain` when `candidates` are the two best tokens at `step`.
    ///
    /// The rule of ``MTPChatTokenMatch`` does not ask which of the two tokens the reference
    /// ranks first, thus either side may round a near tie the other way.
    private static func tokenMatch(
        mtp: [Int], plain: [Int], step: Int, candidates: [GenerateTokenLogProbability]
    ) -> MTPChatTokenMatch {
        let tokens = { (ids: [Int], reported: [GenerateTokenLogProbability]) in
            ids.enumerated().map { index, id in
                MTPChatToken(token: id, candidates: index == step ? reported : [])
            }
        }
        return MTPChatTokenMatch(mtp: tokens(mtp, []), plain: tokens(plain, candidates))
    }

    /// The two best tokens, with their log probabilities, after an uncached prefill of `context`
    /// in chunks of ``referenceChunkTokenCount``.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func referenceCandidates(
        _ container: ModelContainer, context: [Int]
    ) async throws -> [GenerateTokenLogProbability] {
        try await container.perform { modelContext in
            let cache = try modelContext.model.newCache(parameters: nil)
            let chunks = stride(from: 0, to: context.count, by: referenceChunkTokenCount).map {
                Array(context[$0 ..< min($0 + referenceChunkTokenCount, context.count)])
            }
            let output = chunks.reduce(nil as LMOutput?) { previous, chunk in
                let output = modelContext.model(
                    LMInput.Text(tokens: MLXArray(chunk.map(Int32.init))[.newAxis, 0...]),
                    cache: cache, state: previous?.state)
                eval(cache.flatMap(\.state))
                return output
            }
            return Qwen38FourBitMTPModel.topCandidates(of: try #require(output).logits[0, -1])
        }
    }

    // MARK: - The context-length sweep

    /// The two models of the sweep of one load path.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    struct SweepRunners {

        /// Plain decode.
        let plain: SessionRunner

        /// MTP: the raw cost of speculation.
        let mtp: SessionRunner
    }

    /// The decode rates of one sweep point, in tokens per second.
    struct SweepPoint {
        let plain: Double
        let mtp: Double
    }

    /// Loads the two models of a sweep on `path`, compiles the kernels of each, and gives
    /// them with the token counter of their tokenizer.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func makeSweepRunners(
        on path: Qwen38FourBitMTPModel.LoadPath
    ) async throws -> (runners: SweepRunners, countTokens: (String) -> Int) {
        // Both load paths name one model id, thus one container of the model cache.
        await releaseAllGPUMemory()
        let runners = SweepRunners(
            plain: try SessionRunner(
                model: try Qwen38FourBitMTPModel.make(
                    speculates: false, reasons: true, path: path),
                label: "\(path) plain"),
            mtp: try SessionRunner(
                model: try Qwen38FourBitMTPModel.make(
                    speculates: true, reasons: true, path: path),
                label: "\(path) MTP"))
        let container = try await runners.plain.model.loadContainer()
        let tokenizer = await container.tokenizer
        for runner in [runners.plain, runners.mtp] {
            try await runner.warmUp()
        }
        return (runners, { tokenizer.encode(text: $0, addSpecialTokens: false).count })
    }

    /// Decodes ``sweepDecodeTokenCount`` tokens with no tools after a history of
    /// `contextTokenCount` content tokens with MTP and plain decode, back to back; logs the MTP
    /// pass against plain decode; and holds that both passes write the whole budget over the
    /// whole context, and that MTP speculates. Nothing is held of the MTP rate: its numbers show
    /// the raw cost.
    ///
    /// The two passes share one session of a store of their own: MTP prefills the history cold,
    /// with its drafter, and the plain pass after it reuses what of that cache it can, which the
    /// log line states as cached tokens.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func measureSweepPoint(
        _ contextTokenCount: Int, runners: SweepRunners, countTokens: (String) -> Int
    ) async throws -> SweepPoint {
        let entries =
            [SWEBenchTranscript.instructions(declaresTool: false)]
            + (try SWEBenchTranscript.history(
                minimumTokenCount: contextTokenCount,
                maximumRoundCount: Self.maximumHistoryRoundCount, countTokens: countTokens))
            + [SWEBenchTranscript.prompt(Self.longDecodePrompt)]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MTPSWEBenchScaleSweep-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ExecutorPromptCacheStore(directory: directory)
        await store.configure(memoryBudgetBytes: Int.max)
        let passes = try await ExecutorPromptCacheStore.$current.withValue(store) {
            let run = { (runner: SessionRunner) in
                try await runner.run(
                    entries, tokenBudget: Self.sweepDecodeTokenCount, declaresTool: false)
            }
            let mtp = try await run(runners.mtp)
            return (plain: try await run(runners.plain), mtp: mtp)
        }
        let label = "\(runners.mtp.label) vs plain, sweep \(contextTokenCount)"
        let mtp = passes.mtp.completion
        Self.log("\(label): \(Self.describeSweep(mtp: passes.mtp, plain: passes.plain))")
        for pass in [passes.plain, passes.mtp] {
            #expect(pass.events.renderedTokenCount >= contextTokenCount, "\(label)")
            #expect(
                pass.completion.generationTokenCount >= Self.sweepDecodeTokenCount,
                "\(label): a pass wrote \(pass.completion.generationTokenCount) tokens")
        }
        #expect((mtp.proposedDraftTokens ?? 0) > 0, "\(label)")
        #expect(mtp.passthroughReason == nil, "\(label)")
        #expect(mtp.speculativeDecodingFallbackReason == nil, "\(label)")
        return SweepPoint(
            plain: passes.plain.completion.tokensPerSecond, mtp: mtp.tokensPerSecond)
    }

    /// The numbers of one MTP pass of a sweep point against plain decode, in a measurement line.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func describeSweep(mtp: Pass, plain: Pass) -> String {
        let telemetry = mtp.completion.speculativeDecodingTelemetry
        let verifyCalls = Double(max(telemetry?.targetModelCallCount ?? 0, 1))
        let rounds = Double(max(telemetry?.roundCount ?? 0, 1))
        let plainTokens = Double(max(plain.completion.generationTokenCount, 1))
        let plainMilliseconds = plain.completion.generateTime / plainTokens * millisecondsPerSecond
        return """
            prompt \(mtp.events.renderedTokenCount) tokens (MTP prefilled \(mtp.fedTokenCount) \
            in \(mtp.completion.promptTime) s, plain reused \(plain.events.cachedTokenCount)); \
            MTP \(mtp.completion.tokensPerSecond) tok/s, plain \
            \(plain.completion.tokensPerSecond) tok/s, ratio \
            \(mtp.completion.tokensPerSecond / plain.completion.tokensPerSecond); acceptance \
            \(telemetry?.acceptanceRate ?? 0), tokens per verify \
            \(telemetry?.meanEmittedTokensPerTargetCall ?? 0), effective block size \
            \(Double(telemetry?.targetVerifiedTokenCount ?? 0) / verifyCalls), \
            \(mtp.completion.generateTime / rounds * millisecondsPerSecond) ms per round over \
            \(telemetry?.roundCount ?? 0) rounds; plain \(plainMilliseconds) ms per token
            """
    }

    /// Writes `measurement` under the measurement prefix.
    static func log(_ measurement: String) {
        measurementLog.info("\(measurementPrefix, privacy: .public) \(measurement, privacy: .public)")
    }
}

#endif
