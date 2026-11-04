// MTP speculative decoding against plain decoding of the same model, one prompt
// at a time.

import Foundation
import MLXLMCommon

/// The share of the drafted tokens the target accepted.
enum DraftAcceptance {
    /// `accepted` divided by `proposed`, or `nil` when nothing was proposed.
    ///
    /// - Parameters:
    ///   - accepted: the drafted tokens the target accepted
    ///   - proposed: the tokens the drafter proposed
    /// - Returns: the acceptance rate, from 0 to 1, or `nil` for no proposal
    static func rate(accepted: Int, proposed: Int) -> Double? {
        guard proposed > 0 else { return nil }
        return Double(accepted) / Double(proposed)
    }
}

/// A prompt for the MTP benchmark, named by the kind of text it asks for.
public struct MTPBenchmarkPrompt: Sendable {
    /// The kind of text the prompt asks for, which names its result.
    public let name: String
    /// The user message.
    public let text: String

    /// Creates a prompt named `name` that asks `text`.
    ///
    /// - Parameters:
    ///   - name: the kind of text the prompt asks for
    ///   - text: the user message
    public init(name: String, text: String) {
        self.name = name
        self.text = text
    }

    /// Math, code and prose, the categories the DFlash2 measurements on Qwen3.8 use.
    public static let defaults = [
        MTPBenchmarkPrompt(
            name: "math",
            text: "A train leaves at 9:40 and arrives at 13:05. It stops twice for 12 minutes. "
                + "How long was it moving? Then: a shop sells 3 pens for $4. How much do 18 pens cost?"
        ),
        MTPBenchmarkPrompt(
            name: "code",
            text: "Write a Swift function that returns the longest common subsequence of two "
                + "strings, with a few unit tests."),
        MTPBenchmarkPrompt(
            name: "prose", text: "Write a short travel blog post about a weekend in Hawaii."),
    ]
}

/// Plain and MTP decode timings for one prompt.
public struct MTPGenerationResult: Sendable {
    /// The ``MTPBenchmarkPrompt/name`` of the prompt.
    public let prompt: String
    /// Tokens the last MTP run generated.
    public let generatedTokenCount: Int
    /// Median plain decode throughput, in tokens per second.
    public let baselineTokensPerSecond: Double
    /// Median MTP decode throughput, in tokens per second.
    public let mtpTokensPerSecond: Double
    /// Accepted drafts divided by proposed drafts, from the last MTP run.
    public let acceptanceRate: Double
    /// Tokens emitted per target verify pass, from the last MTP run. 1 is plain decoding.
    public let acceptanceLength: Double
    /// Whether the MTP tokens equal the plain tokens. Greedy decodes match; sampled decodes
    /// draw their random values in another order, and need not.
    public let matchesBaseline: Bool
    /// How the greedy MTP tokens compare with the plain tokens, near-ties of plain's two best
    /// tokens allowed (``MTPChatTokenMatch``), or `nil` for a sampled decode.
    public let tokenMatch: MTPChatTokenMatch?
    /// The acceptance of each draft depth in the last MTP run: element `k` is the share of the
    /// rounds that drafted a token at depth `k + 1` whose draft there the target accepted.
    public let depthAcceptanceRates: [Double]

    /// MTP throughput divided by plain throughput.
    public var speedup: Double { mtpTokensPerSecond / baselineTokensPerSecond }
}

/// The tokens and the `.info` event of one decode.
private typealias DecodeRun = (tokens: [Int], info: GenerateCompletionInfo)

extension Benchmark {
    /// The tokens each decode of an MTP benchmark generates at most when the caller names
    /// none.
    public static let defaultMTPMaxTokens = 192

    /// The timed runs of each decode of an MTP benchmark when the caller names none.
    public static let defaultMTPRuns = 3

    /// Benchmark MTP speculative decoding against plain decoding of the same model,
    /// greedy unless `sampling` samples. Each prompt runs once to warm up, then `runs`
    /// timed times each way.
    ///
    /// - Parameters:
    ///   - context: The model.
    ///   - drafter: The MTP drafter of the model.
    ///   - prompts: The prompts, each a result.
    ///   - maxTokens: The tokens each decode generates at most.
    ///   - blockSize: The tokens of each MTP round.
    ///   - minimumDraftProbability: The drafter top-1 probability below which a chained round
    ///     stops drafting.
    ///   - runs: The timed runs of each decode; at least one.
    ///   - sampling: The sampling of both decodes; its `maxTokens` is replaced by `maxTokens`.
    ///     A seed makes each run of a decode sample the same tokens.
    /// - Returns: One result for each prompt.
    public static func mtpGeneration(
        context: ModelContext,
        drafter: any MTPDrafterModel,
        prompts: [MTPBenchmarkPrompt] = MTPBenchmarkPrompt.defaults,
        maxTokens: Int = defaultMTPMaxTokens,
        blockSize: Int = SpeculativeDecodingConfig.defaultMTPBlockSize,
        minimumDraftProbability: Float =
            SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability,
        runs: Int = defaultMTPRuns,
        sampling: GenerateParameters = GenerateParameters(temperature: 0)
    ) async throws -> [MTPGenerationResult] {
        precondition(runs > 0, "runs must be greater than zero")
        var parameters = sampling
        parameters.maxTokens = maxTokens

        var results: [MTPGenerationResult] = []
        for prompt in prompts {
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(prompt.text)]))
            func plain() async throws -> DecodeRun {
                try await collect(
                    generateTokens(input: input, parameters: parameters, context: context))
            }
            func mtp() async throws -> DecodeRun {
                try await collect(
                    generateTokens(
                        input: input, parameters: parameters, context: context,
                        mtpDrafter: drafter, blockSize: blockSize,
                        minimumDraftProbability: minimumDraftProbability))
            }

            _ = try await plain()
            _ = try await mtp()

            var plainRates: [Double] = []
            var mtpRates: [Double] = []
            var baseline: [Int] = []
            var lastMTPRun: DecodeRun?
            for _ in 0 ..< runs {
                let plainRun = try await plain()
                baseline = plainRun.tokens
                plainRates.append(plainRun.info.tokensPerSecond)
                let mtpRun = try await mtp()
                mtpRates.append(mtpRun.info.tokensPerSecond)
                lastMTPRun = mtpRun
            }
            guard let lastMTPRun else {
                preconditionFailure("runs > 0 leaves a last MTP run")
            }

            let tokenMatch =
                parameters.temperature == 0
                ? MTPChatTokenMatch(
                    mtp: lastMTPRun.tokens.map { MTPChatToken(token: $0) },
                    plain: try plainTokensWithCandidates(
                        input: input, context: context, parameters: parameters,
                        count: baseline.count))
                : nil
            let result = MTPGenerationResult(
                lastMTPRun, prompt: prompt.name, baseline: baseline,
                plainRates: plainRates, mtpRates: mtpRates, tokenMatch: tokenMatch)
            results.append(result)
            BenchmarkLog.line(
                result.summaryLine(passthroughReason: lastMTPRun.info.passthroughReason))
        }
        return results
    }

    /// The tokens and the `.info` event of `stream`.
    private static func collect(_ stream: AsyncStream<TokenGeneration>) async throws -> DecodeRun {
        var tokens: [Int] = []
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .token(let token): tokens.append(token)
            case .info(let completion): info = completion
            }
        }
        guard let info else { throw LLMGenerationBenchmarkError.missingCompletionInfo }
        return (tokens, info)
    }
}

extension MTPGenerationResult {
    /// The result of `prompt` whose last MTP run is `lastMTPRun`.
    fileprivate init(
        _ lastMTPRun: DecodeRun, prompt: String, baseline: [Int], plainRates: [Double],
        mtpRates: [Double], tokenMatch: MTPChatTokenMatch?
    ) {
        let info = lastMTPRun.info
        let telemetry = info.speculativeDecodingTelemetry
        let rounds = telemetry?.roundCount ?? 0
        self.init(
            prompt: prompt,
            generatedTokenCount: info.generationTokenCount,
            baselineTokensPerSecond: BenchmarkStats(times: plainRates).median,
            mtpTokensPerSecond: BenchmarkStats(times: mtpRates).median,
            acceptanceRate: DraftAcceptance.rate(
                accepted: info.acceptedDraftTokens ?? 0, proposed: info.proposedDraftTokens ?? 0)
                ?? 0,
            acceptanceLength: rounds > 0 ? Double(info.generationTokenCount) / Double(rounds) : 1,
            matchesBaseline: lastMTPRun.tokens == baseline,
            tokenMatch: tokenMatch,
            depthAcceptanceRates: telemetry?.depthAcceptanceRates ?? [])
    }
}
