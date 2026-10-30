// LLM generation and system prompt cache benchmarks.

import Foundation
import MLXLLM
import MLXLMCommon

// MARK: - LLM Generation Benchmarks

/// Combined prefill and decode timing for an LLM generation run.
public struct LLMGenerationStats: Sendable {
    /// Prompt tokens each run prefilled.
    public let promptTokenCount: Int
    /// Tokens each run generated.
    public let generationTokenCount: Int
    /// Prefill latency per run, in milliseconds.
    public let promptTimeStats: BenchmarkStats
    /// Decode latency per run, in milliseconds (covers all generated tokens).
    public let generateTimeStats: BenchmarkStats

    /// Creates the timing of a generation benchmark.
    ///
    /// - Parameters:
    ///   - promptTokenCount: prompt tokens each run prefilled
    ///   - generationTokenCount: tokens each run generated
    ///   - promptTimeStats: prefill latency per run, in milliseconds
    ///   - generateTimeStats: decode latency per run, in milliseconds
    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTimeStats: BenchmarkStats,
        generateTimeStats: BenchmarkStats
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTimeStats = promptTimeStats
        self.generateTimeStats = generateTimeStats
    }

    /// Median prefill throughput in tokens per second.
    public var prefillTokensPerSecond: Double {
        Double(promptTokenCount) / (promptTimeStats.median / Benchmark.millisecondsPerSecond)
    }

    /// Median decode throughput in tokens per second.
    public var decodeTokensPerSecond: Double {
        Double(generationTokenCount) / (generateTimeStats.median / Benchmark.millisecondsPerSecond)
    }

    /// Writes the prefill and decode throughput under `label`.
    ///
    /// - Parameter label: the name of the benchmark in the first line.
    public func logSummary(label: String) {
        BenchmarkLog.line("\(label) results:")
        BenchmarkLog.line(
            "  Prefill: \(promptTokenCount) tokens, "
                + "\(BenchmarkLog.hundredths(prefillTokensPerSecond)) tok/s "
                + "(median \(BenchmarkLog.decimal(promptTimeStats.median))ms, "
                + "stddev \(BenchmarkLog.decimal(promptTimeStats.stdDev))ms)"
        )
        BenchmarkLog.line(
            "  Decode:  \(generationTokenCount) tokens, "
                + "\(BenchmarkLog.hundredths(decodeTokensPerSecond)) tok/s "
                + "(median \(BenchmarkLog.decimal(generateTimeStats.median))ms, "
                + "stddev \(BenchmarkLog.decimal(generateTimeStats.stdDev))ms)"
        )
    }
}

/// Why a generation benchmark could not measure a run.
public enum LLMGenerationBenchmarkError: LocalizedError {
    /// The generation ended without its `GenerateCompletionInfo`.
    case missingCompletionInfo

    /// A description of the failure for the benchmark output.
    public var errorDescription: String? {
        switch self {
        case .missingCompletionInfo:
            return
                "LLM generation stream completed without emitting GenerateCompletionInfo."
        }
    }
}

extension Benchmark {
    /// Benchmark prefill and decode performance for an LLM. Performs a warm-up run
    /// (loading + first generation) to reach steady-state, then measures `runs` timed
    /// trials and reports prefill/decode throughput.
    ///
    /// Uses ``BenchmarkDefaults/textSource`` for the prompt by default; pass an
    /// explicit `prompt` to override. `temperature` defaults to 0 so trials are
    /// deterministic and tps comparisons are reproducible across runs.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches the model
    ///   - tokenizerLoader: the loader of the tokenizer of the model
    ///   - modelId: the model to load
    ///   - prompt: the prompt, or `nil` for `promptCharacterCount` characters of
    ///     benchmark text
    ///   - promptCharacterCount: the characters of benchmark text a `nil` prompt reads
    ///   - maxTokens: the tokens each run generates at most
    ///   - runs: the timed runs; at least one
    /// - Returns: the prefill and decode timing of the runs
    public static func llmGeneration(
        from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader,
        modelId: String = BenchmarkDefaults.languageModelID,
        prompt: String? = nil,
        promptCharacterCount: Int = 4_000,
        maxTokens: Int = 64,
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> LLMGenerationStats {
        let configuration = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration
        ) { _ in }
        return try await llmGeneration(
            container: container, prompt: prompt, promptCharacterCount: promptCharacterCount,
            maxTokens: maxTokens, runs: runs)
    }

    /// Benchmark prefill and decode performance for an already loaded model.
    ///
    /// - Parameters:
    ///   - container: the model
    ///   - prompt: the prompt, or `nil` for `promptCharacterCount` characters of
    ///     benchmark text
    ///   - promptCharacterCount: the characters of benchmark text a `nil` prompt reads
    ///   - maxTokens: the tokens each run generates at most
    ///   - runs: the timed runs; at least one
    /// - Returns: the prefill and decode timing of the runs
    public static func llmGeneration(
        container: ModelContainer,
        prompt: String? = nil,
        promptCharacterCount: Int = 4_000,
        maxTokens: Int = 64,
        runs: Int = BenchmarkDefaults.loadingRuns
    ) async throws -> LLMGenerationStats {
        precondition(runs > 0, "runs must be greater than zero")
        let benchmarkPrompt: String
        if let prompt {
            benchmarkPrompt = prompt
        } else {
            benchmarkPrompt = try await loadText(characterCount: promptCharacterCount)
        }

        let parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)

        // UserInput is `sending`, so each call builds a fresh instance.
        func startGeneration() async throws -> AsyncStream<Generation> {
            let input = try await container.prepare(input: UserInput(prompt: benchmarkPrompt))
            return try await container.generate(input: input, parameters: parameters)
        }

        // Warm-up: prime caches and weights.
        for await _ in try await startGeneration() {}

        var promptTimes: [Double] = []
        var generateTimes: [Double] = []
        var promptTokenCount = 0
        var generationTokenCount = 0

        for run in 1 ... runs {
            var info: GenerateCompletionInfo?
            for await item in try await startGeneration() {
                if case .info(let completion) = item { info = completion }
            }
            guard let completion = info else {
                throw LLMGenerationBenchmarkError.missingCompletionInfo
            }
            promptTokenCount = completion.promptTokenCount
            generationTokenCount = completion.generationTokenCount
            let promptMs = completion.promptTime * millisecondsPerSecond
            let generateMs = completion.generateTime * millisecondsPerSecond
            promptTimes.append(promptMs)
            generateTimes.append(generateMs)
            BenchmarkLog.line(
                "Generation run \(run): "
                    + "prefill \(completion.promptTokenCount) tok / "
                    + "\(BenchmarkLog.decimal(promptMs))ms "
                    + "(\(BenchmarkLog.hundredths(completion.promptTokensPerSecond)) tok/s), "
                    + "decode \(completion.generationTokenCount) tok / "
                    + "\(BenchmarkLog.decimal(generateMs))ms "
                    + "(\(BenchmarkLog.hundredths(completion.tokensPerSecond)) tok/s)"
            )
        }

        return LLMGenerationStats(
            promptTokenCount: promptTokenCount,
            generationTokenCount: generationTokenCount,
            promptTimeStats: BenchmarkStats(times: promptTimes),
            generateTimeStats: BenchmarkStats(times: generateTimes)
        )
    }
}

// MARK: - Chat Turns

/// What one turn of a ``ChatSession`` streamed: its text, its `.info` event, and
/// when its first text arrived.
struct ChatTurnTranscript {
    /// The generated text.
    let text: String
    /// The `.info` event of the turn.
    let info: GenerateCompletionInfo
    /// The milliseconds from the start of the turn to its first text, or `nil`
    /// when the turn generated no text.
    let firstChunkMs: Double?
    /// The milliseconds from the start of the turn to its end.
    let totalMs: Double

    /// Drains `stream`, handing each event to `observe` before it collects the
    /// text and the `.info` event. The times count from the call.
    ///
    /// - Parameters:
    ///   - stream: the events of the turn
    ///   - observe: reads each event, such as a recorded token, before the transcript does
    /// - Returns: the transcript of the turn
    /// - Throws: ``LLMGenerationBenchmarkError/missingCompletionInfo`` when the turn
    ///   ends without its `.info` event, or the error of the stream
    static func collect<Event: GenerationProviding>(
        _ stream: AsyncThrowingStream<Event, Error>,
        observing observe: (Event) -> Void = { _ in }
    ) async throws -> ChatTurnTranscript {
        let start = CFAbsoluteTimeGetCurrent()
        var text = ""
        var firstChunkMs: Double?
        var info: GenerateCompletionInfo?
        for try await event in stream {
            observe(event)
            switch event.generation {
            case .chunk(let chunk)?:
                text += chunk
                firstChunkMs = firstChunkMs ?? Benchmark.millisecondsSince(start)
            case .info(let completion)?:
                info = completion
            case .toolCall?, .rejectedToolCall?, nil:
                break
            }
        }
        guard let info else { throw LLMGenerationBenchmarkError.missingCompletionInfo }
        return ChatTurnTranscript(
            text: text, info: info, firstChunkMs: firstChunkMs,
            totalMs: Benchmark.millisecondsSince(start))
    }
}

// MARK: - System Prompt Cache Benchmarks

/// The system prompt of a prompt cache benchmark: a preamble followed by benchmark text.
enum BenchmarkSystemPrompt {
    /// The preamble of ``Benchmark/systemPromptCache(container:systemPrompt:systemPromptCharacterCount:coldQuestion:warmQuestion:maxTokens:kvCache:additionalContext:runs:)``.
    static let textQuestionsPreamble = "Answer questions about this text."

    /// The preamble of ``Benchmark/mtpSystemPromptCache(container:drafter:scenarios:systemPrompt:systemPromptCharacterCount:blockSize:adaptivePolicy:)``,
    /// whose turns ask math questions.
    static let mathTutorPreamble =
        "You are a careful math tutor. Answer the questions concisely and show brief working. "
        + "The text below is background reading only."

    /// Returns `systemPrompt` when given, else `preamble` followed by the first
    /// `characterCount` characters of the benchmark text.
    static func load(
        _ systemPrompt: String?, preamble: String, characterCount: Int
    ) async throws -> String {
        if let systemPrompt {
            return systemPrompt
        }
        let text = try await Benchmark.loadText(characterCount: characterCount)
        return preamble + "\n\n" + text
    }
}

/// One turn of a system prompt cache run.
public struct PromptCacheTurn: Sendable {
    /// Prompt tokens that were prefilled.
    public let promptTokenCount: Int
    /// Prompt tokens served from the reused cache prefix.
    public let cachedPromptTokenCount: Int
    /// Prefill time reported by the generation, in milliseconds.
    public let promptTimeMs: Double
    /// Wall-clock time from the call to the first generated chunk, in milliseconds.
    public let firstChunkMs: Double

    /// The share of the rendered prompt the reused cache served, from 0 to 1.
    var cachedShare: Double {
        let total = promptTokenCount + cachedPromptTokenCount
        return total > 0 ? Double(cachedPromptTokenCount) / Double(total) : 0
    }
}

/// Timings for sessions that share a cached system prompt.
public struct PromptCacheStats: Sendable {
    /// First turn of a new session: the whole prompt is prefilled.
    public let cold: [PromptCacheTurn]
    /// First turn of a session forked with an empty history: only the system prompt can match.
    public let fork: [PromptCacheTurn]
    /// Second turn of the cold session: the system prompt and the first turn can match.
    public let followUp: [PromptCacheTurn]

    /// Creates the timings of the turns of each kind, one turn for each run.
    ///
    /// - Parameters:
    ///   - cold: the first turn of each new session
    ///   - fork: the first turn of each fork with an empty history
    ///   - followUp: the second turn of each new session
    public init(cold: [PromptCacheTurn], fork: [PromptCacheTurn], followUp: [PromptCacheTurn]) {
        self.cold = cold
        self.fork = fork
        self.followUp = followUp
    }

    /// Writes one line for each kind of turn under `label`: the cached share of
    /// its last turn, and its median prefill against the cold median.
    ///
    /// - Parameter label: the name of the benchmark in the first line.
    public func logSummary(label: String) {
        BenchmarkLog.line("\(label) system prompt cache results:")
        guard !cold.isEmpty else {
            BenchmarkLog.warning("\(label): no cold turn was recorded, so no turn has a baseline")
            return
        }
        let coldMedian = BenchmarkStats(times: cold.map(\.promptTimeMs)).median
        for (name, turns) in [("Cold", cold), ("Fork", fork), ("Follow-up", followUp)] {
            guard let lastTurn = turns.last else {
                BenchmarkLog.warning("  \(name): no turn was recorded")
                continue
            }
            let prefill = BenchmarkStats(times: turns.map(\.promptTimeMs))
            let first = BenchmarkStats(times: turns.map(\.firstChunkMs))
            BenchmarkLog.line(
                "  \(name): prefilled \(lastTurn.promptTokenCount) tok, "
                    + "cached \(lastTurn.cachedPromptTokenCount) tok "
                    + "(\(BenchmarkLog.decimal(lastTurn.cachedShare * 100))%), "
                    + "prefill median \(BenchmarkLog.decimal(prefill.median))ms "
                    + "(stddev \(BenchmarkLog.decimal(prefill.stdDev))ms, "
                    + "\(BenchmarkLog.decimal(coldMedian / prefill.median))x vs cold), "
                    + "first chunk median \(BenchmarkLog.decimal(first.median))ms"
            )
        }
    }
}

extension Benchmark {
    /// Benchmark the reuse of a cached system prompt. Each run prefills the system
    /// prompt cold in a new `ChatSession`. It then asks a second question two ways:
    /// in a fork of the session with an empty history, and as a follow-up turn of the
    /// session itself.
    ///
    /// The questions differ, so a fork can match only the system prompt.
    ///
    /// - Parameters:
    ///   - container: the model
    ///   - systemPrompt: the system prompt, or `nil` for a preamble and
    ///     `systemPromptCharacterCount` characters of benchmark text
    ///   - systemPromptCharacterCount: the characters of benchmark text a `nil`
    ///     system prompt reads
    ///   - coldQuestion: the first question of each session
    ///   - warmQuestion: the question of each fork and each follow-up turn
    ///   - maxTokens: the tokens each turn generates at most
    ///   - kvCache: the cache configuration of the sessions, or `nil` for the default
    ///   - additionalContext: the chat template context of the sessions
    ///   - runs: the timed runs; at least one
    /// - Returns: the timings of the turns of each kind
    public static func systemPromptCache(
        container: ModelContainer,
        systemPrompt: String? = nil,
        systemPromptCharacterCount: Int = 16_000,
        coldQuestion: String = "In one sentence, who is the main character?",
        warmQuestion: String = "In one sentence, where does the story begin?",
        maxTokens: Int = 16,
        kvCache: KVCacheConfiguration? = nil,
        additionalContext: [String: any Sendable]? = nil,
        runs: Int = 5
    ) async throws -> PromptCacheStats {
        precondition(runs > 0, "runs must be greater than zero")
        let instructions = try await BenchmarkSystemPrompt.load(
            systemPrompt, preamble: BenchmarkSystemPrompt.textQuestionsPreamble,
            characterCount: systemPromptCharacterCount)
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.kvCache = kvCache

        func run() async throws -> (
            cold: PromptCacheTurn, fork: PromptCacheTurn, followUp: PromptCacheTurn
        ) {
            let session = ChatSession(
                container, instructions: instructions, generateParameters: parameters,
                additionalContext: additionalContext)
            let cold = try await promptCacheTurn(in: session, asking: coldQuestion)
            let fork = try await promptCacheTurn(
                in: await session.fork(history: []), asking: warmQuestion)
            let followUp = try await promptCacheTurn(in: session, asking: warmQuestion)
            return (cold, fork, followUp)
        }

        // Warm-up run, not recorded.
        _ = try await run()

        var coldTurns: [PromptCacheTurn] = []
        var forkTurns: [PromptCacheTurn] = []
        var followUpTurns: [PromptCacheTurn] = []
        for runIndex in 1 ... runs {
            let (cold, fork, followUp) = try await run()
            coldTurns.append(cold)
            forkTurns.append(fork)
            followUpTurns.append(followUp)
            BenchmarkLog.line(
                "Prompt cache run \(runIndex): "
                    + "cold \(cold.promptTokenCount) tok / "
                    + "\(BenchmarkLog.decimal(cold.promptTimeMs))ms, "
                    + "fork \(fork.promptTokenCount) tok (cached \(fork.cachedPromptTokenCount)) / "
                    + "\(BenchmarkLog.decimal(fork.promptTimeMs))ms, "
                    + "follow-up \(followUp.promptTokenCount) tok "
                    + "(cached \(followUp.cachedPromptTokenCount)) / "
                    + "\(BenchmarkLog.decimal(followUp.promptTimeMs))ms"
            )
        }

        return PromptCacheStats(cold: coldTurns, fork: forkTurns, followUp: followUpTurns)
    }

    /// Asks `question` in `session` and times its prefill and its first text.
    private static func promptCacheTurn(
        in session: ChatSession, asking question: String
    ) async throws -> PromptCacheTurn {
        let transcript = try await ChatTurnTranscript.collect(session.streamDetails(to: question))
        if transcript.firstChunkMs == nil {
            BenchmarkLog.warning("the turn produced no text, so the session does not record it")
        }
        return PromptCacheTurn(
            promptTokenCount: transcript.info.promptTokenCount,
            cachedPromptTokenCount: transcript.info.cachedPromptTokenCount,
            promptTimeMs: transcript.info.promptTime * millisecondsPerSecond,
            firstChunkMs: transcript.firstChunkMs ?? transcript.totalMs)
    }
}
