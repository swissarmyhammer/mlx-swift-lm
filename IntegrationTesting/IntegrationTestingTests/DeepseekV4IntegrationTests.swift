// Copyright © 2026 Apple Inc.
//
// Real-weights integration tests for `mlx-community/DeepSeek-V4-Flash-4bit`,
// from card ^e7b24ws. The synthetic-weight unit tests in `Tests/MLXLMTests`
// prove the math. These tests prove the port against the published
// checkpoint: the full `LLMModelFactory` load path, greedy-token parity
// against the Python reference, chat and thinking generation, one agentic tool
// round, long-generation stability past 12k tokens (the `ml-explore/mlx-lm`
// issue-1662 landmine), and two-round conversation behavior.
//
// The checkpoint is a 284B-total / 13B-active MoE. Its weight files hold
// 151,482,475,612 bytes, which is 141 GiB, measured on the published snapshot
// on 2026-08-13. The tests never download it. Each real-weights test skips,
// with a message that says why, when the machine has too little memory or when
// no local copy of the checkpoint exists. The two encoder-level cache tests at
// the end run everywhere, because they read no weights.
//
// The card names `Tests/MLXLMIntegrationTests/` as the location. That target
// does not exist, and the root package holds no swift-transformers
// dependency by design, thus a root test target cannot load the real
// tokenizer. This suite therefore follows the real-weights pattern of this
// project (see `MiniMaxM3CacheIntegrationTests`). Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests`
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`, thus `swift build --build-tests` stays at exit 0 with
// a type error in this file. Measured on 2026-08-13 with a deliberate type
// error: `swift build --build-tests` gave exit 0, and
// `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
// gave exit 65. Use that xcodebuild command as the compile evidence for any
// change to this file.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

// MARK: - Gating constants

// The checkpoint location, the Metal wired limit and the one shared load live
// in `DeepseekV4SharedCheckpoint.swift`, because a second suite in this target
// awaits the same 141 GiB load.

/// The number of decoder layers of the published checkpoint.
private let deepseekV4LayerCount = 43

/// The number of tokens the long-generation test asks for. It is past the
/// 12,288-token bound of the card, and well past the ~11.5k tokens at which
/// the mlx-lm issue-1662 buffer leak kills the process.
private let longGenerationTokenCount = 12_400

/// The bound that the long generation must pass: 12k tokens.
private let longGenerationMinimumTokenCount = 12_288

/// The per-test time limit of this suite, in minutes.
private let suiteTimeLimitMinutes = 240

/// The number of seconds in one minute.
private let secondsPerMinute = 60.0

/// The share of the per-test time limit that decode may take: three quarters.
/// The remaining quarter covers the weight load, the prompt, and the variance
/// of one machine against another.
private let longGenerationLimitShare = 0.75

/// The most seconds one decode step may take, so that the
/// ``longGenerationTokenCount``-token run fits inside its share of the per-test
/// time limit: 0.87 s.
///
/// Measured on an M3 Ultra (512 GiB) through this suite, which builds for
/// debug: the steady decode step is 0.593 s with the wired limit raised and
/// 2.124 s without it. The bound therefore holds with a margin of 1.5, and it
/// fails at once when the limit is not raised.
private let maximumSecondsPerDecodeStep =
    Double(suiteTimeLimitMinutes) * secondsPerMinute * longGenerationLimitShare
    / Double(longGenerationTokenCount)

/// The number of decode steps the speed test measures. The first step of a
/// generation runs the whole prompt, thus the test asks for one step more and
/// drops the first.
private let decodeSpeedSampleTokenCount = 16

/// The number of tokens each short generation asks for.
private let shortGenerationTokenCount = 48

/// The number of tokens the tool-call test may produce. A DSML block for one
/// call with one parameter is far shorter than this.
private let toolCallGenerationTokenCount = 128

/// The name of the one tool the tool-call test offers.
private let stockToolName = "get_stock_level"

/// The name of the one parameter of ``stockToolSpec``. The model must write
/// this name, because the `## Tools` section of the prompt states it.
private let stockToolParameterName = "bay"

/// The bay the tool-call test asks the model to read.
private let stockToolBayName = "bay 7"

/// The system turn of the tool-call test.
///
/// The published reference always carries a real system prompt in front of its
/// `## Tools` section, thus the test carries one as well.
private let stockAgentInstructions =
    "You are an inventory agent. You answer with the tools you are given."

/// The one tool the tool-call test offers, as the Swift dictionary a caller
/// writes. `DeepSeekV4ChatEncoder` renders it into the `## Tools` section, and
/// `DSMLToolCallParser` reads the call back out.
private let stockToolSpec: ToolSpec = [
    "type": "function",
    "function": [
        "name": stockToolName,
        "description": "Read the recorded stock level of one warehouse bay",
        "parameters": [
            "type": "object",
            "properties": [
                stockToolParameterName: [
                    "type": "string",
                    "description": "The bay to read, for example bay 7",
                ] as [String: any Sendable]
            ] as [String: any Sendable],
            "required": [stockToolParameterName],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

/// The number of leading fixture tokens the parity test compares.
///
/// The user pinned this window to 32 on 2026-08-12. The measured stable
/// window on these weights is 36 tokens: at generation step 36 the Python
/// reference holds an exact bf16 tie between its top two candidates
/// (both at logit 29.75, the Swift pick 0.375 below), thus every token
/// from there on is a coin flip that floating-point noise decides. An
/// exact 64-token match is not a stable target across two
/// implementations; the first 32 tokens are.
private let parityTokenCount = 32

/// The number of tokens each turn of the two-round test asks for.
private let recallGenerationTokenCount = 60

/// The number that the two-round test asks the model to remember.
private let recallNumber = "4172"

// MARK: - Parity fixture

/// The greedy-parity fixture, produced by the Python reference
/// (`ml-explore/mlx-lm` PR 1189) against the same checkpoint, with one full
/// forward over the growing sequence for each step:
///
/// ```python
/// from mlx_lm import load
/// import json, mlx.core as mx
/// model, tokenizer = load("mlx-community/DeepSeek-V4-Flash-4bit")
/// prompt = tokenizer.encode("<|user|>Write one sentence about the sea.<|assistant|><think>")
/// sequence, ids = list(prompt), []
/// for _ in range(64):
///     logits = model(mx.array(sequence)[None], cache=model.make_cache())
///     token = int(mx.argmax(logits[0, -1].astype(mx.float32)))
///     ids.append(token)
///     sequence.append(token)
/// json.dump({"prompt_token_ids": prompt, "generated_token_ids": ids},
///           open("deepseek-v4-flash-4bit-greedy-parity.json", "w"))
/// ```
///
/// Do NOT produce the fixture with `mlx_lm.generate.generate_step`. Its
/// cached S=1 decode path diverges from the model's own full-prompt
/// forward (measured 2026-08-12 on these weights: step-0 argmax 671
/// against 455, with a 2.0-logit margin in the full forward). The
/// full-forward stream is self-consistent, and the Swift `TokenIterator`
/// reproduces it token for token.
private struct DeepseekV4ParityFixture: Decodable {
    /// The prompt token identifiers the reference fed the model.
    let promptTokenIDs: [Int]
    /// The greedy token identifiers the reference generated.
    let generatedTokenIDs: [Int]

    private enum CodingKeys: String, CodingKey {
        case promptTokenIDs = "prompt_token_ids"
        case generatedTokenIDs = "generated_token_ids"
    }
}

/// The on-disk location of the parity fixture, next to this source file.
private var deepseekV4ParityFixtureURL: URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures", isDirectory: true)
        .appendingPathComponent("deepseek-v4-flash-4bit-greedy-parity.json")
}

// MARK: - The suite

/// Real-weights integration tests for the published DeepSeek-V4-Flash-4bit
/// checkpoint. See the file header for the gating rules and the run command.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct DeepseekV4IntegrationTests {

    /// Runs one generation and collects the text chunks.
    private func generateText(
        container: LLModelContainer, input: UserInput, maxTokens: Int
    ) async throws -> String {
        try await container.perform(nonSendable: input) { context, input in
            let lmInput = try await context.processor.prepare(input: input)
            let stream = try MLXLMCommon.generate(
                input: lmInput,
                parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
                context: context)
            var text = ""
            for try await generation in stream {
                if case .chunk(let chunk) = generation {
                    text += chunk
                }
            }
            return text
        }
    }

    // MARK: Load

    /// The whole load path — configuration decode, type registry, weight
    /// load, `sanitize`, quantization — completes against the real
    /// checkpoint. `loadWeights` verifies with `[.all]`, thus a completed
    /// load proves that no weight key is missing and none is unexpected.
    @Test func loadsTheRealCheckpointEndToEnd() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "loadsTheRealCheckpointEndToEnd")
        else { return }

        let (isDeepseekV4, layerCount) = await container.perform { context in
            guard let model = context.model as? DeepSeekV4Model else {
                return (false, 0)
            }
            return (true, model.kvHeads.count)
        }
        #expect(isDeepseekV4, "the type registry must map deepseek_v4 to DeepSeekV4Model")
        #expect(layerCount == deepseekV4LayerCount)
    }

    // MARK: The Metal wired limit

    /// The load raises the Metal wired limit, the manager applies the whole
    /// request, and the limit covers every weight buffer. Without this test a
    /// change that drops the call to ``raiseWiredMemoryLimit()``, or that asks
    /// for too few bytes, still builds and still passes every other assertion,
    /// and the only symptom is the >12k-token test running for hours.
    @Test func wiredMemoryLimitCoversTheWholeCheckpoint() async throws {
        guard
            let load = await deepseekV4LoadResultOrSkip(
                testName: "wiredMemoryLimitCoversTheWholeCheckpoint")
        else { return }

        guard let wiredMemory = load.wiredMemory else {
            Issue.record(
                "the load must raise the Metal wired limit before it allocates a weight")
            return
        }
        #expect(
            wiredMemory.isFullyApplied,
            """
            the wired-memory manager applied \(wiredMemory.appliedBytes) bytes of the \
            \(wiredMemory.requestedBytes) bytes asked for, thus weight buffers stay \
            outside the Metal residency set
            """)
        #expect(
            wiredMemory.requestedBytes >= deepseekV4CheckpointBytes,
            """
            the wired limit of \(wiredMemory.requestedBytes) bytes must cover the whole \
            \(deepseekV4CheckpointBytes)-byte checkpoint
            """)
    }

    /// The median decode step stays inside ``maximumSecondsPerDecodeStep``,
    /// thus the 12,400-token run of
    /// ``longGenerationPastTwelveThousandTokensCompletes()`` fits inside the
    /// 240-minute suite limit. It fails in seconds when the wired limit is not
    /// raised, where the >12k-token test needs hours.
    ///
    /// The test measures the steady state. It drops the first step, which runs
    /// the whole prompt, and it takes the median of the steps that follow, thus
    /// one slow step does not decide the result. Measured on an M3 Ultra, each
    /// steady step sits within 6% of the median, thus the median times the
    /// token count estimates the whole run.
    @Test func decodeStepStaysInsideTheLongGenerationBudget() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "decodeStepStaysInsideTheLongGenerationBudget")
        else { return }

        let stepSeconds = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user("Count upward from one, forever.")]))
            var iterator = try TokenIterator(
                input: input, model: context.model,
                parameters: GenerateParameters(
                    maxTokens: decodeSpeedSampleTokenCount + 1, temperature: 0))
            _ = iterator.next()
            var seconds: [Double] = []
            while seconds.count < decodeSpeedSampleTokenCount {
                let startTime = Date()
                guard iterator.next() != nil else { break }
                seconds.append(Date().timeIntervalSince(startTime))
            }
            return seconds
        }

        try #require(
            stepSeconds.count == decodeSpeedSampleTokenCount,
            "the generation must run \(decodeSpeedSampleTokenCount) decode steps")
        let medianSeconds = stepSeconds.sorted()[stepSeconds.count / 2]
        print("Decode steps: \(stepSeconds) s, median \(medianSeconds) s")
        #expect(
            medianSeconds <= maximumSecondsPerDecodeStep,
            """
            the median decode step took \(medianSeconds) s, above the \
            \(maximumSecondsPerDecodeStep) s budget; a Metal wired limit that is not \
            raised before the weight load is the known cause, see raiseWiredMemoryLimit()
            """)
    }

    // MARK: Greedy parity

    /// The first ``parityTokenCount`` greedy token identifiers match the
    /// Python reference exactly. Greedy decode is deterministic inside the
    /// stable window, thus an exact match there is a real parity check.
    /// The fixture feeds the reference's own prompt token identifiers, thus
    /// this test isolates model parity from prompt encoding.
    @Test func greedyFirstTokensMatchThePythonFixture() async throws {
        let fixtureURL = deepseekV4ParityFixtureURL
        guard let fixtureData = try? Data(contentsOf: fixtureURL) else {
            print(
                "Skipping greedyFirstTokensMatchThePythonFixture: no fixture at "
                    + "\(fixtureURL.path). Generate it with the Python reference "
                    + "(ml-explore/mlx-lm PR 1189) — see DeepseekV4ParityFixture "
                    + "in this file for the exact script and schema.")
            return
        }
        let fixture = try JSONDecoder().decode(DeepseekV4ParityFixture.self, from: fixtureData)
        let expected = Array(fixture.generatedTokenIDs.prefix(parityTokenCount))
        #expect(
            expected.count == parityTokenCount,
            "the fixture must carry at least \(parityTokenCount) generated ids")
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "greedyFirstTokensMatchThePythonFixture")
        else { return }

        let promptTokenIDs = fixture.promptTokenIDs
        let generated = try await container.perform { context in
            let input = LMInput(tokens: MLXArray(promptTokenIDs.map(Int32.init)))
            var iterator = try TokenIterator(
                input: input, model: context.model,
                parameters: GenerateParameters(maxTokens: parityTokenCount, temperature: 0))
            var tokens: [Int] = []
            while tokens.count < parityTokenCount, let token = iterator.next() {
                tokens.append(token)
            }
            return tokens
        }

        #expect(
            generated == expected,
            "greedy decode must reproduce the Python reference token for token")
    }

    // MARK: Chat and thinking modes

    /// Both generation modes render through the encoder and generate. The
    /// default mode is thinking; `additionalContext: ["thinking": false]`
    /// selects chat mode.
    @Test func chatAndThinkingModesBothGenerate() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "chatAndThinkingModesBothGenerate")
        else { return }

        let prompt = "Write one short sentence about the sea."
        let thinkingOutput = try await generateText(
            container: container,
            input: UserInput(chat: [.user(prompt)]),
            maxTokens: shortGenerationTokenCount)
        let chatOutput = try await generateText(
            container: container,
            input: UserInput(
                chat: [.user(prompt)], additionalContext: ["thinking": false]),
            maxTokens: shortGenerationTokenCount)

        print("Thinking output: \(thinkingOutput)")
        print("Chat output: \(chatOutput)")
        #expect(!thinkingOutput.isEmpty, "thinking mode must generate text")
        #expect(!chatOutput.isEmpty, "chat mode must generate text")
    }

    // MARK: One tool round

    /// A short prompt that offers one tool makes the model write one DSML
    /// call, and `DSMLToolCallParser` reads it back.
    ///
    /// Measured on 2026-08-13, before the member order of the rendered tool
    /// schema was fixed: the model answered with the plain JSON
    /// `{"function": "get_stock_level", "params": {"bay_id": "bay_7"}}`, which
    /// the parser reads none of, thus no tool round completed. The schema of
    /// that run came out of `JSONSerialization`, which writes a Swift
    /// `Dictionary` in its hash order, thus the `## Tools` section carried a
    /// shape the model never saw in training.
    @Test func aShortToolPromptEmitsOneDSMLToolCall() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "aShortToolPromptEmitsOneDSMLToolCall")
        else { return }

        let session = ChatSession(
            container,
            instructions: stockAgentInstructions,
            generateParameters: GenerateParameters(
                maxTokens: toolCallGenerationTokenCount, temperature: 0),
            additionalContext: ["thinking": false],
            tools: [stockToolSpec])

        var text = ""
        var calls: [ToolCall] = []
        let prompt =
            "Call the \(stockToolName) tool for \(stockToolBayName). "
            + "Call the tool before you write an answer."
        for try await generation in session.streamDetails(to: prompt) {
            switch generation {
            case .chunk(let chunk):
                text += chunk
            case .toolCall(let call):
                calls.append(call)
            case .info:
                break
            }
        }

        print("Tool round text: <<<\(text)>>>")
        print("Tool round calls: \(calls)")
        let call = try #require(
            calls.first,
            """
            the model must write one DSML tool call for the round to complete. \
            It wrote: <<<\(text)>>>
            """)
        #expect(call.function.name == stockToolName)
        #expect(
            call.function.arguments[stockToolParameterName] != nil,
            """
            the call must name its argument \(stockToolParameterName), which the \
            ## Tools section of the prompt states. It named \
            \(call.function.arguments.keys.sorted()).
            """)
    }

    // MARK: Long-generation stability

    /// A generation past 12k tokens completes. This is the regression guard
    /// for `ml-explore/mlx-lm` issue 1662: a mis-wired cache leaks one Metal
    /// buffer per layer per token, and the process deterministically dies at
    /// about 11.5k generated tokens. `TokenIterator.next()` never stops at
    /// an end-of-sentence token, thus the run length is deterministic.
    ///
    /// The test asserts completion and records the memory growth. It states
    /// no byte bound, because the leak's symptom is the abort itself, and a
    /// byte bound cannot be calibrated without the weights on hand.
    @Test func longGenerationPastTwelveThousandTokensCompletes() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "longGenerationPastTwelveThousandTokensCompletes")
        else { return }

        let generatedCount = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user("Count upward from one, forever.")]))
            var iterator = try TokenIterator(
                input: input, model: context.model,
                parameters: GenerateParameters(
                    maxTokens: longGenerationTokenCount, temperature: 0))
            let startMemory = Memory.snapshot()
            let startTime = Date()
            var count = 0
            while iterator.next() != nil {
                count += 1
            }
            let elapsed = Date().timeIntervalSince(startTime)
            let endMemory = Memory.snapshot()
            print(
                "Long generation: \(count) tokens in \(elapsed) s "
                    + "(\(Double(count) / elapsed) tokens/s), active memory "
                    + "\(startMemory.activeMemory) -> \(endMemory.activeMemory) bytes, "
                    + "peak \(endMemory.peakMemory) bytes")
            return count
        }

        #expect(
            generatedCount >= longGenerationMinimumTokenCount,
            """
            the generation must run past \(longGenerationMinimumTokenCount) tokens \
            without the issue-1662 Metal buffer crash
            """)
    }

    // MARK: Two-round conversation

    /// A second round answers with a fact from the first round.
    /// `ChatSession` keeps the live KV cache across turns and renders only
    /// the new messages of each turn, thus the encoder's generation priming
    /// does not break the in-session cache. Chat mode keeps the answer
    /// inside the token budget.
    @Test func twoRoundConversationRecallsTheFirstRound() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "twoRoundConversationRecallsTheFirstRound")
        else { return }

        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(
                maxTokens: recallGenerationTokenCount, temperature: 0),
            additionalContext: ["thinking": false])
        let firstRound = try await session.respond(
            to: "Remember this number: \(recallNumber). Reply with one short sentence.")
        let secondRound = try await session.respond(
            to: "What number did I ask you to remember? Reply with just the number.")

        print("Round 1: \(firstRound)")
        print("Round 2: \(secondRound)")
        #expect(
            secondRound.contains(recallNumber),
            """
            round 2 must recall \(recallNumber) through the in-session KV cache, \
            got: \(secondRound)
            """)
    }

    // MARK: Cross-round prompt-cache facts (no weights needed)

    /// A round-1 transcript in thinking mode is NOT a prefix of the round-2
    /// prompt, thus cross-round prompt-prefix caching cannot hold in
    /// thinking mode. Two causes, both from the encoder: round 2 re-renders
    /// round 1's generation tail as `<assistant></think>` where round 1
    /// primed `<assistant><think>`, and round 2 drops the round-1 reasoning
    /// (`dropsEarlierReasoning`). This is the honest record the card asks
    /// for: thinking-mode conversations are not prefix-cacheable, and a
    /// cache must instead persist the KV state itself, as `ChatSession`
    /// does.
    @Test func thinkingModeRoundOneTranscriptIsNotARoundTwoPrefix() {
        let encoder = DeepSeekV4ChatEncoder()
        let question = "What is the sea?"
        let reasoning = "The user asks a short question."
        let answer = "The sea is a large body of salt water."

        let roundOnePrompt = encoder.encode(
            messages: [.user(content: question)], thinkingMode: .thinking)
        let roundOneTranscript =
            roundOnePrompt + reasoning + DeepSeekV4ChatEncoder.SpecialToken.thinkEnd
            + answer + DeepSeekV4ChatEncoder.SpecialToken.endOfSentence
        let roundTwoPrompt = encoder.encode(
            messages: [
                .user(content: question),
                .assistant(content: answer, reasoning: reasoning),
                .user(content: "Say it again."),
            ],
            thinkingMode: .thinking)

        #expect(roundOnePrompt.hasSuffix(DeepSeekV4ChatEncoder.SpecialToken.thinkStart))
        #expect(
            !roundTwoPrompt.hasPrefix(roundOneTranscript),
            """
            thinking mode re-renders the tail and drops earlier reasoning, \
            thus prefix reuse cannot hold
            """)
        #expect(
            !roundTwoPrompt.contains(reasoning),
            "round 2 must drop the round-1 chain of thought")
    }

    /// A round-1 transcript in chat mode IS a prefix of the round-2 prompt:
    /// the tail is `<assistant></think>` in both rounds and no reasoning
    /// exists to drop. Thus chat-mode conversations are prefix-cacheable
    /// across rounds.
    @Test func chatModeRoundOneTranscriptIsARoundTwoPrefix() {
        let encoder = DeepSeekV4ChatEncoder()
        let question = "What is the sea?"
        let answer = "The sea is a large body of salt water."

        let roundOnePrompt = encoder.encode(
            messages: [.user(content: question)], thinkingMode: .chat)
        let roundOneTranscript =
            roundOnePrompt + answer + DeepSeekV4ChatEncoder.SpecialToken.endOfSentence
        let roundTwoPrompt = encoder.encode(
            messages: [
                .user(content: question),
                .assistant(content: answer),
                .user(content: "Say it again."),
            ],
            thinkingMode: .chat)

        #expect(roundOnePrompt.hasSuffix(DeepSeekV4ChatEncoder.SpecialToken.thinkEnd))
        #expect(
            roundTwoPrompt.hasPrefix(roundOneTranscript),
            "chat mode keeps the transcript as a prefix, thus prefix reuse can hold")
    }
}
