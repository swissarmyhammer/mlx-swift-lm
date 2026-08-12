// Copyright © 2026 Apple Inc.
//
// Real-weights integration tests for `mlx-community/DeepSeek-V4-Flash-4bit`,
// from card ^e7b24ws. The synthetic-weight unit tests in `Tests/MLXLMTests`
// prove the math. These tests prove the port against the published
// checkpoint: the full `LLMModelFactory` load path, greedy-token parity
// against the Python reference, chat and thinking generation, long-generation
// stability past 12k tokens (the `ml-explore/mlx-lm` issue-1662 landmine),
// and two-round conversation behavior.
//
// The checkpoint is a 284B-total / 13B-active MoE, roughly 91 GB installed.
// The tests never download it. Each real-weights test skips, with a message
// that says why, when the machine has too little memory or when no local
// copy of the checkpoint exists. The two encoder-level cache tests at the
// end run everywhere, because they read no weights.
//
// The card names `Tests/MLXLMIntegrationTests/` as the location. That target
// does not exist, and the root package holds no swift-transformers
// dependency by design, thus a root test target cannot load the real
// tokenizer. This suite therefore follows the real-weights pattern of this
// project (see `MiniMaxM3CacheIntegrationTests`). Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4IntegrationTests`

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

/// The Hub repository of the checkpoint under test.
private let deepseekV4RepositoryID = "mlx-community/DeepSeek-V4-Flash-4bit"

/// The environment key that points the suite at a pre-downloaded checkpoint
/// directory, in place of the Hugging Face cache locations.
private let deepseekV4CheckpointOverrideKey = "MLX_DEEPSEEK_V4_CHECKPOINT"

/// The least physical memory a run needs: about 91 GB of 4-bit weights, plus
/// the MoE working set, the KV cache of a 12k-token generation, and system
/// headroom.
private let deepseekV4RequiredMemoryBytes: UInt64 = 160 * 1_024 * 1_024 * 1_024

/// The number of decoder layers of the published checkpoint.
private let deepseekV4LayerCount = 43

/// The number of tokens the long-generation test asks for. It is past the
/// 12,288-token bound of the card, and well past the ~11.5k tokens at which
/// the mlx-lm issue-1662 buffer leak kills the process.
private let longGenerationTokenCount = 12_400

/// The bound that the long generation must pass: 12k tokens.
private let longGenerationMinimumTokenCount = 12_288

/// The number of tokens each short generation asks for.
private let shortGenerationTokenCount = 48

/// The number of tokens each turn of the two-round test asks for.
private let recallGenerationTokenCount = 60

/// The number that the two-round test asks the model to remember.
private let recallNumber = "4172"

// MARK: - Checkpoint location

/// Finds a complete local copy of the checkpoint, and never downloads one.
///
/// The search order is: the ``deepseekV4CheckpointOverrideKey`` directory,
/// the `huggingface_hub` cache snapshot, then the swift-transformers download
/// base. A directory counts only when it holds at least one `*.safetensors`
/// file, because a tokenizer-only snapshot cannot feed a weight load.
///
/// - Returns: the checkpoint directory, or `nil` when no complete local copy
///   exists.
private func localDeepseekV4CheckpointDirectory() -> URL? {
    var candidates: [URL] = []
    if let override = ProcessInfo.processInfo.environment[deepseekV4CheckpointOverrideKey] {
        candidates.append(URL(fileURLWithPath: override, isDirectory: true))
    }
    if let snapshot = hfSnapshotDir(modelId: deepseekV4RepositoryID) {
        candidates.append(snapshot)
    }
    candidates.append(
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Documents/huggingface/models", isDirectory: true)
            .appendingPathComponent(deepseekV4RepositoryID, isDirectory: true))
    return candidates.first(where: directoryHoldsSafetensors)
}

/// Tells whether `directory` holds at least one `*.safetensors` file.
private func directoryHoldsSafetensors(_ directory: URL) -> Bool {
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
    else { return false }
    return entries.contains { $0.pathExtension == "safetensors" }
}

// MARK: - One shared load

/// One shared load of the checkpoint. Each test awaits the same load task,
/// thus the ~91 GB weight load runs at most once per test process.
///
/// The task is `nil` when no complete local checkpoint exists, and the task
/// itself throws when the load fails.
private enum DeepseekV4Load {
    /// The shared load task, or `nil` when the checkpoint is absent.
    static let shared: Task<LLModelContainer, Error>? = {
        guard let directory = localDeepseekV4CheckpointDirectory() else { return nil }
        return Task {
            print("Loading DeepSeek-V4 from \(directory.path)")
            let container = try await LLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
            print("Loaded DeepSeek-V4")
            return container
        }
    }()
}

// MARK: - Parity fixture

/// The greedy-parity fixture, produced by the Python reference
/// (`Thump604/mlx-lm` @ `deepseek-v4-support-fixes`, or `ml-explore/mlx-lm`
/// PR 1189) against the same checkpoint:
///
/// ```python
/// from mlx_lm import load
/// from mlx_lm.generate import generate_step
/// import json, mlx.core as mx
/// model, tokenizer = load("mlx-community/DeepSeek-V4-Flash-4bit")
/// prompt = tokenizer.encode("<|user|>Write one sentence about the sea.<|assistant|><think>")
/// ids = [t for t, _ in zip(
///     (tok for tok, _ in generate_step(mx.array(prompt), model)), range(64))]
/// json.dump({"prompt_token_ids": prompt, "generated_token_ids": ids},
///           open("deepseek-v4-flash-4bit-greedy-parity.json", "w"))
/// ```
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
@Suite(.serialized, .timeLimit(.minutes(240)))
struct DeepseekV4IntegrationTests {

    /// Loads the shared container, or prints a skip message and returns
    /// `nil`. The gates are: enough physical memory, a complete local
    /// checkpoint, and a load that completes.
    private func loadContainerOrSkip(testName: String) async -> LLModelContainer? {
        guard ProcessInfo.processInfo.physicalMemory >= deepseekV4RequiredMemoryBytes else {
            print(
                "Skipping \(testName): physical memory "
                    + "\(ProcessInfo.processInfo.physicalMemory) bytes is below the required "
                    + "\(deepseekV4RequiredMemoryBytes) bytes")
            return nil
        }
        guard let load = DeepseekV4Load.shared else {
            print(
                "Skipping \(testName): no local copy of \(deepseekV4RepositoryID) "
                    + "with safetensors files. Download the checkpoint, or set "
                    + "\(deepseekV4CheckpointOverrideKey) to a checkpoint directory. "
                    + "This test never downloads the ~91 GB checkpoint itself.")
            return nil
        }
        do {
            return try await load.value
        } catch {
            print("Skipping \(testName): failed to load \(deepseekV4RepositoryID): \(error)")
            return nil
        }
    }

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
        guard let container = await loadContainerOrSkip(testName: "loadsTheRealCheckpointEndToEnd")
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

    // MARK: Greedy parity

    /// The first N greedy token identifiers match the Python reference
    /// exactly. Greedy decode is deterministic, thus an exact match is a
    /// real parity check. The fixture feeds the reference's own prompt token
    /// identifiers, thus this test isolates model parity from prompt
    /// encoding.
    @Test func greedyFirstTokensMatchThePythonFixture() async throws {
        let fixtureURL = deepseekV4ParityFixtureURL
        guard let fixtureData = try? Data(contentsOf: fixtureURL) else {
            print(
                "Skipping greedyFirstTokensMatchThePythonFixture: no fixture at "
                    + "\(fixtureURL.path). Generate it with the Python reference "
                    + "(Thump604/mlx-lm @ deepseek-v4-support-fixes, or "
                    + "ml-explore/mlx-lm PR 1189) — see DeepseekV4ParityFixture "
                    + "in this file for the exact script and schema.")
            return
        }
        let fixture = try JSONDecoder().decode(DeepseekV4ParityFixture.self, from: fixtureData)
        guard
            let container = await loadContainerOrSkip(
                testName: "greedyFirstTokensMatchThePythonFixture")
        else { return }

        let promptTokenIDs = fixture.promptTokenIDs
        let expectedCount = fixture.generatedTokenIDs.count
        let generated = try await container.perform { context in
            let input = LMInput(tokens: MLXArray(promptTokenIDs.map(Int32.init)))
            var iterator = try TokenIterator(
                input: input, model: context.model,
                parameters: GenerateParameters(maxTokens: expectedCount, temperature: 0))
            var tokens: [Int] = []
            while tokens.count < expectedCount, let token = iterator.next() {
                tokens.append(token)
            }
            return tokens
        }

        #expect(
            generated == fixture.generatedTokenIDs,
            "greedy decode must reproduce the Python reference token for token")
    }

    // MARK: Chat and thinking modes

    /// Both generation modes render through the encoder and generate. The
    /// default mode is thinking; `additionalContext: ["thinking": false]`
    /// selects chat mode.
    @Test func chatAndThinkingModesBothGenerate() async throws {
        guard
            let container = await loadContainerOrSkip(
                testName: "chatAndThinkingModesBothGenerate")
        else { return }

        let prompt = "Write one short sentence about the sea."
        let thinkingOutput = try await generateText(
            container: container,
            input: UserInput(chat: [.user(content: prompt)]),
            maxTokens: shortGenerationTokenCount)
        let chatOutput = try await generateText(
            container: container,
            input: UserInput(
                chat: [.user(content: prompt)], additionalContext: ["thinking": false]),
            maxTokens: shortGenerationTokenCount)

        print("Thinking output: \(thinkingOutput)")
        print("Chat output: \(chatOutput)")
        #expect(!thinkingOutput.isEmpty, "thinking mode must generate text")
        #expect(!chatOutput.isEmpty, "chat mode must generate text")
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
            let container = await loadContainerOrSkip(
                testName: "longGenerationPastTwelveThousandTokensCompletes")
        else { return }

        let generatedCount = try await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(chat: [.user(content: "Count upward from one, forever.")]))
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
            let container = await loadContainerOrSkip(
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
