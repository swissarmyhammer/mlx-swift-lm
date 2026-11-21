// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of the Inkling 2 Mini DSpark drafter (`mtp.safetensors`): greedy MTP
// emits the tokens of plain greedy decoding, a verify pass computes the logits of decode steps
// bit for bit, the cost of the parts of a round, the decode throughput of plain and MTP decoding
// at a short and a 4k-token context with the acceptance rate, and prompt cache reuse across the
// turns of a `ChatSession` with MTP on.
//
// The target is `mlx-community/Inkling-2-Mini-4bit`, which the port loads. The drafter loads
// alone from `mlx-community/Inkling-2-Mini-oQ4e-mtp`, the checkpoint that ships the head; its
// target weights use the oMLX layout, which the port does not load. Both download through the
// Hugging Face downloader, or come from the cache, thus no test skips. The two checkpoints are
// 31.8 GiB.
//
// `swift test` is BLIND to this file. Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2MTPIntegrationTests`

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

// MARK: - Pinned drafter and shape

/// The revision of 2026-11-21 of the oQ4e Inkling 2 Mini checkpoint that ships the DSpark head.
private let inkling2MiniOQ4eMTPRevision = "3eea042751cd296b7266628a8342c1749242e1fe"

/// The DSpark head of Inkling 2 Mini, at ``inkling2MiniOQ4eMTPRevision``.
private let inkling2MiniDrafter = ModelConfiguration(
    id: MTPDrafterRegistry.inkling2_mini_oq4e_mtp.name, revision: inkling2MiniOQ4eMTPRevision)

/// The block of the parity and chat tests: the shared MTP default.
private let defaultBlockSize = SpeculativeDecodingConfig.defaultMTPBlockSize

/// The blocks the throughput test measures, up to the 8 the head drafts.
private let measuredBlockSizes = [3, defaultBlockSize, 5, 6, 8]

/// The verify widths whose logits are compared with decode steps.
private let verifyWidths = [2, defaultBlockSize, 8]

/// The greedy tokens each parity prompt decodes.
private let parityTokens = 160

/// The greedy tokens each throughput run decodes at the short and at the long context.
private let shortContextTokens = 256
private let longContextTokens = 128

/// The timed repetitions of each decode path; the result is their median.
private let repetitions = 3

/// The timed samples of each part of a round.
private let profileSamples = 9

/// The tokens of a warm-up run, which compiles kernels and realizes lazy tensors.
private let warmUpTokens = 16

/// The tokens the chat turns may generate.
private let chatTokens = 384

/// The milliseconds of one second.
private let millisecondsPerSecond = 1_000.0

/// The largest gap of the two best logits at which MTP may write the second best token past the
/// sliding window, as the fork judges MTP against plain decoding.
///
/// Past the window plain decoding reads each sliding layer's keys from a wrapped ring, and a
/// verify pass reads them in temporal order, so the two sum the attention in another order; a
/// bfloat16 logit near 30 has a step of 0.125, so the sums can round a near tie apart.
private let nearTieLogitGap: Float = 0.5

/// The best two candidates whose logits are compared at a divergence.
private let topCandidateCount = 2

// MARK: - Loading

/// The target and the drafter, loaded once per test process.
///
/// The suite is serialized, so one test touches the models at a time: the invariant that makes
/// the box safe to share.
private final class Inkling2MTPPair: @unchecked Sendable {
    let target: ModelContext
    let drafter: Inkling2DraftModel
    let drafterContext: MTPDrafterContext

    init(target: ModelContext, drafterContext: MTPDrafterContext) throws {
        self.target = target
        self.drafterContext = drafterContext
        self.drafter = try #require(drafterContext.model as? Inkling2DraftModel)
    }

    /// The tokens of a chat prompt for `question`, at low thinking effort.
    func input(_ question: String) async throws -> LMInput {
        try await target.processor.prepare(
            input: UserInput(
                chat: [.user(question)], additionalContext: ["reasoning_effort": "low"]))
    }
}

/// Loads the pair on the first call and shares it.
private actor Inkling2MTPPairLoader {
    static let shared = Inkling2MTPPairLoader()

    private var loading: Task<Inkling2MTPPair, Error>?

    func pair() async throws -> Inkling2MTPPair {
        if let loading {
            return try await loading.value
        }
        let load = Task {
            await Inkling2DraftRegistration.register()
            let target = try await VLMModelFactory.shared.load(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: inkling2Mini4Bit)
            let drafter = try await MTPDrafterModelFactory.shared.load(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: inkling2MiniDrafter)
            return try Inkling2MTPPair(target: target, drafterContext: drafter)
        }
        loading = load
        return try await load.value
    }
}

// MARK: - Decoding

/// Every token `iterator` emits.
private func drain<Iterator: TokenIteratorProtocol>(_ iterator: inout Iterator) -> [Int] {
    var tokens: [Int] = []
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}

/// One timed greedy decode.
private struct DecodeRun {
    /// The emitted tokens.
    let tokens: [Int]
    /// The milliseconds of the drain, after the prefill.
    let decodeMilliseconds: Double
    /// The drafts the drafter proposed; zero for plain decoding.
    let proposed: Int
    /// The drafts the target accepted; zero for plain decoding.
    let accepted: Int
    /// The rounds of the MTP decode; zero for plain decoding.
    let rounds: Int
    /// Why the MTP stream stopped speculating, or `nil`.
    let passthroughReason: String?

    /// The decode rate in tokens per second.
    var tokensPerSecond: Double {
        Double(tokens.count) / (decodeMilliseconds / millisecondsPerSecond)
    }
}

/// The value of `body` and its wall milliseconds, with a GPU sync before and after it.
private func wallMilliseconds<Result>(_ body: () throws -> Result) rethrows -> (Result, Double) {
    Stream.gpu.synchronize()
    let start = Date()
    let value = try body()
    Stream.gpu.synchronize()
    return (value, Date().timeIntervalSince(start) * millisecondsPerSecond)
}

/// A plain greedy decode of `tokens` tokens.
private func plainRun(_ pair: Inkling2MTPPair, input: LMInput, tokens: Int) throws -> DecodeRun {
    var iterator = try TokenIterator(
        input: input, model: pair.target.model,
        parameters: GenerateParameters(maxTokens: tokens, temperature: 0))
    let (emitted, milliseconds) = wallMilliseconds { drain(&iterator) }
    return DecodeRun(
        tokens: emitted, decodeMilliseconds: milliseconds, proposed: 0, accepted: 0, rounds: 0,
        passthroughReason: nil)
}

/// A greedy MTP decode of `tokens` tokens in rounds of `blockSize` tokens.
private func mtpRun(_ pair: Inkling2MTPPair, input: LMInput, tokens: Int, blockSize: Int) throws
    -> DecodeRun
{
    var iterator = try MTPSpeculativeTokenIterator(
        input: input, mainModel: pair.target.model, drafter: pair.drafter,
        parameters: GenerateParameters(maxTokens: tokens, temperature: 0), blockSize: blockSize)
    let (emitted, milliseconds) = wallMilliseconds { drain(&iterator) }
    return DecodeRun(
        tokens: emitted, decodeMilliseconds: milliseconds, proposed: iterator.proposedCount,
        accepted: iterator.acceptedCount,
        rounds: iterator.speculativeDecodingTelemetry?.roundCount ?? 0,
        passthroughReason: iterator.passthroughReason)
}

/// The median of `values`.
private func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted[sorted.count / 2]
}

/// Where MTP first parts from plain decoding, and the two best logits of an uncached forward
/// pass over the tokens both share.
private struct Divergence: CustomStringConvertible {
    /// The index of the first token where the two part.
    let step: Int
    /// The token MTP wrote there.
    let mtpToken: Int
    /// The two best tokens of the forward pass, best first.
    let topTokens: [Int]
    /// The logits of ``topTokens``.
    let topLogits: [Float]

    /// Whether MTP wrote one of the two best tokens, within ``nearTieLogitGap`` of each other.
    var isNearTie: Bool {
        topTokens.contains(mtpToken) && topLogits[0] - topLogits[1] <= nearTieLogitGap
    }

    var description: String {
        "MTP parts from plain at token \(step): MTP \(mtpToken), best \(topTokens) \(topLogits)"
    }

    /// Where `mtp` first parts from `plain` after `input`, or `nil` when they agree.
    static func find(
        _ pair: Inkling2MTPPair, input: LMInput, mtp: [Int], plain: [Int]
    ) -> Divergence? {
        guard let step = zip(mtp, plain).enumerated().first(where: { $1.0 != $1.1 })?.offset
        else { return nil }
        let context = input.text.tokens.reshaped(-1).asArray(Int.self) + plain.prefix(step)
        let logits = pair.target.model(
            MLXArray(context.map(Int32.init))[.newAxis, 0...], cache: nil
        )[0, -1].asType(.float32)
        let order = argSort(-logits)[..<topCandidateCount]
        return Divergence(
            step: step, mtpToken: mtp[step], topTokens: order.asArray(Int.self),
            topLogits: logits[order].asArray(Float.self))
    }
}

/// The text and the completion info of one chat turn.
private struct MTPTurn {
    var text = ""
    var info: GenerateCompletionInfo?

    /// Drains `stream` into a turn.
    static func collect(_ stream: AsyncThrowingStream<Generation, Error>) async throws -> MTPTurn {
        var turn = MTPTurn()
        for try await event in stream {
            switch event {
            case .chunk(let chunk): turn.text += chunk
            case .info(let info): turn.info = info
            default: break
            }
        }
        return turn
    }
}

/// One formatted number.
private func formatted(_ value: Double, _ format: String = "%.1f") -> String {
    String(format: format, value)
}

// MARK: - Tests

@Suite(.serialized, .timeLimit(.minutes(60)))
struct Inkling2MTPIntegrationTests {

    /// Three prompts of different kinds: code, prose and arithmetic.
    private static let parityPrompts = [
        "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number.",
        "Describe a lighthouse at night in four sentences.",
        "What is 17 times 23? Show the steps.",
    ]

    /// The prompt of the short-context throughput runs.
    private static let storyPrompt =
        "Write a long, detailed story about a lighthouse keeper and a storm."

    /// Greedy MTP emits the token ids of plain greedy decoding on three prompts.
    @Test
    func greedyMTPMatchesPlainGreedy() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        for prompt in Self.parityPrompts {
            let input = try await pair.input(prompt)
            let plain = try plainRun(pair, input: input, tokens: parityTokens)
            let mtp = try mtpRun(
                pair, input: input, tokens: parityTokens, blockSize: defaultBlockSize)
            let divergence = Divergence.find(
                pair, input: input, mtp: mtp.tokens, plain: plain.tokens)
            Inkling2ResultLog.report(
                "INKLING2-MTP",
                "parity: \(mtp.accepted)/\(mtp.proposed) drafts accepted over \(mtp.rounds) "
                    + "rounds, \(divergence.map(String.init) ?? "equal over \(plain.tokens.count)")"
            )
            #expect(mtp.passthroughReason == nil, "\(mtp.passthroughReason ?? "")")
            #expect(mtp.accepted > 0)
            #expect(mtp.tokens == plain.tokens, "\(divergence.map(String.init) ?? "")")
        }
    }

    /// A verify pass of several tokens computes the logits of each token as a decode step at its
    /// position does, bit for bit, below the sliding window.
    @Test(arguments: verifyWidths)
    func verifyPassMatchesDecodeSteps(width: Int) async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        let model = pair.target.model
        let prompt = try await pair.input(Self.parityPrompts[1]).text.tokens.reshaped(1, -1)
        let tokens = MLXArray((0 ..< width).map { Int32(1000 + 37 * $0) }).reshaped(1, -1)

        let stepped = try model.newCache(parameters: nil)
        eval(model(prompt, cache: stepped))
        let steps = (0 ..< width).map { index in
            let step = model(tokens[0..., index ..< (index + 1)], cache: stepped)
            eval(step)
            return step
        }
        let verified = try model.newCache(parameters: nil)
        eval(model(prompt, cache: verified))
        let verify = model(tokens, cache: verified)

        let difference = abs(verify - concatenated(steps, axis: 1)).max().item(Float.self)
        Inkling2ResultLog.report(
            "INKLING2-MTP",
            "verify L=\(width) vs decode steps: max |logit difference| \(difference)")
        #expect(difference == 0)
    }

    /// The cost of the parts of a round at a short context: one decode step, a verify pass of
    /// each width, which a rewind then takes back, and a drafted block, each the median of
    /// several timed runs.
    @Test
    func roundCostProfile() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        let model = pair.target.model
        let prompt = try await pair.input(Self.storyPrompt).text.tokens.reshaped(1, -1)
        let cache = try model.newCache(parameters: nil)
        eval(model(prompt, cache: cache))
        let token = prompt[0..., (-1)...]

        /// The median milliseconds of `body`, which returns what it evaluates; `after` puts the
        /// cache back.
        func profile(_ label: String, _ body: () -> [MLXArray], after: () -> Void = {}) {
            let samples = (0 ..< profileSamples).map { _ in
                let milliseconds = wallMilliseconds { eval(body()) }.1
                after()
                return milliseconds
            }
            Inkling2ResultLog.report(
                "INKLING2-MTP", "profile \(label): \(formatted(median(samples), "%.2f")) ms")
            #expect(median(samples) > 0)
        }

        profile("decode L=1", { [model(token, cache: cache)] })
        for width in verifyWidths {
            let tokens = repeated(token, count: width, axis: 1)
            profile(
                "verify L=\(width)", { [model(tokens, cache: cache)] },
                after: { trimPromptCache(cache, numTokens: width) })
        }
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        emit[mtpTargetLayerIdsKey] = pair.drafter.targetLayerIds
        let output = model(LMInput.Text(tokens: prompt), cache: nil, state: emit)
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        var state = pair.drafter.makeState(parameters: nil)
        pair.drafter.prepareDrafterState(
            target: model, promptTokens: prompt, targetHidden: hidden,
            firstBonus: token.reshaped(-1), positionDeltas: nil, state: &state,
            sampler: ArgMaxSampler())
        eval(state.cache.flatMap(\.state))
        for blockSize in [defaultBlockSize, 8] {
            profile("draft block \(blockSize)") {
                [
                    pair.drafter.draftBlock(
                        target: model, lastToken: token.reshaped(-1),
                        lastHidden: MLXArray.zeros([1, 1, 1]), sharedKV: [:],
                        positionDeltas: nil, queryOffset: 0, blockSize: blockSize,
                        state: &state, sampler: ArgMaxSampler())
                ]
            }
        }
    }

    /// Plain and MTP decode throughput at a short prompt and at a prompt of about 4k tokens, at
    /// every measured block, each the median of three runs after a warm-up, with the
    /// acceptance. Below the sliding window MTP emits the tokens of plain decoding; past it each
    /// divergence is a bfloat16 near tie (see ``nearTieLogitGap``).
    @Test
    func throughputAtShortAndLongContext() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        let contexts = [
            ("short", try await pair.input(Self.storyPrompt), shortContextTokens, true),
            (
                "long", try await pair.input(Inkling2CheckpointIntegrationTests.longPrompt),
                longContextTokens, false
            ),
        ]
        for (label, input, tokens, exact) in contexts {
            _ = try plainRun(pair, input: input, tokens: warmUpTokens)
            for blockSize in measuredBlockSizes {
                _ = try mtpRun(pair, input: input, tokens: warmUpTokens, blockSize: blockSize)
            }
            Memory.clearCache()

            var plainRuns: [DecodeRun] = []
            var mtpRuns = [Int: [DecodeRun]]()
            for _ in 0 ..< repetitions {
                plainRuns.append(try plainRun(pair, input: input, tokens: tokens))
                for blockSize in measuredBlockSizes {
                    mtpRuns[blockSize, default: []].append(
                        try mtpRun(pair, input: input, tokens: tokens, blockSize: blockSize))
                }
            }
            let plainRate = median(plainRuns.map(\.tokensPerSecond))
            Inkling2ResultLog.report(
                "INKLING2-MTP",
                "\(label): prompt \(input.text.tokens.size) tok, plain \(formatted(plainRate)) tok/s"
            )
            for blockSize in measuredBlockSizes {
                let runs = mtpRuns[blockSize] ?? []
                let rate = median(runs.map(\.tokensPerSecond))
                let accepted = runs.map(\.accepted).reduce(0, +)
                let proposed = runs.map(\.proposed).reduce(0, +)
                let rounds = runs.map(\.rounds).reduce(0, +)
                Inkling2ResultLog.report(
                    "INKLING2-MTP",
                    "\(label) block \(blockSize): MTP \(formatted(rate)) tok/s "
                        + "(\(formatted(rate / plainRate, "%.2f"))x), acceptance \(accepted)/"
                        + "\(proposed) = \(formatted(Double(accepted) / Double(proposed), "%.3f")), "
                        + "\(formatted(Double(accepted + rounds) / Double(rounds), "%.2f")) "
                        + "tokens per round")
                for run in runs {
                    #expect(run.passthroughReason == nil, "\(run.passthroughReason ?? "")")
                    let divergence = Divergence.find(
                        pair, input: input, mtp: run.tokens, plain: plainRuns[0].tokens)
                    let description = divergence.map(String.init) ?? ""
                    if exact {
                        #expect(divergence == nil, "\(label) block \(blockSize): \(description)")
                    } else {
                        #expect(divergence?.isNearTie ?? true, "\(label): \(description)")
                    }
                }
            }
        }
    }

    /// Each follow-up turn of a `ChatSession` with MTP on serves part of its prompt from the
    /// cache of the turns before it, keeps speculating, and answers as a plain session does.
    @Test
    func chatSessionReusesThePromptCacheWithMTP() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        let questions = [
            "My name is Ada and my favourite number is 8214. Say hello.",
            "What is my favourite number?",
            "And what is my name?",
        ]
        let instructions = "You are a concise assistant. Answer in one or two sentences."
        let parameters = GenerateParameters(maxTokens: chatTokens, temperature: 0)
        let speculative = try SpeculativeDecodingConfig(
            mtpDrafter: MTPDrafterContainer(context: pair.drafterContext),
            blockSize: defaultBlockSize)
        let mtpSession = ChatSession(
            pair.target, instructions: instructions, speculativeDecoding: speculative,
            generateParameters: parameters)
        let plainSession = ChatSession(
            pair.target, instructions: instructions, generateParameters: parameters)

        for (index, question) in questions.enumerated() {
            let mtp = try await MTPTurn.collect(mtpSession.streamDetails(to: question))
            let plain = try await MTPTurn.collect(plainSession.streamDetails(to: question))
            let info = try #require(mtp.info)
            let plainInfo = try #require(plain.info)
            Inkling2ResultLog.report(
                "INKLING2-MTP",
                "chat turn \(index): prompt \(info.promptTokenCount) tok (cached "
                    + "\(info.cachedPromptTokenCount), plain cached "
                    + "\(plainInfo.cachedPromptTokenCount)), \(info.acceptedDraftTokens ?? 0)/"
                    + "\(info.proposedDraftTokens ?? 0) drafts, decode "
                    + "\(formatted(info.tokensPerSecond)) tok/s (plain "
                    + "\(formatted(plainInfo.tokensPerSecond))) -> \(mtp.text)")
            #expect(info.passthroughReason == nil, "\(info.passthroughReason ?? "")")
            #expect((info.acceptedDraftTokens ?? 0) > 0, "turn \(index) accepted no drafts")
            if index > 0 {
                #expect(info.cachedPromptTokenCount > 0, "turn \(index) reused no prompt tokens")
            }
            #expect(mtp.text == plain.text, "turn \(index)")
        }
    }
}
