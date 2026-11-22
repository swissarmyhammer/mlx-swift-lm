// Copyright © 2026 Apple Inc.
//
// Real-checkpoint tests of the Inkling 2 Mini DSpark drafter (`mtp.safetensors`): greedy MTP
// emits the tokens of plain greedy decoding, a verify pass computes the logits of decode steps
// bit for bit, the cost of the parts of a round, the decode throughput of plain and MTP decoding
// at a short and a 4k-token context with the acceptance rate, prompt cache reuse across the
// turns of a `ChatSession` with MTP on, and the oQ4e target beside the 4bit target.
//
// The target is `mlx-community/Inkling-2-Mini-4bit`, the checkpoint the Inkling 2 suites share.
// The drafter loads alone from `mlx-community/Inkling-2-Mini-oQ4e-mtp`, the checkpoint that ships
// the head; its target weights, in the oMLX layout that `Inkling2.prepareCheckpoint` maps, load
// as a second target in one test. Both download through the Hugging Face downloader, or come from
// the cache, thus no test skips. The two checkpoints are 31.8 GiB.
//
// `swift test` is BLIND to this file. Build for Release, or the numbers measure the debug build:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:IntegrationTestingTests/Inkling2MTPIntegrationTests`

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing
import Tokenizers

// MARK: - Pinned drafter and shape

/// The revision of 2026-11-21 of the oQ4e Inkling 2 Mini checkpoint that ships the DSpark head.
private let inkling2MiniOQ4eMTPRevision = "3eea042751cd296b7266628a8342c1749242e1fe"

/// The DSpark head of Inkling 2 Mini, at ``inkling2MiniOQ4eMTPRevision``.
private let inkling2MiniDrafter = ModelConfiguration(
    id: MTPDrafterRegistry.inkling2_mini_oq4e_mtp.name, revision: inkling2MiniOQ4eMTPRevision)

/// The oQ4e checkpoint as a target: its weights in the oMLX layout, which
/// `Inkling2.prepareCheckpoint` maps onto the port's modules.
private let inkling2MiniOQ4eTarget = ModelConfiguration(
    id: "mlx-community/Inkling-2-Mini-oQ4e-mtp", revision: inkling2MiniOQ4eMTPRevision)

/// The block of the parity and chat tests: the shared MTP default.
private let defaultBlockSize = SpeculativeDecodingConfig.defaultMTPBlockSize

/// The block the head drafts, the `block_size` of its `mtp_config`.
private let headBlockSize = 8

/// The smallest block the throughput test measures.
private let smallestMeasuredBlockSize = 3

/// The blocks the throughput test measures, up to the block the head drafts.
private let measuredBlockSizes = Array(smallestMeasuredBlockSize ... headBlockSize)

/// The smallest verify: the bonus token and one draft.
private let smallestVerifyWidth = 2

/// The verify widths whose logits are compared with decode steps.
private let verifyWidths = [smallestVerifyWidth, defaultBlockSize, headBlockSize]

/// The first token of the verify passes of the logit comparison, and the step between their
/// tokens: ordinary vocabulary tokens, none of them special.
private let verifyTokenBase = 1_000
private let verifyTokenStride = 37

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

/// The largest gap of the two best logits at which MTP may write the second best token past the
/// sliding window, as the fork judges MTP against plain decoding.
///
/// Past the window plain decoding reads each sliding layer's keys from a wrapped ring, and a
/// verify pass reads them in temporal order, so the two sum the attention in another order; a
/// bfloat16 logit near 30 has a step of 0.125, so the sums can round a near tie apart. Summing
/// the verify in ring order made it exact but cost the long-context verify about 4% and MTP
/// about 5% on the M5 (^2r778tc), thus the port keeps the near tie.
private let nearTieLogitGap: Float = 0.5

// MARK: - Loading

/// The containers of the target and the drafter, loaded once per test process.
private struct Inkling2MTPPair: Sendable {
    let target: ModelContainer
    let drafter: MTPDrafterContainer

    /// Runs `body` on the target's context and the drafter: inside the drafter's `perform`, then
    /// inside the target's, so neither model leaves its container's isolation.
    ///
    /// - Parameter body: The work on the target's context and the drafter.
    /// - Returns: The result of `body`.
    /// - Throws: The error of `body`, or when the drafter is not the Inkling 2 head.
    func perform<Value: Sendable>(
        _ body: @Sendable (ModelContext, Inkling2DraftModel) async throws -> Value
    ) async throws -> Value {
        let target = target
        return try await drafter.perform { drafterContext in
            try await target.perform(nonSendable: drafterContext.model) { context, model in
                try await body(context, try #require(model as? Inkling2DraftModel))
            }
        }
    }
}

/// The tokens of a chat prompt for `question`, at low thinking effort.
private func chatInput(_ context: ModelContext, _ question: String) async throws -> LMInput {
    try await context.processor.prepare(
        input: UserInput(chat: [.user(question)], additionalContext: ["reasoning_effort": "low"]))
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
            let target = try await Inkling2MiniCheckpoint.shared.container()
            let drafter = try await MTPDrafterModelFactory.shared.load(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: inkling2MiniDrafter)
            return Inkling2MTPPair(target: target, drafter: MTPDrafterContainer(context: drafter))
        }
        loading = load
        return try await load.value
    }
}

/// The oQ4e target, loaded on the first call and shared.
private let inkling2MiniOQ4eCheckpoint = Inkling2MiniCheckpoint(
    configuration: inkling2MiniOQ4eTarget)

/// The bytes of the parameters of `model`, in GiB.
private func parameterGiB(_ model: any LanguageModel) -> Double {
    let bytes = model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    return Double(bytes) / Double(1 << 30)
}

// MARK: - Decoding

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
        GPUWallTime.ratePerSecond(tokens.count, milliseconds: decodeMilliseconds)
    }
}

/// A plain greedy decode of `tokens` tokens.
private func plainRun(_ target: any LanguageModel, input: LMInput, tokens: Int) throws
    -> DecodeRun
{
    var iterator = try TokenIterator(
        input: input, model: target,
        parameters: GenerateParameters(maxTokens: tokens, temperature: 0))
    let (emitted, milliseconds) = GPUWallTime.measure { drain(&iterator) }
    return DecodeRun(
        tokens: emitted, decodeMilliseconds: milliseconds, proposed: 0, accepted: 0, rounds: 0,
        passthroughReason: nil)
}

/// A greedy MTP decode of `tokens` tokens in rounds of `blockSize` tokens.
private func mtpRun(
    _ target: any LanguageModel, drafter: Inkling2DraftModel, input: LMInput, tokens: Int,
    blockSize: Int
) throws -> DecodeRun {
    var iterator = try MTPSpeculativeTokenIterator(
        input: input, mainModel: target, drafter: drafter,
        parameters: GenerateParameters(maxTokens: tokens, temperature: 0), blockSize: blockSize)
    let (emitted, milliseconds) = GPUWallTime.measure { drain(&iterator) }
    return DecodeRun(
        tokens: emitted, decodeMilliseconds: milliseconds, proposed: iterator.proposedCount,
        accepted: iterator.acceptedCount,
        rounds: iterator.speculativeDecodingTelemetry?.roundCount ?? 0,
        passthroughReason: iterator.passthroughReason)
}

/// Where MTP first parts from plain decoding after `input`, or `nil` when the two agree.
private func findDivergence(
    _ target: any LanguageModel, input: LMInput, mtp: [Int], plain: [Int]
) async -> MTPDivergence? {
    await MTPDivergence.find(
        mtp: mtp, plain: plain, context: input.text.tokens.reshaped(-1).asArray(Int.self)
    ) { tokens in
        MTPDivergence.topCandidates(
            of: target(MLXArray(tokens.map(Int32.init))[.newAxis, 0...], cache: nil)[0, -1])
    }
}

/// Expects `run` to speculate throughout and to emit `plain`, the tokens of plain greedy
/// decoding after `input`, or to part from it only at a near tie within `nearTieGap`.
///
/// - Parameters:
///   - run: the MTP decode.
///   - target: the target, which finds where the two part.
///   - input: the prompt of both decodes.
///   - plain: the tokens of plain greedy decoding.
///   - label: the label of the expectations.
///   - nearTieGap: the largest logit gap of a near tie MTP may round apart, or `nil` to expect
///     the tokens of plain decoding exactly.
/// - Returns: where MTP first parts from plain decoding, or `nil` when the two agree.
@discardableResult
private func expectMatchesPlain(
    _ run: DecodeRun, _ target: any LanguageModel, input: LMInput, plain: [Int], label: String,
    nearTieGap: Float? = nil
) async -> MTPDivergence? {
    #expect(run.passthroughReason == nil, "\(label): \(run.passthroughReason ?? "")")
    let divergence = await findDivergence(target, input: input, mtp: run.tokens, plain: plain)
    let description = "\(label): \(divergence.map(String.init) ?? "")"
    if let nearTieGap {
        #expect(divergence?.isNearTie(within: nearTieGap) ?? true, "\(description)")
    } else {
        #expect(divergence == nil, "\(description)")
    }
    return divergence
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
        try await pair.perform { context, drafter in
            for prompt in Self.parityPrompts {
                let input = try await chatInput(context, prompt)
                let plain = try plainRun(context.model, input: input, tokens: parityTokens)
                let mtp = try mtpRun(
                    context.model, drafter: drafter, input: input, tokens: parityTokens,
                    blockSize: defaultBlockSize)
                let divergence = await expectMatchesPlain(
                    mtp, context.model, input: input, plain: plain.tokens, label: prompt)
                Inkling2ResultLog.report(
                    "INKLING2-MTP",
                    "parity: \(mtp.accepted)/\(mtp.proposed) drafts accepted over \(mtp.rounds) "
                        + "rounds, "
                        + "\(divergence.map(String.init) ?? "equal over \(plain.tokens.count)")")
                #expect(mtp.accepted > 0)
            }
        }
    }

    /// A verify pass of several tokens computes the logits of each token as a decode step at its
    /// position does, bit for bit, below the sliding window.
    @Test(arguments: verifyWidths)
    func verifyPassMatchesDecodeSteps(width: Int) async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        try await pair.perform { context, _ in
            let model = context.model
            let prompt = try await chatInput(context, Self.parityPrompts[1]).text.tokens
                .reshaped(1, -1)
            let tokens = MLXArray(
                (0 ..< width).map { Int32(verifyTokenBase + verifyTokenStride * $0) }
            ).reshaped(1, -1)

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
                "verify L=\(width) after \(prompt.dim(1)) tokens vs decode steps: max |logit "
                    + "difference| \(difference)")
            #expect(difference == 0)
        }
    }

    /// The cost of the parts of a round at a short context: one decode step, a verify pass of
    /// each width, which a rewind then takes back, and a drafted block, each the median of
    /// several timed runs.
    @Test
    func roundCostProfile() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        try await pair.perform { context, drafter in
            let model = context.model
            let prompt = try await chatInput(context, Self.storyPrompt).text.tokens.reshaped(1, -1)
            let cache = try model.newCache(parameters: nil)
            eval(model(prompt, cache: cache))
            let token = prompt[0..., (-1)...]

            /// The median milliseconds of `body`, which returns what it evaluates; `after` puts
            /// the cache back.
            func profile(_ label: String, _ body: () -> [MLXArray], after: () -> Void = {}) {
                let median = GPUWallTime.median(repeats: profileSamples) {
                    let milliseconds = GPUWallTime.milliseconds { eval(body()) }
                    after()
                    return milliseconds
                }
                Inkling2ResultLog.report(
                    "INKLING2-MTP", "profile \(label): \(formatted(median, "%.2f")) ms")
                #expect(median > 0)
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
            emit[mtpTargetLayerIdsKey] = drafter.targetLayerIds
            let output = model(LMInput.Text(tokens: prompt), cache: nil, state: emit)
            let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
            var state = drafter.makeState(parameters: nil)
            drafter.prepareDrafterState(
                target: model, promptTokens: prompt, targetHidden: hidden,
                firstBonus: token.reshaped(-1), positionDeltas: nil, state: &state,
                sampler: ArgMaxSampler())
            eval(state.cache.flatMap(\.state))
            for blockSize in [defaultBlockSize, headBlockSize] {
                profile("draft block \(blockSize)") {
                    [
                        drafter.draftBlock(
                            target: model, lastToken: token.reshaped(-1),
                            lastHidden: MLXArray.zeros([1, 1, 1]), sharedKV: [:],
                            positionDeltas: nil, queryOffset: 0, blockSize: blockSize,
                            state: &state, sampler: ArgMaxSampler())
                    ]
                }
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
        try await pair.perform { context, drafter in
            let model = context.model
            let contexts = [
                (
                    "short", try await chatInput(context, Self.storyPrompt), shortContextTokens,
                    Float?.none
                ),
                (
                    "long",
                    try await chatInput(context, Inkling2CheckpointIntegrationTests.longPrompt),
                    longContextTokens, Float?.some(nearTieLogitGap)
                ),
            ]
            for (label, input, tokens, nearTieGap) in contexts {
                _ = try plainRun(model, input: input, tokens: warmUpTokens)
                for blockSize in measuredBlockSizes {
                    _ = try mtpRun(
                        model, drafter: drafter, input: input, tokens: warmUpTokens,
                        blockSize: blockSize)
                }
                Memory.clearCache()

                // Each repetition runs plain decoding, then every block.
                let repeats = try (0 ..< repetitions).map { _ in
                    (
                        plain: try plainRun(model, input: input, tokens: tokens),
                        mtp: try measuredBlockSizes.map { blockSize in
                            try mtpRun(
                                model, drafter: drafter, input: input, tokens: tokens,
                                blockSize: blockSize)
                        }
                    )
                }
                let plainRuns = repeats.map(\.plain)
                let plainRate = SampleMedian.of(plainRuns.map(\.tokensPerSecond))
                Inkling2ResultLog.report(
                    "INKLING2-MTP",
                    "\(label): prompt \(input.text.tokens.size) tok, plain "
                        + "\(formatted(plainRate)) tok/s")
                for (index, blockSize) in measuredBlockSizes.enumerated() {
                    let runs = repeats.map { $0.mtp[index] }
                    let rate = SampleMedian.of(runs.map(\.tokensPerSecond))
                    let accepted = runs.map(\.accepted).reduce(0, +)
                    let proposed = runs.map(\.proposed).reduce(0, +)
                    let rounds = runs.map(\.rounds).reduce(0, +)
                    Inkling2ResultLog.report(
                        "INKLING2-MTP",
                        "\(label) block \(blockSize): MTP \(formatted(rate)) tok/s "
                            + "(\(formatted(rate / plainRate, "%.2f"))x), acceptance \(accepted)/"
                            + "\(proposed) = "
                            + "\(formatted(Double(accepted) / Double(proposed), "%.3f")), "
                            + "\(formatted(Double(accepted + rounds) / Double(rounds), "%.2f")) "
                            + "tokens per round")
                    for run in runs {
                        await expectMatchesPlain(
                            run, model, input: input, plain: plainRuns[0].tokens,
                            label: "\(label) block \(blockSize)", nearTieGap: nearTieGap)
                    }
                }
            }
        }
    }

    /// The oQ4e target (oMLX layout) beside the 4bit target, with one protocol: the parameter
    /// memory of each, its plain and MTP decode throughput at the short and the long prompt at the
    /// drafter's preferred block and at the shared default (each the median of three runs after a
    /// warm-up), and greedy MTP emitting the tokens of plain greedy decoding (past the sliding
    /// window up to a near tie, see ``nearTieLogitGap``).
    @Test
    func oQ4eTargetBesideThe4BitTarget() async throws {
        let pair = try await Inkling2MTPPairLoader.shared.pair()
        let targets = [
            ("4bit", pair.target), ("oQ4e", try await inkling2MiniOQ4eCheckpoint.container()),
        ]
        for (name, target) in targets {
            try await pair.drafter.perform { drafterContext in
                try await target.perform(nonSendable: drafterContext.model) {
                    context, drafterModel in
                    let drafter = try #require(drafterModel as? Inkling2DraftModel)
                    let model = context.model
                    let preferred = SpeculativeDecodingConfig.mtpBlockSize(nil, for: drafter)
                    let blocks = [preferred, defaultBlockSize]
                    Inkling2ResultLog.report(
                        "INKLING2-OQ4E",
                        "\(name): parameters \(formatted(parameterGiB(model), "%.2f")) GiB, "
                            + "active \(formatted(Double(Memory.activeMemory) / Double(1 << 30), "%.2f")) GiB"
                    )
                    let contexts = [
                        (
                            "short", try await chatInput(context, Self.storyPrompt),
                            shortContextTokens, Float?.none
                        ),
                        (
                            "long",
                            try await chatInput(
                                context, Inkling2CheckpointIntegrationTests.longPrompt),
                            longContextTokens, Float?.some(nearTieLogitGap)
                        ),
                    ]
                    for (label, input, tokens, nearTieGap) in contexts {
                        _ = try plainRun(model, input: input, tokens: warmUpTokens)
                        for blockSize in blocks {
                            _ = try mtpRun(
                                model, drafter: drafter, input: input, tokens: warmUpTokens,
                                blockSize: blockSize)
                        }
                        Memory.clearCache()
                        Memory.peakMemory = 0
                        let plain = try (0 ..< repetitions).map { _ in
                            try plainRun(model, input: input, tokens: tokens)
                        }
                        let plainRate = SampleMedian.of(plain.map(\.tokensPerSecond))
                        Inkling2ResultLog.report(
                            "INKLING2-OQ4E",
                            "\(name) \(label): prompt \(input.text.tokens.size) tok, plain "
                                + "\(formatted(plainRate)) tok/s, peak "
                                + "\(formatted(Double(Memory.peakMemory) / Double(1 << 30), "%.2f")) GiB"
                        )
                        for blockSize in blocks {
                            let runs = try (0 ..< repetitions).map { _ in
                                try mtpRun(
                                    model, drafter: drafter, input: input, tokens: tokens,
                                    blockSize: blockSize)
                            }
                            let rate = SampleMedian.of(runs.map(\.tokensPerSecond))
                            let accepted = runs.map(\.accepted).reduce(0, +)
                            let proposed = runs.map(\.proposed).reduce(0, +)
                            Inkling2ResultLog.report(
                                "INKLING2-OQ4E",
                                "\(name) \(label) block \(blockSize): MTP \(formatted(rate)) tok/s "
                                    + "(\(formatted(rate / plainRate, "%.2f"))x), acceptance "
                                    + "\(accepted)/\(proposed)")
                            for run in runs {
                                await expectMatchesPlain(
                                    run, model, input: input, plain: plain[0].tokens,
                                    label: "\(name) \(label) block \(blockSize)",
                                    nearTieGap: nearTieGap)
                            }
                        }
                    }
                }
            }
        }
    }

    /// Each follow-up turn of a `ChatSession` with MTP on, at the drafter's preferred block,
    /// serves part of its prompt from the cache of the turns before it, keeps speculating, and
    /// answers as a plain session does.
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
        // No block size, as a router sets none: the session takes the drafter's preferred block.
        let speculative = try SpeculativeDecodingConfig(mtpDrafter: pair.drafter)
        let mtpSession = ChatSession(
            pair.target, instructions: instructions, speculativeDecoding: speculative,
            generateParameters: parameters)
        let plainSession = ChatSession(
            pair.target, instructions: instructions, generateParameters: parameters)

        for (index, question) in questions.enumerated() {
            let mtp = try await CollectedGeneration.collect(mtpSession.streamDetails(to: question))
            let plain = try await CollectedGeneration.collect(
                plainSession.streamDetails(to: question))
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
