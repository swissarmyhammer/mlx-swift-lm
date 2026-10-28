// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import MLXVLM
import Testing

@testable import MLXLMCommon

/// Qwen MTP at temperature > 0: drafts that read no target random state, verified so that the
/// emitted tokens keep the distribution of plain decode, in each ``MTPSampledVerification`` mode.
@Suite(.serialized)
struct Qwen35MTPSamplingTests {

    /// The sampled setting of the release benchmark.
    private static let temperature: Float = 0.7

    /// The nucleus of the sampled setting of the release benchmark.
    private static let topP: Float = 0.95

    /// The temperature of the distribution test: hot enough that the second token is often not
    /// the argmax token, so a verifier that keeps more drafts than it should moves the counts.
    private static let distributionTemperature: Float = 1.5

    /// The top-k filter of the distribution test, which keeps the pairs few enough that each
    /// expected count stands as a cell of its own.
    private static let distributionTopK = 4

    /// The seed of a stream that does not vary its seed.
    private static let seed: UInt64 = 7

    /// The prompt the streams of the suite continue.
    private static let prompt = [3, 17, 42, 5]

    /// The tokens a stream of the suite generates.
    private static let maxTokens = 16

    /// The tokens each chat turn of the suite generates.
    private static let chatTurnTokens = 4

    /// The sampled generation parameters of the suite, with `seed` and `verification`.
    private static func sampledParameters(
        seed: UInt64 = seed, verification: MTPSampledVerification, maxTokens: Int = maxTokens,
        temperature: Float = temperature, topK: Int = 0
    ) -> GenerateParameters {
        var parameters = GenerateParameters(
            maxTokens: maxTokens, temperature: temperature, topP: topP, topK: topK, seed: seed)
        parameters.mtpSampledVerification = verification
        return parameters
    }

    /// An MTP iterator of the text fixture over ``prompt``.
    private static func makeIterator(
        _ fixture: Qwen35MTPFixture, parameters: GenerateParameters
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, parameters: parameters,
            blockSize: Qwen35MTPFixture.blockSize)
    }

    /// Reads every token of `iterator`.
    private static func drain(_ iterator: inout MTPSpeculativeTokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return tokens
    }

    @Test("at temperature 0 the MTP tokens equal the plain greedy tokens")
    func temperatureZeroMatchesPlainDecode() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.maxTokens, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.maxTokens, speculative: true)

        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
        #expect(speculative.info.passthroughReason == nil)
    }

    @Test(
        "a sampled text stream speculates without passthrough",
        arguments: MTPSampledVerification.allCases)
    func sampledTextStreamSpeculates(verification: MTPSampledVerification) throws {
        let fixture = try Qwen35MTPFixture()
        var iterator = try Self.makeIterator(
            fixture, parameters: Self.sampledParameters(verification: verification))

        let tokens = Self.drain(&iterator)

        #expect(tokens.count == Self.maxTokens)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        #expect(iterator.acceptedCount > 0)
    }

    @Test(
        "a sampled VLM stream speculates without passthrough",
        arguments: MTPSampledVerification.allCases)
    func sampledVLMStreamSpeculates(verification: MTPSampledVerification) throws {
        let fixture = try Qwen35VLMMTPFixture()
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(Self.prompt.map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter,
            parameters: Self.sampledParameters(verification: verification),
            blockSize: Qwen35MTPFixture.blockSize)

        let tokens = Self.drain(&iterator)

        #expect(tokens.count == Self.maxTokens)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
    }

    @Test(
        "a sampled stream repeats its tokens for its seed",
        arguments: MTPSampledVerification.allCases)
    func sampledStreamRepeatsForSeed(verification: MTPSampledVerification) throws {
        let fixture = try Qwen35MTPFixture()
        let parameters = Self.sampledParameters(verification: verification)
        var first = try Self.makeIterator(fixture, parameters: parameters)
        var second = try Self.makeIterator(fixture, parameters: parameters)

        let firstTokens = Self.drain(&first)
        let secondTokens = Self.drain(&second)

        #expect(firstTokens == secondTokens)
        #expect(first.proposedCount > 0)
    }

    @Test(
        "the first two sampled MTP tokens follow the plain decode distribution",
        arguments: MTPSampledVerification.allCases)
    func firstTwoTokensFollowPlainDistribution(verification: MTPSampledVerification) throws {
        let fixture = try Qwen35MTPFixture()
        let expected = try SampledPairDistribution(
            fixture: fixture, prompt: Self.prompt,
            sampler: TopPSampler(
                temperature: Self.distributionTemperature, topP: Self.topP,
                topK: Self.distributionTopK))

        var counts: [SampledPairDistribution.Pair: Int] = [:]
        var speculatingTrials = 0
        var acceptingTrials = 0
        var rejectingTrials = 0
        for trial in 0 ..< SampledPairDistribution.trialCount {
            var iterator = try Self.makeIterator(
                fixture,
                parameters: Self.sampledParameters(
                    seed: UInt64(trial), verification: verification,
                    maxTokens: SampledPairDistribution.trialTokens,
                    temperature: Self.distributionTemperature, topK: Self.distributionTopK))
            let pair = try SampledPairDistribution.Pair(Self.drain(&iterator))
            counts[pair, default: 0] += 1
            if iterator.proposedCount > 0 { speculatingTrials += 1 }
            if iterator.acceptedCount > 0 { acceptingTrials += 1 } else { rejectingTrials += 1 }
        }

        let pValue = expected.chiSquarePValue(counts: counts)
        #expect(speculatingTrials == SampledPairDistribution.trialCount)
        #expect(pValue > SampledPairDistribution.significance, "p = \(pValue), counts \(counts)")
        #expect(acceptingTrials > 0)
        #expect(rejectingTrials > 0)
    }

    @Test(
        "a sampled chat session speculates on a follow-up turn that reuses the prompt cache",
        arguments: MTPSampledVerification.allCases)
    func sampledChatSessionReusesPromptCache(verification: MTPSampledVerification) async throws {
        let fixture = try Qwen35MTPFixture()
        let session = ChatSession(
            fixture.makeModelContext(),
            speculativeDecoding: try fixture.makeSpeculativeDecodingConfig(),
            generateParameters: Self.sampledParameters(
                verification: verification, maxTokens: Self.chatTurnTokens))

        let first = try await collectSampledChat(
            session.streamDetails(to: fixture.text(for: [1])))
        let second = try await collectSampledChat(
            session.streamDetails(to: fixture.text(for: [2])))

        #expect((first.proposedDraftTokens ?? 0) > 0)
        #expect(second.cachedPromptTokenCount > 0)
        #expect(second.promptTokenCount < second.cachedPromptTokenCount)
        #expect((second.proposedDraftTokens ?? 0) > 0)
        #expect(second.passthroughReason == nil)
        #expect(second.speculativeDecodingFallbackReason == nil)
    }
}

/// The completion information of one chat turn.
private func collectSampledChat(
    _ stream: AsyncThrowingStream<Generation, Error>
) async throws -> GenerateCompletionInfo {
    var info: GenerateCompletionInfo?
    for try await event in stream {
        if let completion = event.info { info = completion }
    }
    return try #require(info)
}

@Suite
struct QwenMTPDistributionDraftTests {

    /// A prompt for the drafter state of the tests.
    private static let promptTokens: [Int32] = [3, 17, 42, 5]

    /// The bonus token the target sampled after ``promptTokens``.
    private static let firstBonus: Int32 = 9

    /// A drafter state prefilled over ``promptTokens`` from random target hidden rows.
    private static func preparedState(_ fixture: Qwen35MTPFixture) -> MTPDrafterState {
        var state = fixture.drafter.makeState(parameters: nil)
        let targetHidden = MLXRandom.normal(
            [1, promptTokens.count, Qwen35MTPFixture.hiddenSize],
            key: MLXRandom.key(UInt64(Qwen35MTPFixture.defaultSeed)))
        fixture.drafter.prepareDrafterState(
            target: fixture.target, promptTokens: MLXArray(promptTokens)[.newAxis, 0...],
            targetHidden: targetHidden, firstBonus: MLXArray([firstBonus]),
            positionDeltas: nil, state: &state, sampler: ArgMaxSampler())
        return state
    }

    @Test("the seed draft with logits is the argmax seed under an argmax sampler")
    func seedDraftWithLogitsMatchesArgmaxSeed() throws {
        let fixture = try Qwen35MTPFixture()
        var tokenState = Self.preparedState(fixture)
        var distributionState = Self.preparedState(fixture)

        let token = fixture.drafter.draftBlock(
            target: fixture.target, lastToken: MLXArray([Self.firstBonus]),
            lastHidden: MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize]), sharedKV: [:],
            positionDeltas: nil, queryOffset: Self.promptTokens.count,
            blockSize: Qwen35MTPFixture.blockSize, state: &tokenState, sampler: ArgMaxSampler())
        let output = fixture.drafter.draftBlockWithLogits(
            target: fixture.target, lastToken: MLXArray([Self.firstBonus]),
            lastHidden: MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize]), sharedKV: [:],
            positionDeltas: nil, queryOffset: Self.promptTokens.count,
            blockSize: Qwen35MTPFixture.blockSize, state: &distributionState, processor: nil,
            sampler: ArgMaxSampler())

        #expect(output.tokens.asArray(Int.self) == token.asArray(Int.self))
        #expect(output.processedLogits.shape == [1, 1, Qwen35MTPFixture.vocabularySize])
        #expect(
            argMax(output.processedLogits, axis: -1).asArray(Int.self)
                == token.asArray(Int.self))
        #expect(distributionState.seedToken == nil)
        #expect(distributionState.seedHidden == nil)
        #expect(distributionState.proposalAppended == 0)
        #expect(distributionState.nextPosition == tokenState.nextPosition)
    }

    @Test("a chained draft with logits appends its tentative entries")
    func chainedDraftWithLogitsAppendsTentativeEntries() throws {
        let fixture = try Qwen35MTPFixture()
        var state = fixture.drafter.makeState(parameters: nil)

        let output = fixture.drafter.draftBlockWithLogits(
            target: fixture.target, lastToken: MLXArray([Self.firstBonus]),
            lastHidden: MLXArray.zeros([1, 1, Qwen35MTPFixture.hiddenSize]), sharedKV: [:],
            positionDeltas: nil, queryOffset: 0, blockSize: Qwen35MTPFixture.blockSize,
            state: &state, processor: nil, sampler: ArgMaxSampler())

        let draftCount = Qwen35MTPFixture.blockSize - 1
        #expect(output.tokens.shape == [1, draftCount])
        #expect(output.processedLogits.shape == [1, draftCount, Qwen35MTPFixture.vocabularySize])
        #expect(
            argMax(output.processedLogits, axis: -1).asArray(Int.self)
                == output.tokens.asArray(Int.self))
        #expect(state.proposalAppended == draftCount)
        #expect(state.nextPosition == draftCount)
        #expect(state.isAligned)
    }
}

@Suite
struct BatchedSampledMTPVerificationTests {

    /// A logit far above the others, so the sampler draws its token every time.
    private static let peak: Float = 1_000

    /// The vocabulary of the logits of the tests.
    private static let vocabularySize = 12

    /// The temperature of the samplers of the tests.
    private static let temperature: Float = 0.7

    /// The nucleus of the top-p sampler of the tests.
    private static let topP: Float = 0.95

    /// The seed of the samplers of the tests.
    private static let seed: UInt64 = 1

    /// Logits whose row `i` peaks at `peaks[i]`.
    private static func peakedLogits(_ peaks: [Int]) -> MLXArray {
        let rows = peaks.map { peakToken in
            (0 ..< vocabularySize).map { $0 == peakToken ? peak : 0 }
        }
        return MLXArray(rows.flatMap { $0 }, [peaks.count, vocabularySize])
    }

    @Test("sampled verification keeps the drafts its samples match, then the first sample after")
    func sampledVerificationStopsAtFirstMismatch() {
        let targetPeaks = [5, 9, 2]
        let drafts = MLXArray([Int32(5), 7])

        let result = BatchedMTPVerification.sampled(
            logits: Self.peakedLogits(targetPeaks), draftTokens: drafts,
            sampler: TopPSampler(temperature: Self.temperature, topP: Self.topP, seed: Self.seed))

        #expect(result.accepted == 1)
        #expect(result.tokens == [5, 9])
        #expect(result.finalToken.asArray(Int.self) == [9])
    }

    @Test("sampled verification emits the bonus after every matched draft")
    func sampledVerificationEmitsBonusAfterFullMatch() {
        let targetPeaks = [5, 7, 2]
        let drafts = MLXArray([Int32(5), 7])

        let result = BatchedMTPVerification.sampled(
            logits: Self.peakedLogits(targetPeaks), draftTokens: drafts,
            sampler: CategoricalSampler(temperature: Self.temperature, seed: Self.seed))

        #expect(result.accepted == 2)
        #expect(result.tokens == targetPeaks)
        #expect(result.finalToken.asArray(Int.self) == [2])
    }
}

/// The exact distribution of the first two tokens plain decode samples after a prompt, and the
/// chi-square test of observed MTP counts against it.
private struct SampledPairDistribution {

    /// The first two sampled tokens of one stream.
    struct Pair: Hashable {
        let first: Int
        let second: Int

        /// The pair of the first two of `tokens`.
        init(_ tokens: [Int]) throws {
            try #require(tokens.count >= SampledPairDistribution.pairLength)
            first = tokens[0]
            second = tokens[1]
        }

        init(first: Int, second: Int) {
            self.first = first
            self.second = second
        }
    }

    /// The tokens of a pair.
    static let pairLength = 2

    /// The tokens each trial generates: the prefill bonus, then one round of a draft and its
    /// verification. A stream with a single token left decodes it without a draft.
    static let trialTokens = pairLength + 1

    /// The seeded trials of one test.
    static let trialCount = 400

    /// The smallest expected count a chi-square cell keeps; smaller cells pool into one.
    static let minimumExpectedCount = 5.0

    /// The p-value below which the counts reject the plain decode distribution.
    static let significance = 0.01

    /// The probability of each pair plain decode samples.
    let probabilities: [Pair: Double]

    /// Computes the distribution from one target forward per possible first token.
    ///
    /// - Parameters:
    ///   - fixture: The fixture whose target decodes.
    ///   - prompt: The prompt tokens.
    ///   - sampler: The sampler of plain decode, which states its distribution.
    init(fixture: Qwen35MTPFixture, prompt: [Int], sampler: TopPSampler) throws {
        let first = try Self.nextProbabilities(fixture, tokens: prompt, sampler: sampler)
        var probabilities: [Pair: Double] = [:]
        for (firstToken, firstProbability) in first.enumerated() where firstProbability > 0 {
            let second = try Self.nextProbabilities(
                fixture, tokens: prompt + [firstToken], sampler: sampler)
            for (secondToken, secondProbability) in second.enumerated()
            where secondProbability > 0 {
                probabilities[Pair(first: firstToken, second: secondToken)] =
                    firstProbability * secondProbability
            }
        }
        self.probabilities = probabilities
    }

    /// The probability of each next token after `tokens`, from a fresh target cache.
    private static func nextProbabilities(
        _ fixture: Qwen35MTPFixture, tokens: [Int], sampler: TopPSampler
    ) throws -> [Double] {
        let cache = try fixture.target.newCache(parameters: nil)
        let logits = fixture.target(
            MLXArray(tokens.map(Int32.init))[.newAxis, 0...], cache: cache)
        let logProbabilities = sampler.logProbabilities(logits: logits[0..., -1, 0...])
        return exp(logProbabilities).asType(.float32).asArray(Float.self).map(Double.init)
    }

    /// The chi-square p-value of `counts` against ``probabilities``.
    ///
    /// Cells whose expected count is below ``minimumExpectedCount`` pool into one cell, and an
    /// observed pair outside the support counts in that cell.
    func chiSquarePValue(counts: [Pair: Int]) -> Double {
        let total = Double(counts.values.reduce(0, +))
        let isKept = { (pair: Pair) in
            (probabilities[pair] ?? 0) * total >= Self.minimumExpectedCount
        }
        let kept = probabilities.keys.filter(isKept)
        let keptStatistic = kept.reduce(0.0) { sum, pair in
            let expected = probabilities[pair, default: 0] * total
            let observed = Double(counts[pair] ?? 0)
            return sum + (observed - expected) * (observed - expected) / expected
        }
        let pooledExpected = total - kept.reduce(0.0) { $0 + probabilities[$1, default: 0] * total }
        let pooledObserved = Double(counts.filter { !isKept($0.key) }.values.reduce(0, +))
        let pooledStatistic =
            pooledExpected > 0
            ? (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected)
                / pooledExpected
            : 0
        let cells = kept.count + (pooledExpected > 0 ? 1 : 0)
        return RegularizedGamma.chiSquareSurvival(
            statistic: keptStatistic + pooledStatistic, degreesOfFreedom: cells - 1)
    }
}

/// The regularized upper incomplete gamma function `Q(a, x)`, and the chi-square survival
/// function it gives.
private enum RegularizedGamma {

    /// The chi-square distribution of `k` degrees of freedom is the gamma distribution of shape
    /// `k / 2` at `x / 2`.
    private static let chiSquareScale = 2.0

    /// The step of the denominator term of the continued fraction.
    private static let fractionStep = 2.0

    /// The iterations after which the series and the continued fraction stop.
    private static let maximumIterations = 500

    /// The relative change below which an iteration has converged.
    private static let tolerance = 1e-12

    /// The smallest magnitude the continued fraction divides by.
    private static let tiny = 1e-300

    /// The probability that a chi-square variable of `degreesOfFreedom` exceeds `statistic`.
    static func chiSquareSurvival(statistic: Double, degreesOfFreedom: Int) -> Double {
        upper(shape: Double(degreesOfFreedom) / chiSquareScale, x: statistic / chiSquareScale)
    }

    /// `Q(shape, x)`.
    private static func upper(shape: Double, x: Double) -> Double {
        guard x > 0 else { return 1 }
        return x < shape + 1
            ? 1 - lowerSeries(shape: shape, x: x) : upperFraction(shape: shape, x: x)
    }

    /// `P(shape, x)` by its power series.
    private static func lowerSeries(shape: Double, x: Double) -> Double {
        var denominator = shape
        var term = 1 / shape
        var sum = term
        for _ in 0 ..< maximumIterations {
            denominator += 1
            term *= x / denominator
            sum += term
            if abs(term) < abs(sum) * tolerance { break }
        }
        return sum * Foundation.exp(-x + shape * Foundation.log(x) - lgamma(shape))
    }

    /// `Q(shape, x)` by its continued fraction (modified Lentz).
    private static func upperFraction(shape: Double, x: Double) -> Double {
        var b = x + 1 - shape
        var c = 1 / tiny
        var d = 1 / b
        var fraction = d
        for index in 1 ... maximumIterations {
            let step = Double(index)
            let a = -step * (step - shape)
            b += fractionStep
            d = a * d + b
            d = abs(d) < tiny ? tiny : d
            c = b + a / c
            c = abs(c) < tiny ? tiny : c
            d = 1 / d
            let delta = d * c
            fraction *= delta
            if abs(delta - 1) < tolerance { break }
        }
        return Foundation.exp(-x + shape * Foundation.log(x) - lgamma(shape)) * fraction
    }
}
