import Foundation
import MLX
import MLXLMTestSupport
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Bound for comparing drafter cache entries that the same tokens reach through matmuls of
/// different shapes. Wider on machines whose float32 matmuls run as TF32; see
/// ``MatmulPrecision``.
private let drafterCacheTolerance = MatmulPrecision.tolerance(
    float32: ieeeFloat32Tolerance, reduced: tf32ReshapedMatmulTolerance)

/// Expects `actual` drafter cache arrays to match `expected` pair by pair: the same count, the
/// same shapes, and values within ``drafterCacheTolerance``.
///
/// - Parameters:
///   - actual: The drafter cache arrays under test.
///   - expected: The drafter cache arrays of the reference.
///   - sourceLocation: The caller of the expectations.
/// - Throws: When the counts or the shapes differ.
private func expectDrafterCachesMatch(
    _ actual: [MLXArray], _ expected: [MLXArray],
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    try #require(actual.count == expected.count, sourceLocation: sourceLocation)
    for (actual, expected) in zip(actual, expected) {
        eval(actual, expected)
        try #require(actual.shape == expected.shape, sourceLocation: sourceLocation)
        #expect(
            allClose(
                actual, expected, rtol: drafterCacheTolerance, atol: drafterCacheTolerance
            ).item(Bool.self), sourceLocation: sourceLocation)
    }
}

/// The drafter state API beyond a cold prefill at position zero: a cold start at a position
/// the target cache already holds, and trimming and appending entries.
@Suite(.serialized)
struct Qwen35MTPDrafterStateTests {

    /// Prompt tokens of the fixture vocabulary.
    private let prompt = [3, 9, 14, 22, 5, 40]

    /// The token the target sampled after ``prompt``.
    private let bonusToken: Int32 = 17

    /// ``bonusToken`` as the iterator hands it to the drafter.
    private var bonus: MLXArray { MLXArray([bonusToken]) }

    /// The prompt token the next turn appends after ``bonusToken``.
    private let nextTurnPromptToken: Int32 = 8

    /// The token the target sampled after the next turn's suffix.
    private let nextTurnBonusToken: Int32 = 11

    /// Suffix tokens the VLM cold start feeds after the cached prefix.
    private let vlmSuffixTokens: [Int32] = [5, 6]

    /// The token the VLM target sampled after ``vlmSuffixTokens``.
    private let vlmBonusToken: Int32 = 7

    /// The multimodal RoPE position delta the VLM cold start applies.
    private let vlmPositionDelta: Int32 = 2

    /// Entries the trim and append test removes and then appends again.
    private let replayedEntryCount = 3

    /// Tokens of the prompt that a warm target cache already holds.
    private let cachedPrefixLength = 4

    /// Tokens each greedy decode generates.
    private let generatedTokenCount = 8

    /// The target hidden rows of `tokens`, emitted as the MTP iterator asks for them.
    private func targetHidden(
        _ fixture: Qwen35MTPFixture, tokens: [Int], cache: [KVCache]? = nil
    ) throws -> MLXArray {
        try targetHiddenStates(
            fixture.target, tokens: MLXArray(tokens.map(Int32.init))[.newAxis, 0...],
            cache: cache)
    }

    /// A drafter state prefilled over ``prompt`` from position zero.
    private func preparedState(
        _ fixture: Qwen35MTPFixture, hidden: MLXArray, sampler: any LogitSampler
    ) -> MTPDrafterState {
        var state = fixture.drafter.makeState(parameters: nil)
        fixture.drafter.prepareDrafterState(
            target: fixture.target,
            promptTokens: MLXArray(prompt.map(Int32.init))[.newAxis, 0...],
            targetHidden: hidden, firstBonus: bonus, positionDeltas: nil, state: &state,
            sampler: sampler)
        return state
    }

    /// Consumes the seed proposal, then drafts one tentative entry past it.
    private func appendTentativeEntry(
        _ fixture: Qwen35MTPFixture, hidden: MLXArray, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let oneDraftBlockSize = MTPDraftBlockOutput.minimumChainedMTPBlockSize
        for _ in 0 ..< oneDraftBlockSize {
            let proposal = fixture.drafter.draftBlock(
                target: fixture.target, lastToken: bonus,
                lastHidden: hidden[0..., (-1)..., 0...], sharedKV: [:], positionDeltas: nil,
                queryOffset: state.nextPosition, blockSize: oneDraftBlockSize,
                state: &state, sampler: sampler)
            eval(proposal)
        }
    }

    @Test("trimming entries and appending them again matches a cold prefill")
    func trimAndAppendMatchColdPrefill() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        let cold = preparedState(fixture, hidden: hidden, sampler: sampler)
        var replayed = preparedState(fixture, hidden: hidden, sampler: sampler)
        appendTentativeEntry(fixture, hidden: hidden, state: &replayed, sampler: sampler)
        #expect(replayed.proposalAppended == 1)

        let removed = replayed.removeLastEntries(replayed.proposalAppended + replayedEntryCount)
        #expect(removed)
        #expect(replayed.proposalAppended == 0)
        #expect(replayed.seedToken == nil)
        let keptEntries = prompt.count - replayedEntryCount
        let tokens = MLXArray(prompt[(keptEntries + 1)...].map(Int32.init))[.newAxis, 0...]
        #expect(
            fixture.drafter.appendDrafterState(
                target: fixture.target,
                tokens: concatenated([tokens, bonus[.newAxis, 0...]], axis: 1),
                targetHidden: hidden[0..., keptEntries..., 0...], positionDeltas: nil,
                state: &replayed, sampler: sampler))

        #expect(replayed.nextPosition == cold.nextPosition)
        #expect(replayed.isAligned)
        try expectDrafterCachesMatch(
            replayed.cache.flatMap(\.state), cold.cache.flatMap(\.state))
        let replayedSeed = try #require(replayed.seedToken)
        let coldSeed = try #require(cold.seedToken)
        #expect(replayedSeed.asArray(Int32.self) == coldSeed.asArray(Int32.self))
    }

    @Test("finalization drops tentative proposal entries before the target boundary")
    func finalizationDropsTentativeEntries() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        var state = preparedState(fixture, hidden: hidden, sampler: sampler)
        appendTentativeEntry(fixture, hidden: hidden, state: &state, sampler: sampler)

        #expect(
            fixture.drafter.finalizeDrafterState(
                target: fixture.target, targetBoundaryHidden: hidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: prompt.count, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == prompt.count - 1)
        #expect(state.proposalAppended == 0)
        #expect(state.isAligned)
    }

    @Test("a cold start at a cached position resumes as a continuation")
    func coldStartAtCachedPositionResumes() throws {
        let fixture = try Qwen35MTPFixture()
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)
        let suffix = Array(prompt.dropFirst(cachedPrefixLength))
        let suffixHidden = try targetHidden(fixture, tokens: suffix, cache: cache)
        var state = fixture.drafter.makeState(parameters: nil)

        #expect(
            fixture.drafter.coldStartDrafterState(
                target: fixture.target,
                suffixTokens: MLXArray(suffix.map(Int32.init))[.newAxis, 0...],
                suffixTargetHidden: suffixHidden, firstBonus: bonus,
                position: cachedPrefixLength, positionDeltas: nil, state: &state,
                sampler: sampler))
        let seed = try #require(state.seedToken)
        eval(seed)
        #expect(state.cacheStartPosition == cachedPrefixLength)
        #expect(state.nextPosition == prompt.count)
        #expect(state.cache.allSatisfy { $0.offset == suffix.count })

        #expect(
            fixture.drafter.finalizeDrafterState(
                target: fixture.target, targetBoundaryHidden: suffixHidden[0..., (-1)..., 0...],
                targetProcessedTokenCount: prompt.count, discardedTargetTokens: 0,
                positionDeltas: nil, state: &state))
        #expect(state.nextPosition == prompt.count - 1)

        let nextSuffix = MLXArray([bonusToken, nextTurnPromptToken])[.newAxis, 0...]
        #expect(
            fixture.drafter.resumeDrafterState(
                target: fixture.target, suffixTokens: nextSuffix,
                suffixTargetHidden: MLXArray.zeros([
                    1, nextSuffix.dim(1), Qwen35MTPFixture.hiddenSize,
                ]),
                targetBoundaryHidden: suffixHidden[0..., (-1)..., 0...],
                firstBonus: MLXArray([nextTurnBonusToken]), positionDeltas: nil, state: &state,
                sampler: sampler))
        #expect(state.nextPosition == prompt.count + nextSuffix.dim(1))
        #expect(state.isAligned)
    }

    @Test("the VLM drafter cold-starts at a cached position with its position deltas")
    func vlmColdStartAtCachedPosition() throws {
        let configuration = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let target = MLXVLM.Qwen35(configuration)
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(configuration)
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let suffix = MLXArray(vlmSuffixTokens)[.newAxis, 0...]
        var state = drafter.makeState(parameters: nil)

        #expect(
            drafter.coldStartDrafterState(
                target: target, suffixTokens: suffix,
                suffixTargetHidden: MLXArray.zeros(
                    [1, suffix.dim(1), configuration.textConfiguration.hiddenSize]),
                firstBonus: MLXArray([vlmBonusToken]), position: cachedPrefixLength,
                positionDeltas: MLXArray([vlmPositionDelta]), state: &state, sampler: sampler))
        let seed = try #require(state.seedToken)
        eval(seed)
        #expect(state.cacheStartPosition == cachedPrefixLength)
        #expect(state.nextPosition == cachedPrefixLength + suffix.dim(1))
        #expect(state.isAligned)
    }

    @Test("a warm target cache without a continuation speculates and matches greedy decode")
    func warmCacheWithoutContinuationSpeculates() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)

        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(
                tokens: MLXArray(prompt.dropFirst(cachedPrefixLength).map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, mainCache: cache,
            parameters: GenerateParameters(maxTokens: generatedTokenCount, temperature: 0),
            blockSize: Qwen35MTPFixture.blockSize)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }

        #expect(tokens == plain.tokens)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
        #expect(iterator.acceptedCount > 0)
    }

    /// What an MTP decode with a carried continuation left: its tokens, where the drafter
    /// entries began and ended right after the prefill, and whether the stream speculated.
    private struct ContinuedDecode {
        let tokens: [Int]
        let drafterStartPosition: Int
        let drafterNextPosition: Int
        let drafterIsAligned: Bool
        let passthroughReason: String?
        let proposedCount: Int
    }

    /// A continuation whose drafter holds ``prompt`` and whose boundary witnesses
    /// `targetProcessedTokenCount` target tokens.
    private func continuation(
        _ fixture: Qwen35MTPFixture, targetProcessedTokenCount: Int
    ) throws -> MTPDrafterContinuation {
        let sampler = GenerateParameters(temperature: 0).makeSampler()
        let hidden = try targetHidden(fixture, tokens: prompt)
        return MTPDrafterContinuation(
            state: preparedState(fixture, hidden: hidden, sampler: sampler),
            targetBoundaryHidden: hidden[0..., (-1)..., 0...],
            targetProcessedTokenCount: targetProcessedTokenCount,
            drafterID: fixture.drafter.drafterID)
    }

    /// Decodes greedily with MTP after the tokens of ``prompt`` that `cache` does not hold,
    /// carrying `continuation`. The prompt is shorter than one prefill chunk, so the target
    /// prefills it without calling the chunk handler.
    private func continuedDecode(
        _ fixture: Qwen35MTPFixture, cache: [KVCache], cachedTokenCount: Int,
        continuation: MTPDrafterContinuation
    ) throws -> ContinuedDecode {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.dropFirst(cachedTokenCount).map(Int32.init))),
            mainModel: fixture.target, drafter: fixture.drafter, mainCache: cache,
            mtpContinuation: continuation,
            parameters: GenerateParameters(maxTokens: generatedTokenCount, temperature: 0),
            blockSize: Qwen35MTPFixture.blockSize)
        let primed = try #require(iterator.drafterState)
        let drafterIsAligned = primed.isAligned
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return ContinuedDecode(
            tokens: tokens, drafterStartPosition: primed.cacheStartPosition,
            drafterNextPosition: primed.nextPosition, drafterIsAligned: drafterIsAligned,
            passthroughReason: iterator.passthroughReason, proposedCount: iterator.proposedCount)
    }

    @Test(
        "an unaligned continuation on a warm cache cold-starts the drafter at the cached position")
    func unalignedContinuationColdStartsAtCachedPosition() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)
        let cache = try fixture.target.newCache(parameters: nil)
        _ = try targetHidden(
            fixture, tokens: Array(prompt.prefix(cachedPrefixLength)), cache: cache)

        let decode = try continuedDecode(
            fixture, cache: cache, cachedTokenCount: cachedPrefixLength,
            continuation: try continuation(fixture, targetProcessedTokenCount: prompt.count))

        #expect(decode.passthroughReason == nil)
        #expect(decode.drafterStartPosition == cachedPrefixLength)
        #expect(decode.drafterNextPosition == prompt.count)
        #expect(decode.drafterIsAligned)
        #expect(decode.proposedCount > 0)
        #expect(decode.tokens == plain.tokens)
    }

    @Test("an unaligned continuation on an empty cache prefills the drafter from position zero")
    func unalignedContinuationPrefillsFromPositionZero() async throws {
        let fixture = try Qwen35MTPFixture()
        let plain = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: generatedTokenCount, speculative: false)

        let decode = try continuedDecode(
            fixture, cache: try fixture.target.newCache(parameters: nil), cachedTokenCount: 0,
            continuation: try continuation(fixture, targetProcessedTokenCount: cachedPrefixLength))

        #expect(decode.passthroughReason == nil)
        #expect(decode.drafterStartPosition == 0)
        #expect(decode.drafterNextPosition == prompt.count)
        #expect(decode.drafterIsAligned)
        #expect(decode.proposedCount > 0)
        #expect(decode.tokens == plain.tokens)
    }
}

/// The MTP prefill of a prompt in chunks, each of which feeds the drafter, against the reference
/// prefill of the whole prompt in one forward.
@Suite(.serialized)
struct Qwen35MTPChunkedPrefillTests {

    /// A Qwen 3.5 target with its MTP drafter.
    enum Target: String, CaseIterable, CustomTestStringConvertible {

        /// The MLXLLM text target of ``Qwen35MTPFixture``.
        case text

        /// The MLXVLM target of ``Qwen35VLMMTPFixture``.
        case visionLanguage

        var testDescription: String { rawValue }

        /// The target and its drafter, on seeded weights.
        func models() throws -> (target: any LanguageModel, drafter: any MTPDrafterModel) {
            switch self {
            case .text:
                let fixture = try Qwen35MTPFixture()
                return (fixture.target, fixture.drafter)
            case .visionLanguage:
                let fixture = try Qwen35VLMMTPFixture()
                return (fixture.target, fixture.drafter)
            }
        }
    }

    /// What one MTP decode left: its tokens, the drafter cache right after the prefill, and
    /// whether it speculated.
    private struct MTPDecode {
        let tokens: [Int]
        let drafterCache: [MLXArray]
        let passthroughReason: String?
        let proposedCount: Int
    }

    /// Prompt tokens of the chunked prefill: more than 300, so a step of ``stepSize`` cuts
    /// several chunks.
    private let promptLength = 320

    /// The chunk ceiling of the chunked prefill.
    private let stepSize = 64

    /// The chunks the balanced prefill cuts from the 319 positions before the reserved logits
    /// position at a ceiling of 64: five chunks of at most 64.
    private let expectedChunkCount = 5

    /// Prompt tokens a warm target cache already holds before the continuation.
    private let cachedPrefixLength = 96

    /// Tokens each decode generates.
    private let generatedTokenCount = 8

    /// Prompt tokens of the prefill whose peak memory is measured.
    private let longPromptLength = 4096

    /// The chunk ceiling of the prefill whose peak memory is measured.
    private let longPromptStepSize = 512

    /// The seed of the prompt tokens.
    private let promptSeed: UInt64 = 7

    /// `length` prompt tokens of the fixture vocabulary, from ``promptSeed``.
    private func prompt(length: Int) -> MLXArray {
        seededTokens(length, vocabularySize: Qwen35MTPFixture.vocabularySize, seed: promptSeed)
    }

    /// Generation parameters of a greedy decode with `prefill`.
    private func parameters(prefill: PrefillParameters, maxTokens: Int) -> GenerateParameters {
        var parameters = GenerateParameters(maxTokens: maxTokens, temperature: 0)
        parameters.prefill = prefill
        return parameters
    }

    /// Decodes greedily with MTP after `prompt`, on `cache` when it is given.
    private func decode(
        _ models: (target: any LanguageModel, drafter: any MTPDrafterModel), prompt: MLXArray,
        prefill: PrefillParameters, cache: [KVCache]? = nil, state: LMOutput.State? = nil
    ) throws -> MTPDecode {
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: models.target, drafter: models.drafter,
            mainCache: cache, mainState: state,
            parameters: parameters(prefill: prefill, maxTokens: generatedTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        let drafterCache = try #require(iterator.drafterState).cache.flatMap(\.state)
        eval(drafterCache)
        var tokens: [Int] = []
        while let token = iterator.next() {
            tokens.append(token)
        }
        return MTPDecode(
            tokens: tokens, drafterCache: drafterCache,
            passthroughReason: iterator.passthroughReason, proposedCount: iterator.proposedCount)
    }

    /// A cache of `target` that holds `prefix`, and the model state of that cache.
    private func warmCache(_ target: any LanguageModel, prefix: MLXArray) throws -> (
        cache: [KVCache], state: LMOutput.State
    ) {
        let cache = try target.newCache(parameters: nil)
        var state: LMOutput.State?
        if case .logits(let output) = try target.prepare(
            LMInput(tokens: prefix), cache: cache, state: nil, prefill: .init())
        {
            state = output.state
        }
        return (cache, try #require(state))
    }

    @Test("a chunked MTP prefill matches the prefill in one forward", arguments: Target.allCases)
    func chunkedPrefillMatchesUnchunked(target: Target) throws {
        let models = try target.models()
        let prompt = prompt(length: promptLength)

        let unchunked = try decode(models, prompt: prompt, prefill: .init(chunking: .unchunked))
        let chunked = try decode(models, prompt: prompt, prefill: .init(stepSize: stepSize))

        #expect(chunked.tokens == unchunked.tokens)
        #expect(chunked.passthroughReason == nil)
        #expect(chunked.proposedCount > 0)
        try expectDrafterCachesMatch(chunked.drafterCache, unchunked.drafterCache)
    }

    @Test(
        "the target prepare calls the chunk handler once for each chunk", arguments: Target.allCases
    )
    func prepareCallsChunkHandlerForEachChunk(target: Target) throws {
        let models = try target.models()
        let prompt = prompt(length: promptLength)
        var chunks: [(tokens: [Int32], hiddenRows: Int)] = []
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpPrefillChunkHandlerKey] = { tokens, hiddenStates, _ in
            chunks.append((tokens.asArray(Int32.self), hiddenStates.dim(1)))
        }

        _ = try models.target.prepare(
            LMInput(tokens: prompt), cache: try models.target.newCache(parameters: nil),
            state: state, prefill: .init(stepSize: stepSize))

        #expect(chunks.count == expectedChunkCount)
        #expect(chunks.allSatisfy { $0.tokens.count == $0.hiddenRows })
        #expect(chunks.allSatisfy { $0.tokens.count <= stepSize })
        #expect(chunks.flatMap(\.tokens) == Array(prompt.asArray(Int32.self).dropLast()))
    }

    @Test("a warm VLM continuation prefilled in chunks speculates and matches greedy decode")
    func warmVLMContinuationSpeculates() throws {
        let fixture = try Qwen35VLMMTPFixture()
        let prompt = prompt(length: promptLength)
        let prefix = prompt[..<cachedPrefixLength]
        let suffix = prompt[cachedPrefixLength...]

        let plainStart = try warmCache(fixture.target, prefix: prefix)
        var plainIterator = try TokenIterator(
            input: LMInput(tokens: suffix), model: fixture.target, cache: plainStart.cache,
            state: plainStart.state,
            parameters: parameters(
                prefill: .init(stepSize: stepSize), maxTokens: generatedTokenCount))
        var plainTokens: [Int] = []
        while let token = plainIterator.next() {
            plainTokens.append(token)
        }

        let mtpStart = try warmCache(fixture.target, prefix: prefix)
        let mtp = try decode(
            (fixture.target, fixture.drafter), prompt: suffix, prefill: .init(stepSize: stepSize),
            cache: mtpStart.cache, state: mtpStart.state)

        #expect(mtp.tokens == plainTokens)
        #expect(mtp.passthroughReason == nil)
        #expect(mtp.proposedCount > 0)
    }

    @Test("a chunked MTP prefill of a long prompt peaks lower than the prefill in one forward")
    func chunkedPrefillPeaksLower() throws {
        let fixture = try Qwen35MTPFixture()
        let prompt = prompt(length: longPromptLength)

        let unchunked = try peakPrefillMemory(
            fixture, prompt: prompt, prefill: .init(chunking: .unchunked))
        let chunked = try peakPrefillMemory(
            fixture, prompt: prompt, prefill: .init(stepSize: longPromptStepSize))

        #expect(chunked < unchunked)
    }

    /// The peak memory of an MTP prefill of `prompt`, with the drafter state it builds.
    private func peakPrefillMemory(
        _ fixture: Qwen35MTPFixture, prompt: MLXArray, prefill: PrefillParameters
    ) throws -> Int {
        Memory.clearCache()
        Memory.peakMemory = 0
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: prompt), mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters(prefill: prefill, maxTokens: 1),
            blockSize: Qwen35MTPFixture.blockSize)
        eval(try #require(iterator.drafterState).cache.flatMap(\.state))
        return Memory.peakMemory
    }
}
