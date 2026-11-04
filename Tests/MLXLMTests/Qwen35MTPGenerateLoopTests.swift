import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Where a token of an MTP stream of the fixture sits in its round.
enum Qwen35MTPRoundSlot: CaseIterable, Sendable {

    /// An accepted draft. The verify pass of its round fed it to the main cache.
    case draft

    /// The verifier's correction after a rejected draft. No forward pass has fed it yet.
    case correction

    /// The verifier's bonus after every draft of its round was accepted. No forward pass has
    /// fed it yet.
    case bonus
}

/// How the first call of a continued decode ends.
enum Qwen35MTPFirstCall: CaseIterable, Sendable {

    /// The first call generates its whole token budget, over several MTP rounds.
    case complete

    /// The first call generates the prefill bonus only, and runs no MTP round.
    case withoutRounds

    /// The consumer stops reading the first call early, which cancels it.
    case cancelled

    /// The token budget of a complete first call: several rounds of ``Qwen35MTPFixture/blockSize``.
    private static let completeTokenBudget = 6

    /// The token budget of a first call that ends at its prefill bonus.
    private static let prefillBonusOnlyTokenBudget = 1

    /// The token budget of a cancelled first call: more than it generates before the
    /// cancellation reaches it.
    private static let cancelledTokenBudget = 24

    /// The tokens the consumer reads before it cancels the first call.
    private static let tokensReadBeforeCancellation = 2

    /// The tokens the first call may generate.
    var tokenBudget: Int {
        switch self {
        case .complete: Self.completeTokenBudget
        case .withoutRounds: Self.prefillBonusOnlyTokenBudget
        case .cancelled: Self.cancelledTokenBudget
        }
    }

    /// The tokens the consumer reads before it stops, or `nil` to read the whole stream.
    var consumedTokenCount: Int? {
        switch self {
        case .complete, .withoutRounds: nil
        case .cancelled: Self.tokensReadBeforeCancellation
        }
    }
}

/// The contract between the MTP iterator and the shared generation loop: the cache state that a
/// stop token reports, the drafter continuation that the token-task overload gives back, the
/// state the iterator exports, and plain decode for log probabilities.
@Suite(.serialized)
struct Qwen35MTPGenerateLoopContractTests {

    /// Prompt tokens of the fixture vocabulary.
    private let prompt = [3, 9, 14, 22, 5, 40]

    /// Tokens the slot classification decodes. Enough for each slot to hold a token that does
    /// not occur earlier in the stream.
    private let classifiedTokenCount = 32

    /// Prompt tokens a second call appends after the sequence the cache holds.
    private let followUpPrompt = [8, 30, 2]

    /// Tokens a second call generates.
    private let secondCallTokenCount = 8

    /// The top candidates a stream asks the log probabilities of.
    private let logProbabilityCandidateCount = 2

    /// Plain decode and the plain fallback of the MTP iterator run the same forward passes over
    /// the same cache, thus their log probabilities differ by float rounding at most.
    private let logProbabilityTolerance: Float = 1e-4

    /// The prompt as model input.
    private var promptInput: LMInput { LMInput(tokens: MLXArray(prompt.map(Int32.init))) }

    /// A greedy generation of `maxTokens` tokens.
    private func greedyParameters(maxTokens: Int) -> GenerateParameters {
        GenerateParameters(maxTokens: maxTokens, temperature: 0)
    }

    /// The tokens a cache holds, as the iterator infers them.
    private func cachedTokenCount(_ cache: [KVCache]) -> Int {
        KVCacheStorage(cache, plan: .disabled).processedTokenCount
    }

    /// Decodes ``classifiedTokenCount`` greedy MTP tokens after ``prompt`` and names the slot
    /// of each, or `nil` for a token no round produced (the prefill bonus).
    private func classifiedTokens(_ fixture: Qwen35MTPFixture) throws
        -> [(token: Int, slot: Qwen35MTPRoundSlot?)]
    {
        var iterator = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: greedyParameters(maxTokens: classifiedTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        var classified: [(token: Int, slot: Qwen35MTPRoundSlot?)] = []
        var proposedCount = iterator.proposedCount
        var acceptedCount = iterator.acceptedCount
        var followsAcceptedDraft = false
        while let token = iterator.next() {
            let ranRound = iterator.proposedCount > proposedCount
            let acceptedDraft = iterator.acceptedCount > acceptedCount
            proposedCount = iterator.proposedCount
            acceptedCount = iterator.acceptedCount
            let roundSlot: Qwen35MTPRoundSlot = acceptedDraft ? .draft : .correction
            classified.append((token, ranRound ? roundSlot : followsAcceptedDraft ? .bonus : nil))
            followsAcceptedDraft = acceptedDraft
        }
        return classified
    }

    /// The first index of `slot` whose token does not occur earlier in `classified`, so that
    /// a stop on that token stops there.
    private func firstUnrepeatedIndex(
        of slot: Qwen35MTPRoundSlot, in classified: [(token: Int, slot: Qwen35MTPRoundSlot?)]
    ) -> Int? {
        classified.indices.first { index in
            classified[index].slot == slot
                && !classified[..<index].contains { $0.token == classified[index].token }
        }
    }

    @Test(
        "a stop token reports whether the main cache holds it",
        arguments: Qwen35MTPRoundSlot.allCases)
    func stopTokenReportsCacheState(slot: Qwen35MTPRoundSlot) async throws {
        let fixture = try Qwen35MTPFixture()
        let classified = try classifiedTokens(fixture)
        let stopIndex = try #require(firstUnrepeatedIndex(of: slot, in: classified))
        let stopToken = classified[stopIndex].token
        let cache = try fixture.target.newCache(parameters: nil)

        let decode = try await Qwen35MTPFixture.collect(
            try generateTokens(
                input: promptInput, cache: cache,
                parameters: greedyParameters(maxTokens: classifiedTokenCount),
                context: fixture.makeModelContext(eosTokenIds: [stopToken]),
                mtpDrafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize))

        #expect(decode.tokens == classified.prefix(stopIndex).map(\.token))
        #expect(decode.info.stopReason == .stop)
        #expect(decode.info.stopTokenFedToCache == (slot == .draft ? stopToken : nil))
        let ledger =
            decode.info.promptTokenCount + decode.info.generationTokenCount
            + (decode.info.stopTokenFedToCache == nil ? 0 : 1)
        #expect(cachedTokenCount(cache) == ledger)
    }

    /// What a decode in two calls leaves: the whole prompt of the second call, the tokens the
    /// cache held between the calls, the continuation the first call gave back, and the second
    /// call's tokens.
    private struct ContinuedDecode {
        let sequence: [Int]
        let cachedTokenCount: Int
        let continuation: MTPDrafterContinuation?
        let second: Qwen35MTPFixture.GreedyDecode
    }

    /// Reads `stream` until it ends, or until it yielded `limit` tokens.
    private func read(_ stream: AsyncStream<TokenGeneration>, upTo limit: Int?) async {
        var received = 0
        for await event in stream {
            if case .token = event { received += 1 }
            if received == limit { break }
        }
    }

    /// Decodes after ``prompt`` in a first call, then continues the main cache and the drafter
    /// continuation of that call with ``followUpPrompt`` in a second call.
    private func continuedDecode(_ fixture: Qwen35MTPFixture, firstCall: Qwen35MTPFirstCall)
        async throws -> ContinuedDecode
    {
        let reference = try await fixture.decodeGreedily(
            prompt: prompt, maxTokens: firstCall.tokenBudget, speculative: true)
        let context = fixture.makeModelContext()
        let cache = try fixture.target.newCache(parameters: nil)
        let (firstStream, firstTask) = try generateTokensTask(
            input: promptInput, cache: cache,
            parameters: greedyParameters(maxTokens: firstCall.tokenBudget), context: context,
            mtpDrafter: fixture.drafter, blockSize: Qwen35MTPFixture.blockSize)
        await read(firstStream, upTo: firstCall.consumedTokenCount)
        let continuation = await firstTask.value

        let cached = cachedTokenCount(cache)
        let sequence = Array((prompt + reference.tokens).prefix(cached)) + followUpPrompt
        let (secondStream, _) = try generateTokensTask(
            input: LMInput(tokens: MLXArray(sequence[cached...].map(Int32.init))), cache: cache,
            parameters: greedyParameters(maxTokens: secondCallTokenCount), context: context,
            mtpDrafter: fixture.drafter, mtpContinuation: continuation,
            blockSize: Qwen35MTPFixture.blockSize)
        return ContinuedDecode(
            sequence: sequence, cachedTokenCount: cached, continuation: continuation,
            second: try await Qwen35MTPFixture.collect(secondStream))
    }

    @Test(
        "a second call that continues the cache and the drafter matches a cold call and plain decode",
        arguments: Qwen35MTPFirstCall.allCases)
    func continuedCallMatchesColdCall(firstCall: Qwen35MTPFirstCall) async throws {
        let fixture = try Qwen35MTPFixture()
        let continued = try await continuedDecode(fixture, firstCall: firstCall)
        let cold = try await fixture.decodeGreedily(
            prompt: continued.sequence, maxTokens: secondCallTokenCount, speculative: true)
        let plain = try await fixture.decodeGreedily(
            prompt: continued.sequence, maxTokens: secondCallTokenCount, speculative: false)

        let continuation = try #require(continued.continuation)
        #expect(continuation.targetProcessedTokenCount == continued.cachedTokenCount)
        #expect(continued.second.tokens == cold.tokens)
        #expect(continued.second.tokens == plain.tokens)
        #expect(continued.second.info.passthroughReason == nil)
    }

    @Test("the exported state of an MTP stream saves to a prompt cache")
    func exportedStateSavesToPromptCache() throws {
        let fixture = try Qwen35MTPFixture()
        let cache = try fixture.target.newCache(parameters: nil)
        var iterator = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            mainCache: cache, parameters: greedyParameters(maxTokens: secondCallTokenCount),
            blockSize: Qwen35MTPFixture.blockSize)
        while iterator.next() != nil {}

        // The exchange values the target hands the drafter are not arrays, so a prompt cache
        // cannot carry them; the exported state leaves them out.
        #expect(iterator.mainState?[mtpSharedKVStatesKey] != nil)
        let error = #expect(throws: LMOutput.State.SerializationError.self) {
            try preparePromptCacheSave(cache: cache, state: iterator.mainState)
        }
        switch error {
        case .unsupportedValue(let key, _):
            #expect(key.hasPrefix("mtp."), "\(key) is not an MTP exchange value")
        default:
            Issue.record("unexpected serialization error: \(String(describing: error))")
        }
        _ = try preparePromptCacheSave(cache: cache, state: iterator.state)
    }

    /// Reads the log probabilities of each of the `count` tokens `iterator` generates.
    private func logProbabilities<Iterator: TokenIteratorProtocol>(
        of iterator: inout Iterator, count: Int
    ) throws -> [GenerateTokenLogProbabilities] {
        try (0 ..< count).map { _ in
            let token = iterator.next()
            try #require(token != nil)
            return try #require(iterator.lastLogProbabilities).materialize()
        }
    }

    @Test("a stream that asks for log probabilities decodes plainly and reports them")
    func logProbabilitiesDecodePlainly() throws {
        let fixture = try Qwen35MTPFixture()
        var parameters = greedyParameters(maxTokens: secondCallTokenCount)
        parameters.logProbabilities = logProbabilityCandidateCount
        var plain = try TokenIterator(
            input: promptInput, model: fixture.target, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: promptInput, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters, blockSize: Qwen35MTPFixture.blockSize)

        let expected = try logProbabilities(of: &plain, count: secondCallTokenCount)
        let actual = try logProbabilities(of: &speculative, count: secondCallTokenCount)

        #expect(
            speculative.passthroughReason
                == MTPSpeculativeTokenIterator.logProbabilitiesReason)
        #expect(speculative.proposedCount == 0)
        #expect(actual.map(\.chosen.token) == expected.map(\.chosen.token))
        #expect(
            actual.map { $0.topLogProbabilities.map(\.token) }
                == expected.map { $0.topLogProbabilities.map(\.token) })
        for (reported, reference) in zip(actual, expected) {
            #expect(
                abs(reported.chosen.logProbability - reference.chosen.logProbability)
                    <= logProbabilityTolerance)
        }
    }
}

/// The compiled verify schedule of the Qwen 3.5 text target: a verify of two to
/// ``GatedDeltaDecodeKernel/mostTokens`` tokens runs as traced segments, as a decode step does,
/// and gives what the general path gives.
@Suite(.serialized)
struct Qwen35CompiledVerifyTests {

    /// The verify lengths of the compiled schedule: the bonus token and one draft, up to the most
    /// tokens the GDN kernel holds.
    static let verifyTokenCounts = Array(2 ... GatedDeltaDecodeKernel.mostTokens)

    /// The decoder layers of the test model: GDN, attention, GDN, attention. The schedule then
    /// holds each kind of segment: one that opens the step, one between two attention layers,
    /// and one that closes it.
    private static let hiddenLayers = 4

    /// The dtype of the test model: one the GDN kernel reads, so the verify takes snapshots.
    static let dtype = DType.bfloat16

    /// The seed of the weights of the test model.
    private static let weightSeed: UInt64 = 20_261_028

    /// The seed of the prompt tokens.
    private static let promptSeed: UInt64 = 20_261_029

    /// The seed of the verify tokens.
    private static let verifySeed: UInt64 = 20_261_030

    /// The prompt tokens the cache holds before the verify. More than
    /// ``GatedDeltaDecodeKernel/mostTokens``, so the prefill itself takes the general path.
    private static let promptLength = 8

    /// The prompt of each greedy decode of these tests.
    static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens each greedy decode of these tests generates.
    private static let generatedTokenCount = 48

    /// A text target whose GDN layers the kernel runs, on seeded weights, in ``dtype``.
    static func kernelModel() throws -> Qwen35TextModel {
        let configuration = try GDNDecodeKernelFixture.configuration(
            Qwen35TextConfiguration.self, GDNDecodeKernelFixture.smallHeadCase,
            hiddenLayers: Self.hiddenLayers)
        let model = withRandomState(MLXRandom.RandomState(seed: Self.weightSeed)) {
            Qwen35TextModel(configuration)
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(Self.dtype) })
        model.train(false)
        eval(model)
        return model
    }

    /// `count` seeded tokens of the vocabulary of `model`, shape `[count]`.
    private func tokens(_ count: Int, of model: Qwen35TextModel, seed: UInt64) -> MLXArray {
        seededTokens(count, vocabularySize: model.vocabularySize, seed: seed)
    }

    /// Prefills the prompt, runs one verify of `tokenCount` tokens on the compiled or the
    /// general path, then rolls the verify back by `rollback` tokens as a rejected round does.
    private func verify(
        _ model: Qwen35TextModel, tokenCount: Int, rollingBack rollback: Int, compiled: Bool
    ) throws -> CompiledVerifyOutcome {
        let cache = try model.newCache(parameters: nil)
        let prompt = tokens(Self.promptLength, of: model, seed: Self.promptSeed)
        eval(model(LMInput.Text(tokens: prompt[.newAxis]), cache: cache, state: nil).logits)

        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpVerifyKeepsSnapshotsKey] = true
        let output = model.forward(
            LMInput.Text(tokens: tokens(tokenCount, of: model, seed: Self.verifySeed)[.newAxis]),
            cache: cache, state: state, lastTokenOnly: false, useCompiledDecode: compiled)
        return CompiledVerifyOutcome(
            logits: output.logits, hidden: try #require(output.state?[mtpLastHiddenStatesKey]),
            cache: cache, rollingBack: rollback)
    }

    @Test(
        "a compiled verify stays within tolerance of the general path, for each rollback",
        arguments: verifyTokenCounts)
    func compiledVerifyMatchesGeneralPath(tokenCount: Int) throws {
        let model = try Self.kernelModel()
        let tolerance = try #require(AttentionKernelTolerance.of(Self.dtype))

        for rollback in 0 ..< tokenCount {
            let general = try verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: false)
            let compiled = try verify(
                model, tokenCount: tokenCount, rollingBack: rollback, compiled: true)
            try compiled.expectMatches(
                general, tolerance: tolerance, tokenCount: tokenCount, rollback: rollback)
        }
        #expect(model.model.compiledVerifySegmentCount > 0)
    }

    @Test("a verify on the general path compiles no verify segment")
    func generalPathCompilesNoVerifySegment() throws {
        let model = try Self.kernelModel()
        _ = try verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: false)
        #expect(model.model.compiledVerifySegmentCount == 0)
    }

    @Test("invalidating the compiled traces of the model drops its verify segments")
    func invalidationDropsVerifySegments() throws {
        let model = try Self.kernelModel()
        _ = try verify(
            model, tokenCount: CompiledVerifyShape.fewestTokens, rollingBack: 0, compiled: true)
        #expect(model.model.compiledVerifySegmentCount > 0)

        model.invalidateCompiledTraces()

        #expect(model.model.compiledVerifySegmentCount == 0)
    }

    @Test("a verify shape holds two to the most kernel tokens")
    func verifyShapeBounds() {
        let most = GatedDeltaDecodeKernel.mostTokens
        #expect(CompiledVerifyShape(tokens: CompiledVerifyShape.fewestTokens) != nil)
        #expect(CompiledVerifyShape(tokens: most) != nil)
        #expect(CompiledVerifyShape(tokens: GatedDeltaDecodeKernel.decodeTokens) == nil)
        #expect(CompiledVerifyShape(tokens: most + 1) == nil)
    }

    @Test("snapshot slots keep their arrays through the trace arrays")
    func speculativeStatesRoundTripThroughTraceArrays() throws {
        let headCase = GDNDecodeKernelFixture.smallHeadCase
        let recurrent = MLXRandom.normal(
            [headCase.batch, headCase.valueHeads, headCase.valueHeadDim, headCase.keyHeadDim])
        let conv = MLXRandom.normal([headCase.batch, GDNDecodeKernelFixture.convTaps - 1, 1])
        let slots = stacked([recurrent, recurrent + 1])
        let window = concatenated([conv, conv], axis: 1)

        let snapshots = GatedDeltaDecodeKernel.Snapshots(
            traceArrays: GatedDeltaDecodeKernel.Snapshots(recurrent: slots, convWindow: window)
                .traceArrays)

        #expect(arrayEqual(snapshots.recurrent, slots).item(Bool.self))
        #expect(arrayEqual(snapshots.convWindow, window).item(Bool.self))
    }

    @Test(
        "greedy MTP decode of the text target equals plain decode through the compiled verify",
        arguments: Qwen35MTPFixture.DrafterMode.allCases)
    func textTargetMTPEqualsPlainDecode(mode: Qwen35MTPFixture.DrafterMode) async throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)

        let plain = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: false)
        let speculative = try await fixture.decodeGreedily(
            prompt: Self.prompt, maxTokens: Self.generatedTokenCount, speculative: true)

        #expect(speculative.tokens == plain.tokens)
        #expect((speculative.info.proposedDraftTokens ?? 0) > 0)
        #expect(fixture.target.model.compiledVerifySegmentCount > 0)
    }

    @Test("greedy MTP decode of the VLM target equals plain decode")
    func visionLanguageTargetMTPEqualsPlainDecode() throws {
        try Self.expectGreedyMTPEqualsPlainDecode(on: Qwen35VLMMTPFixture())
    }

    /// Decodes greedily with the target of `fixture`, then with MTP over the same target, and
    /// holds that the two give the same tokens and that MTP speculates.
    ///
    /// - Parameter fixture: The target and its drafter.
    /// - Throws: The error of an iterator.
    static func expectGreedyMTPEqualsPlainDecode(on fixture: Qwen35VLMMTPFixture) throws {
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let parameters = GenerateParameters(maxTokens: generatedTokenCount, temperature: 0)

        var plain = try TokenIterator(input: input, model: fixture.target, parameters: parameters)
        var speculative = try MTPSpeculativeTokenIterator(
            input: input, mainModel: fixture.target, drafter: fixture.drafter,
            parameters: parameters, blockSize: Qwen35MTPFixture.blockSize)
        var plainTokens: [Int] = []
        while let token = plain.next() {
            plainTokens.append(token)
        }
        var speculativeTokens: [Int] = []
        while let token = speculative.next() {
            speculativeTokens.append(token)
        }

        #expect(plainTokens.count == generatedTokenCount)
        #expect(speculativeTokens == plainTokens)
        #expect(speculative.passthroughReason == nil)
        #expect(speculative.proposedCount > 0)
    }
}
