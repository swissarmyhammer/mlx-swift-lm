// Copyright © 2025 Apple Inc.

import Foundation
import MLX
@_spi(Testing) import MLXLLM
import MLXLMTestSupport
import MLXNN
import MLXOptimizers
import MLXScriptedLM
import XCTest

@testable import MLXLMCommon

final class ChatSessionSpeculativeDecodingTests: XCTestCase, ChatSessionFixtures {
    func testSpeculativeDecodingMemoryPolicyFallbackUsesDefaultGeneration() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
        XCTAssertEqual(completionInfo.speculativeDecodingFallbackReason, .memoryBudgetExceeded)
    }

    func testSpeculativeDecodingMemoryPolicyFallbackReusesMainCacheAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault)),
            generateParameters: GenerateParameters(maxTokens: 3, temperature: 0))

        _ = try await session.respond(to: "first")
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        XCTAssertEqual(
            completionInfo?.promptTokenCount,
            secondRenderedLength - firstRenderedLength - 3)
        XCTAssertNil(completionInfo?.speculativeDecodingTelemetry)
        XCTAssertEqual(completionInfo?.speculativeDecodingFallbackReason, .memoryBudgetExceeded)
    }

    func testActiveSpeculativeDecodingSafelyRebuildsAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesPrefixOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let container = ModelContainer(context: model(processor: processor))
        let speculativeDecoding = SpeculativeDecodingConfig(
            draftModel: container,
            numDraftTokens: 2)
        let parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        let session = ChatSession(
            container,
            speculativeDecoding: speculativeDecoding,
            generateParameters: parameters)

        let first = try await collectGeneration(session.streamDetails(to: "first"))
        _ = await lengthIterator.next()
        let firstTelemetry = try XCTUnwrap(first.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(firstTelemetry.roundCount, 0)
        XCTAssertGreaterThan(firstTelemetry.draftTokenCount, 0)
        XCTAssertEqual(firstTelemetry.emittedTokenCount, first.info.generationTokenCount)

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        let secondTelemetry = try XCTUnwrap(second.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(secondTelemetry.roundCount, 0)
        XCTAssertGreaterThan(secondTelemetry.draftTokenCount, 0)
        XCTAssertEqual(secondTelemetry.emittedTokenCount, second.info.generationTokenCount)
        XCTAssertEqual(second.info.promptTokenCount, secondRenderedLength)

        let cleanSession = ChatSession(
            container,
            history: [.user("first"), .assistant(first.text)],
            speculativeDecoding: speculativeDecoding,
            generateParameters: parameters)
        let cleanSecond = try await collectGeneration(
            cleanSession.streamDetails(to: "second"))
        let cleanRenderedLengthValue = await lengthIterator.next()
        let cleanRenderedLength = try XCTUnwrap(cleanRenderedLengthValue)

        let cleanTelemetry = try XCTUnwrap(cleanSecond.info.speculativeDecodingTelemetry)
        XCTAssertGreaterThan(cleanTelemetry.draftTokenCount, 0)
        XCTAssertEqual(cleanTelemetry.emittedTokenCount, cleanSecond.info.generationTokenCount)
        XCTAssertEqual(cleanSecond.info.promptTokenCount, cleanRenderedLength)
        XCTAssertEqual(second.text, cleanSecond.text)
    }

    func testActiveSpeculativeDecodingReusesAlignedStorageAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        // Fixed weights make the accept and reject pattern of the two
        // different models the same at each run.
        let (mainContext, draftContext) = withRandomState(
            MLXRandom.RandomState(seed: Self.speculativeWeightSeed)
        ) {
            (model(processor: processor), model())
        }
        let session = ChatSession(
            mainContext,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: draftContext),
                numDraftTokens: Self.speculativeDraftTokenCount),
            generateParameters: Self.speculativeTurnParameters)

        _ = try await assertSpeculativeStorageIsReusedAcrossTurns(
            session, renderedLengths: renderedLengths)
    }

    /// A draft model that is the main model accepts each draft at temperature 0.
    /// The first turn (3 tokens, 2 drafts in each round) thus stops after the
    /// loop took all the tokens of a fully accepted round. That round leaves
    /// the draft cache one token behind the main cache until the next round,
    /// and there is no next round. The main and the draft caches must
    /// nevertheless end the turn aligned, and the second turn must reuse them.
    func testFullyAcceptedSpeculativeRoundLeavesAlignedStorageAcrossTurns() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let container = withRandomState(MLXRandom.RandomState(seed: Self.speculativeWeightSeed)) {
            ModelContainer(context: model(processor: processor))
        }
        let session = ChatSession(
            container,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: container,
                numDraftTokens: Self.speculativeDraftTokenCount),
            generateParameters: Self.speculativeTurnParameters)

        let firstInfo = try await assertSpeculativeStorageIsReusedAcrossTurns(
            session, renderedLengths: renderedLengths)

        let firstTelemetry = try XCTUnwrap(firstInfo.speculativeDecodingTelemetry)
        XCTAssertEqual(firstTelemetry.roundCount, 1)
        XCTAssertEqual(firstTelemetry.acceptedDraftTokenCount, firstTelemetry.draftTokenCount)
        XCTAssertEqual(firstInfo.generationTokenCount, Self.speculativeTurnParameters.maxTokens)
    }

    /// The seed of the random weights of the speculative decoding tests.
    private static let speculativeWeightSeed: UInt64 = 20_260_925

    /// The count of drafts in each round of the speculative decoding tests.
    private static let speculativeDraftTokenCount = 2

    /// The parameters of each turn of the speculative decoding tests: one
    /// fully accepted round of ``speculativeDraftTokenCount`` drafts gives
    /// all the tokens of the turn.
    private static let speculativeTurnParameters = GenerateParameters(
        maxTokens: speculativeDraftTokenCount + 1, temperature: 0)

    /// Runs two turns ("first" and "second") on `session` and asserts that the
    /// first turn leaves the main and the draft caches aligned, and that the
    /// second turn prefills only the tokens that the caches do not hold.
    ///
    /// - Parameters:
    ///   - session: a session with active speculative decoding and a
    ///     ``PrefixPreservingTokenizer`` that sends each rendered length.
    ///   - renderedLengths: the stream of rendered lengths from that tokenizer.
    /// - Returns: the completion information of the first turn.
    private func assertSpeculativeStorageIsReusedAcrossTurns(
        _ session: ChatSession,
        renderedLengths: AsyncStream<Int>
    ) async throws -> GenerateCompletionInfo {
        var lengthIterator = renderedLengths.makeAsyncIterator()

        let first = try await collectGeneration(session.streamDetails(to: "first"))
        let firstRenderedLengthValue = await lengthIterator.next()
        _ = try XCTUnwrap(firstRenderedLengthValue)
        let optionalFirstProgress = await session.cacheProgress()
        let firstProgress = try XCTUnwrap(optionalFirstProgress)
        XCTAssertEqual(firstProgress.main, firstProgress.draft)

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        XCTAssertNotNil(second.info.speculativeDecodingTelemetry)
        XCTAssertEqual(
            second.info.promptTokenCount,
            secondRenderedLength - firstProgress.main)
        return first.info
    }

    func testSpeculativeDecodingMemoryPolicyFailThrows() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFallbackDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
        XCTAssertEqual(completionInfo.speculativeDecodingFallbackReason, .memoryBudgetExceeded)
    }

    func testSpeculativeDecodingFallsBackForPreparedMedia() async throws {
        var context = model()
        context.processor = PreparedImageInputProcessor()
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
        XCTAssertEqual(completionInfo.speculativeDecodingFallbackReason, .unsupportedMedia)
    }

    func testSpeculativeDecodingFallsBackForHistoricalMedia() async throws {
        var context = model()
        context.processor = MediaAwareInputProcessor(tokenizer: context.tokenizer)
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(draftModelBytes: 0) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0))

        let first = try await collectGeneration(
            session.streamDetails(
                to: "inspect this", images: [.array(MLXArray([Float(0)]))]))
        let second = try await collectGeneration(session.streamDetails(to: "describe it"))

        for info in [first.info, second.info] {
            XCTAssertNil(info.speculativeDecodingTelemetry)
            XCTAssertEqual(info.speculativeDecodingFallbackReason, .unsupportedMedia)
        }
    }

    /// Carried model state does not disqualify a turn from speculation. The
    /// anchors these models carry position from the cache offset, and a
    /// rejected proposal rewinds that offset along with the KV rows.
    func testSpeculativeDecodingRunsWithCarriedModelState() async throws {
        let context = model()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let cache = try context.model.newCache(parameters: parameters)
        let stateKey = LMOutput.Key<MLXArray>("test.carriedState")
        var state = LMOutput.State()
        state[stateKey] = MLXArray([Int32(1)])
        let session = ChatSession(
            context,
            cache: cache,
            state: state,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: parameters
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNotNil(completionInfo.speculativeDecodingTelemetry)
    }

    /// Speculation refused on one turn must not disable it for the rest of the session.
    /// The media turn drops the draft cache; the next text turn extends the cached prefix, which
    /// sets `reusedMainCacheWithoutDraft` so both caches are rebuilt from the full input and
    /// speculation resumes. Without that, the gate stays false forever once the main cache is
    /// warm and the draft cache is nil.
    func testSpeculativeDecodingResumesAfterAMediaTurn() async throws {
        // The prefix-preserving tokenizer makes turn 2 render as an exact extension of turn 1, so
        // the session takes the suffix-reuse path and keeps a warm main cache with a valid ledger.
        // That is what sets `reusedMainCacheWithoutDraft`; with an ordinary tokenizer turn 2
        // rebuilds from cold instead and never exercises this path.
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var context = model()
        context.processor = FirstTurnImageInputProcessor(
            base: TestInputProcessor(
                tokenizer: PrefixPreservingTokenizer(renderedLengthContinuation: continuation),
                configuration: ModelConfiguration(id: "test"),
                messageGenerator: DefaultMessageGenerator()))
        _ = renderedLengths
        let session = ChatSession(
            context,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: model()),
                numDraftTokens: 2
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        func infoForTurn(_ prompt: String) async throws -> GenerateCompletionInfo {
            var info: GenerateCompletionInfo?
            for try await generation in session.streamDetails(
                to: prompt,
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {
                if let generationInfo = generation.info {
                    info = generationInfo
                }
            }
            return try XCTUnwrap(info)
        }

        let mediaTurn = try await infoForTurn("describe this")
        XCTAssertNil(
            mediaTurn.speculativeDecodingTelemetry,
            "a media turn cannot speculate — the draft would have to prefill it")
        XCTAssertEqual(mediaTurn.speculativeDecodingFallbackReason, .unsupportedMedia)

        let textTurn = try await infoForTurn("and now in one word")
        XCTAssertNotNil(
            textTurn.speculativeDecodingTelemetry,
            "speculation must resume once a text turn can rebuild both caches from the transcript")
        XCTAssertNil(textTurn.speculativeDecodingFallbackReason)
    }

    func testSpeculativeDecodingFallsBackForPrebuiltCacheWithoutDraftCache() async throws {
        let context = model()
        let parameters = GenerateParameters(maxTokens: 4, temperature: 0.0)
        let cache = try context.model.newCache(parameters: parameters)
        let input = try await context.processor.prepare(
            input: UserInput(chat: [.user("cached prefix")]))
        _ = try TokenIterator(
            input: input,
            model: context.model,
            cache: cache,
            parameters: parameters)
        XCTAssertTrue(cache.contains { $0.offset > 0 })

        let session = ChatSession(
            context,
            cache: cache,
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: parameters
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
        XCTAssertEqual(completionInfo.speculativeDecodingFallbackReason, .unavailableContinuation)
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFailDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch is UnexpectedDraftModelLoadError {
            XCTFail("draft model loader should not be called")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingLoadsDraftModelOnceAcrossTurns() async throws {
        let loadCounter = DraftModelLoadCounter()
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                await loadCounter.increment()
                return ModelContainer(context: Self.makeModel())
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        _ = try await session.respond(to: "hello")
        _ = try await session.respond(to: "again")

        let loadCount = await loadCounter.value
        XCTAssertEqual(loadCount, 1)
    }
}
