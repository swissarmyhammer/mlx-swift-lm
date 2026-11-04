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

final class ChatSessionCacheReuseTests: XCTestCase, ChatSessionFixtures {
    func testStructuredContinuationRendersCompleteTranscriptAcrossToolTurns() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: PseudoWordTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let history: [Chat.Message] = (0 ..< 8).flatMap { index in
            [
                .user("question \(index)"),
                .assistant("answer \(index)"),
            ]
        }
        let continuations: [[Chat.Message]] = [
            [.tool("first tool result")],
            [.tool("second tool result")],
            [.user("final answer")],
        ]
        let session = ChatSession(
            model(processor: processor),
            history: history,
            generateParameters: GenerateParameters(maxTokens: 1))

        for messages in continuations {
            _ = try await session.respond(to: messages)
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(
            calls.map(\.count), [history.count + 1, history.count + 3, history.count + 5])
        XCTAssertEqual(calls[0].map(\.role), history.map(\.role) + [.tool])
        XCTAssertEqual(
            calls[1].map(\.role),
            history.map(\.role) + [.tool, .assistant, .tool])
        XCTAssertEqual(
            calls[2].map(\.role),
            history.map(\.role) + [.tool, .assistant, .tool, .assistant, .user])
    }

    func testLegacyStringContinuationPrefillsOnlyUncachedSuffix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let info = try XCTUnwrap(completionInfo)
        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertLessThan(info.promptTokenCount, fullSecondPromptLength)
        XCTAssertEqual(
            info.promptTokenCount,
            fullSecondPromptLength - firstPromptLength - 3)
    }

    func testStructuredContinuationPrefillsOnlyUncachedSuffix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: [.user("first")])
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: [.user("second")]) {
            if let info = item.info {
                completionInfo = info
            }
        }

        let info = try XCTUnwrap(completionInfo)
        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertLessThan(info.promptTokenCount, fullSecondPromptLength)
        XCTAssertEqual(
            info.promptTokenCount,
            fullSecondPromptLength - firstPromptLength - 3)
    }

    func testCompletionInfoAttributesReusedCachePrefix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        let first = try await collectGeneration(session.streamDetails(to: "first"))
        let firstRenderedLength = await lengthIterator.next()
        let firstPromptLength = try XCTUnwrap(firstRenderedLength)

        XCTAssertEqual(first.info.cachedPromptTokenCount, 0)
        XCTAssertEqual(first.info.totalPromptTokenCount, firstPromptLength)
        XCTAssertEqual(first.info.cacheEfficiency, 0)

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)

        // The reused prefix is the first prompt plus the tokens it generated.
        XCTAssertEqual(second.info.cachedPromptTokenCount, firstPromptLength + 3)
        XCTAssertEqual(second.info.totalPromptTokenCount, fullSecondPromptLength)
        XCTAssertGreaterThan(second.info.cacheEfficiency, 0)
    }

    func testExactPrefixReuseRebuildsWhenPreparedInputHasMask() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MaskedInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        _ = await lengthIterator.next()

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        XCTAssertEqual(second.info.promptTokenCount, secondRenderedLength)
    }

    func testContinuationRebuildsCacheWhenTemplateRewritesPrefix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesPrefixOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let secondRenderedLength = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(secondRenderedLength)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
    }

    func testContinuationTrimsCacheToLongestCommonPrefix() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(to: "first")
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "second") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        let expectedCommonPrefixLength = firstRenderedLength - 1
        XCTAssertEqual(
            completionInfo?.promptTokenCount,
            fullSecondPromptLength - expectedCommonPrefixLength)
        XCTAssertLessThan(completionInfo?.promptTokenCount ?? .max, fullSecondPromptLength)
    }

    /// A template that re-renders the cached tail rewinds the cache before the
    /// turn is prefilled, so the prefix announced up front is what remains: the
    /// figure the turn's completion info attributes to the cache once it ends.
    func testPrefillPrefixAnnouncesTheReusedCache() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: DefaultMessageGenerator())
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))
        let (prefixes, prefixContinuation) = AsyncStream<Int>.makeStream()
        var prefixIterator = prefixes.makeAsyncIterator()
        session.generateParameters.prefill.prefix = { prefixContinuation.yield($0) }

        let first = try await collectGeneration(session.streamDetails(to: "first"))
        let firstPrefix = await prefixIterator.next()
        let firstRenderedLength = await lengthIterator.next()

        let second = try await collectGeneration(session.streamDetails(to: "second"))
        let secondPrefix = await prefixIterator.next()

        XCTAssertEqual(firstPrefix, 0)
        XCTAssertEqual(firstPrefix, first.info.cachedPromptTokenCount)
        XCTAssertEqual(secondPrefix, try XCTUnwrap(firstRenderedLength) - 1)
        XCTAssertEqual(secondPrefix, second.info.cachedPromptTokenCount)
    }

    func testHistoricalMediaReusesSuffixButNewMediaRebuildsCache() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        let firstRenderedLengthValue = await lengthIterator.next()
        let firstRenderedLength = try XCTUnwrap(firstRenderedLengthValue)

        var textFollowUpInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                textFollowUpInfo = info
            }
        }
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)
        XCTAssertEqual(
            textFollowUpInfo?.promptTokenCount,
            secondRenderedLength - firstRenderedLength - 3)

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)
        XCTAssertEqual(newMediaInfo?.promptTokenCount, thirdRenderedLength)
        XCTAssertNil(newMediaInfo?.speculativeDecodingFallbackReason)
    }

    /// Returning *a* suffix is not enough. A conformer that returns the wrong tokens
    /// must be treated exactly like one that declines: the ledger advances to
    /// `representedTokens` on the strength of the requested boundary, so accepting
    /// any other suffix would leave the record describing a cache never built.
    func testAppendOnlyMediaRebuildsWhenModelSplitsAtTheWrongBoundary() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)

        var context = Self.makeModel(
            processor: processor,
            configuration: processor.configuration,
            tokenizer: processor.tokenizer)
        context.model = MisSplittingLanguageModel(context.model, extraDrop: 2)

        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        for try await _ in session.streamDetails(to: "describe it") {}
        _ = await lengthIterator.next()

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)

        XCTAssertEqual(
            newMediaInfo?.promptTokenCount, thirdRenderedLength,
            "a wrong-boundary split must downgrade to a full rebuild")
    }

    func testLongestCommonPrefixTrimmingFallsBackForHistoricalMedia() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                completionInfo = info
            }
        }

        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
    }

    /// The append-only counterpart of
    /// `testHistoricalMediaReusesSuffixButNewMediaRebuildsCache`: same transcript,
    /// same turns, but a model that can split its own prepared input. The new-image
    /// turn now prefills only the uncached suffix instead of the whole prompt.
    func testAppendOnlyMediaReusesCacheWhenModelSplitsPreparedInput() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        for try await _ in session.streamDetails(to: "describe it") {}
        let secondRenderedLengthValue = await lengthIterator.next()
        let secondRenderedLength = try XCTUnwrap(secondRenderedLengthValue)

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)

        XCTAssertEqual(
            newMediaInfo?.promptTokenCount,
            thirdRenderedLength - secondRenderedLength - 3,
            "an append-only media turn should prefill only the uncached suffix")
        // The split boundary is what the turn did not prefill, so it is also what
        // the completion info must attribute to the cache: the previous prompt
        // plus the tokens it generated.
        XCTAssertEqual(
            newMediaInfo?.cachedPromptTokenCount,
            secondRenderedLength + 3,
            "the reused prefix should be attributed to the cache, not dropped")
        XCTAssertEqual(newMediaInfo?.totalPromptTokenCount, thirdRenderedLength)
    }

    /// A model that declines the split keeps the pre-existing behavior, which is
    /// also what every non-conforming model gets.
    func testAppendOnlyMediaRebuildsWhenModelDeclinesToSplit() async throws {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(renderedLengthContinuation: continuation)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true, declinesToSplit: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        for try await _ in session.streamDetails(to: "describe it") {}
        _ = await lengthIterator.next()

        var newMediaInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(
            to: "now inspect this",
            role: .user,
            images: [.array(MLXArray([Float(1)]))],
            videos: [])
        {
            if let info = item.info {
                newMediaInfo = info
            }
        }
        let thirdRenderedLengthValue = await lengthIterator.next()
        let thirdRenderedLength = try XCTUnwrap(thirdRenderedLengthValue)

        XCTAssertEqual(newMediaInfo?.promptTokenCount, thirdRenderedLength)
    }

    /// Splitting is offered only where the transcript is *extended*. When the
    /// template rewrites an already-cached tail the turn needs a rewind, and media
    /// still forces a rebuild there -- the guard PR #472 added is untouched.
    func testLongestCommonPrefixTrimmingStillFallsBackForMediaWhenModelCanSplit()
        async throws
    {
        let (renderedLengths, continuation) = AsyncStream<Int>.makeStream()
        var lengthIterator = renderedLengths.makeAsyncIterator()
        let tokenizer = PrefixPreservingTokenizer(
            renderedLengthContinuation: continuation,
            rewritesCachedTailOnContinuation: true)
        let processor = MediaAwareInputProcessor(tokenizer: tokenizer)
        let session = ChatSession(
            model(processor: processor, splitting: true),
            generateParameters: GenerateParameters(maxTokens: 3))

        _ = try await session.respond(
            to: "inspect this",
            image: .array(MLXArray([Float(0)])))
        _ = await lengthIterator.next()

        var completionInfo: GenerateCompletionInfo?
        for try await item in session.streamDetails(to: "describe it") {
            if let info = item.info {
                completionInfo = info
            }
        }
        let fullSecondPromptLengthValue = await lengthIterator.next()
        let fullSecondPromptLength = try XCTUnwrap(fullSecondPromptLengthValue)
        XCTAssertEqual(completionInfo?.promptTokenCount, fullSecondPromptLength)
    }
}
