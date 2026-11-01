// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// What a synchronous token loop produced.
struct SynchronousGenerationLoopResult {
    /// The generated tokens, without the stop token.
    let generatedTokenIds: [Int]
    /// The completion information of the loop.
    let info: GenerateCompletionInfo
}

/// Builds the set of token IDs that end a generation.
///
/// Pulls from three sources, all required for chat-tuned models to stop correctly:
///
/// 1. `modelConfiguration.eosTokenIds`, loaded from `config.json` and
///    `generation_config.json` at model-load time. Chat models like Gemma 3 ship
///    `eos_token_id` as an array (e.g. `[1, 106]` for `<eos>` + `<end_of_turn>`); this source is
///    the only way to pick up the turn-ender when the tokenizer's primary EOS is the completion
///    EOS.
/// 2. `tokenizer.eosTokenId`, the tokenizer's single primary EOS.
/// 3. `modelConfiguration.extraEOSTokens`, token strings from registry entries (e.g.
///    `["<end_of_turn>"]` on some Gemma variants). Callers needing extra stop tokens add them
///    here, not as a per-call argument.
///
/// - Parameters:
///   - modelConfiguration: The configuration of the model.
///   - tokenizer: The tokenizer of the model.
/// - Returns: The stop token IDs.
package func buildStopTokenIds(
    modelConfiguration: ModelConfiguration,
    tokenizer: any Tokenizer
) -> Set<Int> {
    var stopTokenIds = modelConfiguration.eosTokenIds
    if let tokenizerEOS = tokenizer.eosTokenId {
        stopTokenIds.insert(tokenizerEOS)
    }
    for token in modelConfiguration.extraEOSTokens {
        if let id = tokenizer.convertTokenToId(token) {
            stopTokenIds.insert(id)
        }
    }
    return stopTokenIds
}

/// Whether `token` ends a generation.
private func isStopToken(_ token: Int, tokenizer: Tokenizer, stopTokenIds: Set<Int>) -> Bool {
    token == tokenizer.unknownTokenId || stopTokenIds.contains(token)
}

/// The next token of `iterator`, with the autoreleased temporaries of the step drained.
private func nextPooledToken<Iterator: TokenIteratorProtocol>(
    _ iterator: inout Iterator
) -> Int? {
    autoreleasepool(invoking: { iterator.next() })
}

/// Why a token loop ended when it recorded no reason itself: the token limit, or else a
/// cancellation.
///
/// - Parameters:
///   - isAtTokenLimit: Whether the iterator generated every token its limit allows.
///   - isCancelled: Whether the task of the loop is cancelled.
/// - Returns: The stop reason.
private func inferredStopReason(isAtTokenLimit: Bool, isCancelled: Bool) -> GenerateStopReason {
    !isCancelled && isAtTokenLimit ? .length : .cancelled
}

/// The generated tokens outside the reasoning span, when the reasoning span is counted.
private func answerTokenCount(generated: Int, reasoning: Int?) -> Int? {
    reasoning.map { generated - $0 }
}

/// The prompt and generation times of a token loop.
private struct GenerationTiming {
    private var start = Date.timeIntervalSinceReferenceDate

    /// The time to the first token, zero until the first token arrives.
    private(set) var promptTime: TimeInterval = 0

    /// The time since the first token, or since the loop started before it.
    var generateTime: TimeInterval { Date.timeIntervalSinceReferenceDate - start }

    /// Ends the prompt phase at the first token. Later tokens change nothing.
    mutating func recordToken() {
        guard promptTime == 0 else { return }
        let now = Date.timeIntervalSinceReferenceDate
        promptTime = now - start
        start = now
    }
}

/// Turns tokens into nothing and applies the disposition of a synchronous token visitor.
private struct SynchronousTokenLoopHandler: TokenLoopHandler {
    /// Visits a generated token.
    let didGenerate: (Int) -> GenerateDisposition

    mutating func onToken(
        _ token: Int,
        logProbabilities _: DeferredTokenLogProbabilities?,
        emit _: (sending GenerateCompletionInfo) -> Bool
    ) -> TokenLoopDisposition {
        didGenerate(token) == .stop ? .cancelled : .more
    }

    mutating func onStopToken(
        _: Int,
        logProbabilities _: DeferredTokenLogProbabilities?,
        emit _: (sending GenerateCompletionInfo) -> Bool
    ) -> TokenLoopDisposition { .more }

    mutating func onGenerationEnd(
        emit _: (sending GenerateCompletionInfo) -> Bool
    ) -> TokenLoopDisposition { .more }

    func infoEvent(_ info: GenerateCompletionInfo) -> GenerateCompletionInfo { info }
}

/// The token loop of ``generateLoopTask``, run on the calling thread.
enum SynchronousGenerationLoop {

    /// Runs a token iterator on the calling thread, through the loop of ``generateLoopTask``.
    ///
    /// The loop does not read the cancellation of the calling task: `didGenerate` alone stops it
    /// early.
    ///
    /// - Parameters:
    ///   - promptTokenCount: the number of tokens in the prompt.
    ///   - modelConfiguration: the configuration of the model, for its stop tokens.
    ///   - tokenizer: the tokenizer of the model.
    ///   - iterator: the iterator to run.
    ///   - didGenerate: visits each generated token with the tokens so far.
    /// - Returns: the generated tokens and the completion information.
    static func run(
        promptTokenCount: Int,
        modelConfiguration: ModelConfiguration,
        tokenizer: Tokenizer,
        iterator: any TokenIteratorProtocol,
        didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
    ) -> SynchronousGenerationLoopResult {
        var generatedTokenIds: [Int] = []
        let info = withoutActuallyEscaping(didGenerate) { didGenerate in
            runTokenLoop(
                iterator: iterator,
                handler: SynchronousTokenLoopHandler { token in
                    generatedTokenIds.append(token)
                    return didGenerate(token, generatedTokenIds)
                },
                tokenCollector: IgnoringGeneratedTokens(),
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                includeStopToken: false,
                isCancelled: { false },
                emit: { _ in true }
            ).info
        }
        return SynchronousGenerationLoopResult(generatedTokenIds: generatedTokenIds, info: info)
    }
}

protocol GeneratedTokenCollector: Sendable {
    associatedtype Result: Sendable

    mutating func record(_ token: Int)
    consuming func result(iterator: any TokenIteratorProtocol) -> Result
}

extension GeneratedTokenCollector {

    /// The drafter continuation an MTP iterator left at the end of its generation.
    ///
    /// - Parameter iterator: the iterator of the generation.
    /// - Returns: the continuation, or `nil` when `iterator` is not an MTP iterator or left none.
    fileprivate static func drafterContinuation(
        leftBy iterator: any TokenIteratorProtocol
    ) -> MTPDrafterContinuation? {
        (iterator as? any MTPDrafterContinuationProviding)?.mtpDrafterContinuation
    }
}

struct IgnoringGeneratedTokens: GeneratedTokenCollector {
    mutating func record(_ token: Int) {}
    consuming func result(iterator _: any TokenIteratorProtocol) {}
}

/// Records the generated tokens, and hands the drafter continuation an MTP iterator left to a
/// callback, for callers that keep a token ledger and take the drafter state apart from it.
struct RecordingGeneratedTokenList: GeneratedTokenCollector {
    private var tokens: [Int] = []

    /// Receives the drafter continuation when the generation ends, or `nil` when the iterator
    /// left none.
    private let receiveDrafterContinuation: @Sendable (MTPDrafterContinuation?) -> Void

    /// Creates a collector that hands the drafter continuation to `drafterContinuation`.
    ///
    /// - Parameter drafterContinuation: receives the continuation; the default drops it.
    init(drafterContinuation: @escaping @Sendable (MTPDrafterContinuation?) -> Void = { _ in }) {
        self.receiveDrafterContinuation = drafterContinuation
    }

    mutating func record(_ token: Int) {
        tokens.append(token)
    }

    consuming func result(iterator: any TokenIteratorProtocol) -> [Int] {
        receiveDrafterContinuation(Self.drafterContinuation(leftBy: iterator))
        return tokens
    }
}

struct RecordingGeneratedTokens: GeneratedTokenCollector {
    private var tokens: [Int] = []

    mutating func record(_ token: Int) {
        tokens.append(token)
    }

    consuming func result(
        iterator: any TokenIteratorProtocol
    ) -> SendableBox<RecordedGenerationResult> {
        SendableBox(
            RecordedGenerationResult(
                generatedTokens: tokens,
                state: iterator.state,
                mtpDrafterContinuation: Self.drafterContinuation(leftBy: iterator)))
    }
}

/// Records no token, and answers the drafter continuation an MTP iterator left,
/// for the callers that continue the cache with the same drafter.
struct MTPContinuationCollector: GeneratedTokenCollector {
    mutating func record(_ token: Int) {}

    consuming func result(iterator: any TokenIteratorProtocol) -> MTPDrafterContinuation? {
        Self.drafterContinuation(leftBy: iterator)
    }
}

/// Runs a token iterator with a custom output handler.
///
/// The task owns the iterator and handler until generation finishes. Await its
/// value after cancelling or stopping consumption to wait for MLX cleanup.
public func generateLoopTask<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        tokenCollector: IgnoringGeneratedTokens(),
        handler: handler)
}

func generateLoopTask<
    Handler: TokenLoopHandler, Collector: GeneratedTokenCollector
>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    includeStopToken: Bool = false,
    tokenCollector: consuming Collector,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Collector.Result, Never>) {
    let (stream, continuation) = AsyncStream<Handler.Output>.makeStream()

    let iterator = SendableBox(iterator)
    let handler = SendableBox(handler)
    let tokenCollector = consume tokenCollector

    // Keep task cancellation and task-local values while moving blocking MLX work
    // onto a separate queue for each generation.
    let task = Task {
        let worker = GenerationWorker()
        let performIteration = { @Sendable in
            let (result, _) = runTokenLoop(
                iterator: iterator.consume(),
                handler: handler.consume(),
                tokenCollector: tokenCollector,
                promptTokenCount: promptTokenCount,
                modelConfiguration: modelConfiguration,
                tokenizer: tokenizer,
                includeStopToken: includeStopToken,
                isCancelled: { Task.isCancelled },
                emit: {
                    if case .terminated = continuation.yield($0) { return false }
                    return true
                })

            // Finalize the stream
            continuation.finish()

            return result
        }

        if let ticket = wiredMemoryTicket {
            return await WiredMemoryTicket.withWiredLimit(ticket) {
                await worker.run(performIteration)
            }
        } else {
            return await worker.run(performIteration)
        }
    }

    // When the consumer cancels (or ends) the stream, cancel our underlying task.
    continuation.onTermination = { termination in
        if case .cancelled = termination {
            task.cancel()
        }
    }

    return (stream, task)
}

/// Runs one token iterator to its end through `handler`, on the calling thread.
///
/// The paths through ``generateLoopTask`` and ``SynchronousGenerationLoop/run`` share this loop: it
/// stops on a stop token, a handler disposition, the token limit or a cancellation, records the
/// reasoning count, measures the phases, finalizes the iterator and synchronizes with the stream.
/// Guided generation runs loops of its own.
///
/// - Parameters:
///   - iterator: the iterator to run.
///   - handler: turns the tokens into output.
///   - tokenCollector: records what the caller answers when the generation ends.
///   - promptTokenCount: the number of tokens in the prompt.
///   - modelConfiguration: the configuration of the model, for its stop tokens.
///   - tokenizer: the tokenizer of the model.
///   - includeStopToken: whether the stop token counts as output.
///   - isCancelled: whether the generation was cancelled, read before each token.
///   - emit: receives each output of `handler`, the info event last. It returns `false` when the
///     consumer has ended the stream.
/// - Returns: the result of `tokenCollector`, and the completion information.
private func runTokenLoop<Handler: TokenLoopHandler, Collector: GeneratedTokenCollector>(
    iterator: consuming any TokenIteratorProtocol,
    handler: consuming Handler,
    tokenCollector: consuming Collector,
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    includeStopToken: Bool,
    isCancelled: () -> Bool,
    emit: (sending Handler.Output) -> Bool
) -> (result: Collector.Result, info: GenerateCompletionInfo) {
    var iterator = consume iterator
    var handler = consume handler
    var tokenCollector = consume tokenCollector

    var timing = GenerationTiming()
    var tokenCount = 0
    var stopReason: GenerateStopReason?
    // The terminal stop token whose forward pass already advanced the
    // cache but which is discarded (not emitted) below when
    // `includeStopToken == false`. Surfaced in the final `.info` so a
    // prompt-cache consumer can reconcile the resulting one-token
    // offset gap -- see `GenerateCompletionInfo.stopTokenFedToCache`.
    var stopTokenFedToCache: Int?
    // The terminal stop token that stays out of the output, whether or
    // not a forward pass fed it -- see
    // `GenerateCompletionInfo.discardedStopToken`.
    var discardedStopToken: Int?
    let tracksReasoningTokenCount = iterator.reasoningTokenCount != nil

    var stopTokenIds = buildStopTokenIds(
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer
    )
    stopTokenIds.formUnion(handler.additionalStopTokenIDs)

    // Dispatches a newly generated (non-stop) token through the handler,
    // counting it toward the loop's token budget.
    func handleGeneratedToken(_ token: Int) -> TokenLoopDisposition {
        tokenCount += 1
        if tracksReasoningTokenCount {
            iterator.recordEmittedToken(token)
        }
        return handler.onToken(
            token,
            logProbabilities: iterator.lastLogProbabilities,
            emit: emit)
    }

    // Check cancellation BEFORE iterator.next(): next() calls asyncEval() to
    // pipeline the next GPU evaluation, so checking after it (the previous
    // `while let token = iterator.next()` form) allowed one extra asyncEval to be
    // submitted post-cancellation, which faults if the app has backgrounded
    // (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted). The
    // post-loop block below assigns `.cancelled`; Stream.defaultStream.synchronize() still
    // settles any in-flight evaluation at the end of the task body.
    tokenLoop: while !isCancelled() {
        guard let token = nextPooledToken(&iterator) else { break }
        tokenCollector.record(token)

        timing.recordToken()

        // Check for end-of-sequence tokens. When `includeStopToken` is
        // false the stop token stays out of the emitted output and out
        // of `tokenCount`, whether the handler receives it or not. When
        // its forward pass already advanced the cache, its identity
        // goes into `stopTokenFedToCache` for the caller that keeps a
        // token ledger beside the cache. A speculative iterator can
        // return a stop token that no forward pass has fed.
        if isStopToken(token, tokenizer: tokenizer, stopTokenIds: stopTokenIds) {
            if !includeStopToken {
                discardedStopToken = token
                if iterator.hasCachedLastToken {
                    stopTokenFedToCache = token
                }
            }
            if includeStopToken || handler.receivesStopTokens {
                if includeStopToken {
                    tokenCount += 1
                    if tracksReasoningTokenCount {
                        iterator.recordEmittedToken(token)
                    }
                } else {
                    iterator.discardGeneratedToken()
                }
                let disposition = handler.onStopToken(
                    token,
                    logProbabilities: includeStopToken
                        ? iterator.lastLogProbabilities : nil,
                    emit: emit)
                if let reason = disposition.stopReason {
                    stopReason = reason
                    break tokenLoop
                }
            } else {
                iterator.discardGeneratedToken()
            }
            stopReason = .stop
            break
        }

        if let reason = handleGeneratedToken(token).stopReason {
            stopReason = reason
            break tokenLoop
        }
    }

    if stopReason == nil {
        stopReason = inferredStopReason(
            isAtTokenLimit: iterator.isAtTokenLimit, isCancelled: isCancelled())
    }

    // Speculative iterators verify several candidates at once. A stop
    // token, consumer termination, or token limit can leave verified
    // but unreturned candidates in their shared caches. Remove that
    // lookahead before ChatSession reconciles its token ledger.
    if var finalizing = iterator as? any GenerationFinalizingTokenIterator {
        finalizing.finalizeGeneration()
        // Write back: the cast copies the iterator, and the trim also
        // updates state held inline by the iterator (not just the
        // reference-typed caches) that later reads still observe.
        iterator = finalizing
    }

    switch handler.onGenerationEnd(emit: emit) {
    case .more:
        break
    case .stop:
        if stopReason != .cancelled { stopReason = .stop }
    case .cancelled:
        stopReason = .cancelled
    }

    let generateTime = timing.generateTime

    let mtpStats = iterator as? MTPStatsCollecting
    let speculationFallback = (iterator as? TokenIterator)?.speculationFallback
    let reasoningTokenCount = iterator.reasoningTokenCount.map {
        clampedCount($0, to: tokenCount)
    }
    let info = GenerateCompletionInfo(
        promptTokenCount: promptTokenCount,
        generationTokenCount: tokenCount,
        evictedTokenCount: iterator.evictedTokenCount,
        reasoningTokenCount: reasoningTokenCount,
        answerTokenCount: answerTokenCount(
            generated: tokenCount, reasoning: reasoningTokenCount),
        promptTime: timing.promptTime + iterator.promptPrefillTime,
        generationTime: generateTime,
        stopReason: stopReason ?? .cancelled,
        stopTokenFedToCache: stopTokenFedToCache,
        discardedStopToken: discardedStopToken,
        proposedDraftTokens: mtpStats?.proposedDraftTokens,
        acceptedDraftTokens: mtpStats?.acceptedDraftTokens,
        passthroughReason: mtpStats?.passthroughReason ?? speculationFallback?.statedReason,
        speculativeDecodingTelemetry: iterator.speculativeDecodingTelemetry,
        speculativeDecodingFallbackReason: speculationFallback?.reason
    )
    _ = emit(handler.infoEvent(info))

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down. Synchronize with the stream
    // to make sure it is complete.
    Stream.defaultStream.synchronize()

    return (tokenCollector.result(iterator: iterator), info)
}
