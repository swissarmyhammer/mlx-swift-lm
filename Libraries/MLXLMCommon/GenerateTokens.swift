// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Generates raw token IDs asynchronously using the provided language model input, parameters, and context.
///
/// This is similar to `generate(input:cache:state:parameters:context:)`, but yields raw token IDs instead of decoded text/tool calls.
/// This is useful for downstream parsers that need access to token IDs directly.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - state: optional model state saved with `cache`, as
///     ``PromptCacheSnapshot/state``. Pass it whenever `cache` came from a
///     snapshot: models that position from a carried anchor need it to place
///     later tokens correctly.
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, state: state,
        parameters: parameters, components: components)
    let (stream, _) = generateTokenTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        includeStopToken: includeStopToken,
        wiredMemoryTicket: wiredMemoryTicket
    )
    return stream
}

/// Generates raw token IDs asynchronously using speculative decoding with a draft model.
///
/// This is similar to `generate(input:cache:state:parameters:context:draftModel:draftCache:numDraftTokens:components:wiredMemoryTicket:)`,
/// but yields raw token IDs instead of decoded text/tool calls.
///
/// Both models must share the same tokenizer.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache`` for the main model.
///   - state: optional model state saved with `cache`, as
///     ``PromptCacheSnapshot/state``. Pass it whenever `cache` came from a
///     snapshot: models that position from a carried anchor need it to place
///     later tokens correctly.
///   - parameters: The configuration options for token generation.
///   - context: The model context for the main (verifier) model.
///   - draftModel: The draft ``LanguageModel`` for speculative token proposals.
///   - draftCache: optional ``KVCache`` for the draft model.
///   - numDraftTokens: Number of tokens the draft model proposes per round (default: 2).
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values.
/// - Throws: An error if the iterator initialization fails.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    let iterator = try SpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        draftModel: draftModel,
        mainCache: cache,
        draftCache: draftCache,
        mainState: state,
        parameters: parameters,
        numDraftTokens: numDraftTokens,
        components: components
    )
    return generateRawTokenStream(
        iterator: iterator, input: input, context: context, wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: IgnoringGeneratedTokens()
    ).0
}

/// Runs a speculative-decoding iterator through a ``RawTokenLoopHandler``: the wiring of the
/// raw-token speculative entry points, which differ only in their iterator and collector.
///
/// - Parameter tokenCollector: what the task answers when the generation ends.
/// - Returns: the stream, and the task that generates it.
private func generateRawTokenStream<Collector: GeneratedTokenCollector>(
    iterator: consuming any TokenIteratorProtocol,
    input: LMInput,
    context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket?,
    tokenCollector: Collector
) -> (AsyncStream<TokenGeneration>, Task<Collector.Result, Never>) {
    generateStream(
        iterator: iterator, input: input, context: context, wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: tokenCollector, handler: RawTokenLoopHandler())
}

/// Generates raw token IDs asynchronously using MTP speculative decoding.
///
/// Parallels
/// ``generateTokens(input:cache:state:parameters:context:draftModel:draftCache:numDraftTokens:components:wiredMemoryTicket:)``
/// but for MTP drafters. Yields raw token IDs instead of decoded text or
/// tool calls.
///
/// Speculative rounds are staged and committed rather than written and rewound,
/// so a sliding-window model speculates for the whole stream rather than only
/// while total context stays inside the window. If the target ever stops
/// emitting drafter state — a KV cache that quantizes mid-stream, say — the
/// iterator logs once, reports
/// ``GenerateCompletionInfo/passthroughReason`` on the emitted `.info` event,
/// and finishes the stream with ordinary single-token generation.
///
/// To continue `cache` in a later call with the drafter state of this one, call
/// ``generateTokensTask(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:)``,
/// whose task gives back the ``MTPDrafterContinuation``.
///
/// - Parameters: the parameters of
///   ``generate(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:tools:)``,
///   without the tool schemas.
/// - Returns: an `AsyncStream<TokenGeneration>` yielding token IDs and a final `.info`.
/// - Throws: an error if the iterator initialization fails.
public func generateTokens(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    mtpDrafter: any MTPDrafterModel,
    mtpContinuation: MTPDrafterContinuation? = nil,
    blockSize: Int = 4,
    minimumDraftProbability: Float = SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> AsyncStream<TokenGeneration> {
    try generateTokensTask(
        input: input, cache: cache, state: state, parameters: parameters, context: context,
        mtpDrafter: mtpDrafter, mtpContinuation: mtpContinuation, blockSize: blockSize,
        minimumDraftProbability: minimumDraftProbability, components: components,
        wiredMemoryTicket: wiredMemoryTicket
    ).0
}

/// Generates raw token IDs with MTP speculative decoding, and returns the
/// stream plus the `Task` that generates it.
///
/// The same generation as
/// ``generateTokens(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:)``.
/// The value of the task is the ``MTPDrafterContinuation`` the drafter left at
/// the end of the generation, as
/// ``generateTask(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:tools:)``
/// describes it.
///
/// - Parameters: the parameters of
///   ``generateTokens(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:)``.
/// - Returns: the stream of token IDs, and the task that generates it.
/// - Throws: an error if the iterator initialization fails.
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    mtpDrafter: any MTPDrafterModel,
    mtpContinuation: MTPDrafterContinuation? = nil,
    blockSize: Int = 4,
    minimumDraftProbability: Float = SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<MTPDrafterContinuation?, Never>) {
    let iterator = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: context.model,
        drafter: mtpDrafter,
        mainCache: cache,
        mainState: state,
        mtpContinuation: mtpContinuation,
        parameters: parameters,
        blockSize: blockSize,
        minimumDraftProbability: minimumDraftProbability,
        components: components
    )
    return generateRawTokenStream(
        iterator: iterator, input: input, context: context, wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: MTPContinuationCollector())
}

/// Generates raw token IDs asynchronously and returns the stream plus a `Task`.
///
/// Prefer this overload if you want to be able to observe when the underlying generation work is finished
/// (especially if the consumer terminates the stream early).
///
/// - Returns: An `AsyncStream` that emits `TokenGeneration` values and a `Task`.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - state: optional model state saved with `cache`, as
///     ``PromptCacheSnapshot/state``. Pass it whenever `cache` came from a
///     snapshot: models that position from a carried anchor need it to place
///     later tokens correctly.
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
public func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    includeStopToken: Bool = false,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    try generateTokensTask(
        input: input, cache: cache, state: state, parameters: parameters, context: context,
        speculation: nil, includeStopToken: includeStopToken, components: components,
        wiredMemoryTicket: wiredMemoryTicket)
}

/// Package variant of
/// ``generateTokensTask(input:cache:state:parameters:context:includeStopToken:components:wiredMemoryTicket:)``
/// that speculates with an MTP drafter when the drafter can draft for the model.
///
/// - Parameters:
///   - speculation: the MTP drafter the generation speculates with, or `nil`
///     for plain decode, as
///     ``MTPSpeculation/makeTokenIterator(speculation:input:cache:state:mtpContinuation:parameters:model:components:)``
///     takes it.
///   - The other parameters are those of the public variant.
/// - Returns: the stream of token IDs, and the task that generates it.
/// - Throws: an error when the iterator cannot be built.
package func generateTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    speculation: MTPSpeculation?,
    includeStopToken: Bool = false,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil
) throws -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    let iterator = try MTPSpeculation.makeTokenIterator(
        speculation: speculation, input: input, cache: cache, state: state,
        parameters: parameters, model: context.model, components: components)
    return generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler()
    )
}

/// Package-only raw generation for framed response protocols.
///
/// The decoder remains owned by the caller; this function only applies its
/// semantic stop-token policy to the generic generation loop.
///
/// The task answers every token the iterator generated, in order, which is
/// more than the stream yields: a suppressed stop token stays out of the
/// stream and its forward pass has already advanced the cache. A caller that
/// keeps a token ledger of that cache needs the whole list.
///
///   - speculation: the MTP drafter the generation speculates with, or `nil`
///     for plain decode, as
///     ``generateTaskRecordingTokens(input:cache:state:parameters:context:speculation:mtpContinuation:components:wiredMemoryTicket:tools:preparedState:drafterContinuation:)``
///     takes it.
///   - mtpContinuation: the drafter continuation saved with `cache`, or `nil`,
///     as that function takes it.
///   - preparedState: receives what the prefill left, as that function hands it
///     over.
///   - drafterContinuation: receives, when the generation ends, the drafter
///     continuation aligned with what `cache` then holds, or `nil`.
/// - Returns: the stream of raw tokens, and the task that generates them. The
///   value of the task is every generated token.
package func generateProtocolTokensTask(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    decoder: (any TokenStreamDecoder)?,
    speculation: MTPSpeculation? = nil,
    mtpContinuation: MTPDrafterContinuation? = nil,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    preparedState: PreparedGenerationState = { _, _ in },
    drafterContinuation: @escaping @Sendable (MTPDrafterContinuation?) -> Void = { _ in }
) throws -> (AsyncStream<TokenGeneration>, Task<[Int], Never>) {
    let iterator = try makePreparedIterator(
        input: input, cache: cache, state: state, parameters: parameters, context: context,
        speculation: speculation, mtpContinuation: mtpContinuation, components: components,
        preparedState: preparedState)
    return generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: RecordingGeneratedTokenList(drafterContinuation: drafterContinuation),
        handler: RawTokenLoopHandler(
            additionalStopTokenIDs: decoder?.additionalStopTokenIDs ?? [],
            receivesStopTokens: decoder?.receivesStopTokens ?? false)
    )
}

/// Low-level raw token generation using a `TokenIterator`, returning an
/// `AsyncStream<TokenGeneration>` and a `Task`.
///
/// This is useful for parsers that need access to token IDs directly, without
/// detokenization or tool-call parsing.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens)
///   - tokenizer: tokenizer (for EOS id and unknown token id)
///   - iterator: token iterator
///   - includeStopToken: when true, the terminating EOS/unknown token is yielded before finishing
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
/// - Returns: An `AsyncStream` that emits token IDs and a final `.info`, plus a `Task`.
public func generateTokenTask(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming TokenIterator,
    includeStopToken: Bool = false,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> (AsyncStream<TokenGeneration>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        includeStopToken: includeStopToken,
        handler: RawTokenLoopHandler()
    )
}
