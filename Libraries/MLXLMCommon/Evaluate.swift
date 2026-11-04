// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Given prompt tokens generate text using the given model and parameters.
///
/// ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - promptTokens: tokenized prompt
///   - parameters: generation parameters
///   - model: model to evaluate
///   - tokenizer: tokenizer to convert tokens back into strings and recognize special tokens
///   - extraEOSTokens: any additional stop tokens
///   - didGenerate: visitor for the tokens as they are generated
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:state:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    promptTokens: [Int], parameters: GenerateParameters, model: any LanguageModel,
    tokenizer: Tokenizer,
    extraEOSTokens: Set<String>? = nil,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let tokens = MLXArray(promptTokens)
    let iterator = try TokenIterator(
        prompt: tokens, model: model, parameters: parameters)

    // this is a compatibility cover -- create the required values
    // for the iteration
    let input = LMInput(tokens: tokens)
    let configuration = ModelConfiguration(id: "stand-in", extraEOSTokens: extraEOSTokens ?? [])
    let context = ModelContext(
        configuration: configuration, model: model, processor: StandInUserInputProcessor(),
        tokenizer: tokenizer)

    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Builds the `TokenIterator` shared by the deprecated `generate(input:parameters:context:didGenerate:)`
/// overloads below -- both wrap this same construction before delegating to their
/// respective low-level `generate(input:context:iterator:didGenerate:)`.
private func makeDeprecatedTokenIterator(
    input: LMInput, parameters: GenerateParameters, context: ModelContext
) throws -> TokenIterator {
    try TokenIterator(input: input, model: context.model, parameters: parameters)
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:state:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try makeDeprecatedTokenIterator(
        input: input, parameters: parameters, context: context)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:state:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    let result = runSynchronousGeneration(input: input, context: context, iterator: iterator) {
        _, generatedTokens in
        didGenerate(generatedTokens)
    }

    return GenerateResult(
        inputText: input.text, tokenIDs: result.generatedTokenIds,
        output: context.tokenizer.decode(tokenIds: result.generatedTokenIds),
        promptTime: result.info.promptTime,
        generateTime: result.info.generateTime
    )
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// Prefer using ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>` instead.
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:state:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    let iterator = try makeDeprecatedTokenIterator(
        input: input, parameters: parameters, context: context)
    return generate(
        input: input, context: context, iterator: iterator,
        didGenerate: didGenerate)
}

/// Low-level token generation using a ``TokenIterator``.
///
/// ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)`` returning `AsyncStream<Generation>` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
@available(
    *, deprecated,
    message:
        "Use the AsyncStream-based generate(input:cache:state:parameters:context:) instead for better Swift concurrency support"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    runSynchronousGeneration(input: input, context: context, iterator: iterator) { token, _ in
        didGenerate(token)
    }.info
}

/// Runs `iterator` over the prompt of `input` and the model of `context`, on the calling thread.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: visits each generated token with the tokens so far
/// - Returns: the generated tokens and the completion information
private func runSynchronousGeneration(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (_ token: Int, _ generatedTokenIds: [Int]) -> GenerateDisposition
) -> SynchronousGenerationLoopResult {
    SynchronousGenerationLoop.run(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        didGenerate: didGenerate)
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent text chunks, tool calls,
/// or summary completion information.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  This is typically OK for
/// one-shot calls, but for "chat session" type calls consider using
/// ``generateTask(promptTokenCount:modelConfiguration:tokenizer:iterator:wiredMemoryTicket:tools:toolCallPolicy:)``
/// so that the end of the generation task can be observed.
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
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination across
///     concurrent tasks. This is opt-in and only applied on GPU devices that support wired
///     memory control (macOS 15 / iOS 18 / tvOS 18 or newer).
///   - tools: Optional tool schemas used to parse tool-call arguments and authorize function names.
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated text chunks (`.chunk`),
///   accepted tool calls (`.toolCall`), rejected tool-call attempts (`.rejectedToolCall`), and
///   completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let context: ModelContext
///
/// let lmInput = try context.processor.prepare(input: input)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: generateParameters, context: context)
///
/// // Process the stream asynchronously to handle text chunks and completion info.
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     case .rejectedToolCall(let rejection):
///         print("Rejected tool call: \(rejection.reason)")
///     }
/// }
/// ```
public func generate(
    input: LMInput, cache: [KVCache]? = nil, state: LMOutput.State? = nil,
    parameters: GenerateParameters, context: ModelContext,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) throws -> AsyncStream<Generation> {
    let iterator = try TokenIterator(
        input: input, model: context.model, cache: cache, state: state,
        parameters: parameters, components: components)
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        tools: tools,
        toolCallPolicy: parameters.toolCallPolicy)
    return stream
}

/// Generates text and tool calls asynchronously using speculative decoding with a draft model.
///
/// This function uses a smaller draft model to propose tokens that are verified in batch
/// by the main model, potentially accelerating generation. The resulting stream yields
/// decoded text chunks, tool calls, and completion information. It has the same output as the
/// non-speculative ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)``.
///
/// Both models must share the same tokenizer.
///
/// ### Example Usage:
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: UserInput
/// let mainContext: ModelContext
/// let draftModel: LanguageModel
///
/// let lmInput = try mainContext.processor.prepare(input: input)
///
/// let stream = try generate(
///     input: lmInput, parameters: generateParameters,
///     context: mainContext, draftModel: draftModel)
///
/// for await generation in stream {
///     switch generation {
///     case .chunk(let text):
///         print("Generated text: \(text)")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     case .toolCall(let call):
///         print("Tool call: \(call.function.name)")
///     case .rejectedToolCall(let rejection):
///         print("Rejected tool call: \(rejection.reason)")
///     }
/// }
/// ```
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
///   - tools: Optional tool schemas used to parse arguments and authorize function names.
/// - Returns: An `AsyncStream` that emits text, accepted or rejected tool calls, and completion
///   information as `Generation` values.
/// - Throws: An error if the iterator initialization fails.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    draftModel: any LanguageModel,
    draftCache: [KVCache]? = nil,
    numDraftTokens: Int = 2,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) throws -> AsyncStream<Generation> {
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
    return generateTextStream(
        iterator: iterator, input: input, parameters: parameters, context: context,
        wiredMemoryTicket: wiredMemoryTicket, tools: tools,
        tokenCollector: IgnoringGeneratedTokens()
    ).0
}

/// Runs `iterator` through the text/tool-call handler of `context`: the wiring of the
/// `AsyncStream<Generation>` entry points that differ only in their iterator and collector.
///
/// - Parameters:
///   - tools: optional tool schemas used to parse tool-call arguments.
///   - tokenCollector: what the task answers when the generation ends.
///   - The other parameters are those of
///     ``generateStream(iterator:input:context:wiredMemoryTicket:tokenCollector:handler:)``.
/// - Returns: the stream, and the task that generates it.
private func generateTextStream<Collector: GeneratedTokenCollector>(
    iterator: consuming any TokenIteratorProtocol,
    input: LMInput,
    parameters: GenerateParameters,
    context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket?,
    tools: [[String: any Sendable]]?,
    tokenCollector: Collector
) -> (AsyncStream<Generation>, Task<Collector.Result, Never>) {
    generateStream(
        iterator: iterator, input: input, context: context,
        wiredMemoryTicket: wiredMemoryTicket, tokenCollector: tokenCollector,
        handler: textToolHandler(
            tokenizer: context.tokenizer, modelConfiguration: context.configuration,
            tools: tools, toolCallPolicy: parameters.toolCallPolicy))
}

/// Runs `iterator` for `input` through `handler`, the stream wiring shared by the entry points
/// that build an iterator from an ``LMInput`` and a ``ModelContext``: the text/tool-call
/// streams (through `generateTextStream`) and the raw-token speculative streams (through
/// `generateRawTokenStream`). Each constructs its own iterator and collector; the handler
/// decides the output.
///
/// - Parameters:
///   - tokenCollector: what the task answers when the generation ends.
///   - handler: turns the tokens into the output of the stream.
/// - Returns: the stream, and the task that generates it.
func generateStream<Handler: TokenLoopHandler, Collector: GeneratedTokenCollector>(
    iterator: consuming any TokenIteratorProtocol,
    input: LMInput,
    context: ModelContext,
    wiredMemoryTicket: WiredMemoryTicket?,
    tokenCollector: Collector,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<Collector.Result, Never>) {
    generateLoopTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: tokenCollector,
        handler: handler
    )
}

/// Deprecated low-level entry point: wraps an existing `TokenIterator` in the same
/// text/tool-call `AsyncStream<Generation>` pipeline as
/// `generate(input:cache:parameters:context:wiredMemoryTicket:tools:)`, without
/// exposing the underlying `Task` handle.
@available(
    *, deprecated,
    message: "use a higher level generate() call or use generateTask() for fine grained control"
)
public func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil
) -> AsyncStream<Generation> {
    let (stream, _) = generateTask(
        promptTokenCount: input.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket)
    return stream
}

/// Low-level token generation returning an `AsyncStream<Generation>` and a `Task`.
///
/// Accepts any ``TokenIteratorProtocol`` conformer, including ``TokenIterator`` and
/// ``SpeculativeTokenIterator``. Swift infers the concrete type at the call site.
///
/// * Important: if the stream is terminated early (e.g. break from the loop) computation will continue
/// using the model, parameters, KVCache, etc. for some time (typically a few ms).  Callers can await
/// the `task` to observe when the use of the parameters is complete.
///
/// - Parameters:
///   - promptTokenCount: number of tokens in the prompt
///   - modelConfiguration: model configuration (for EOS/extra EOS tokens and tool-call format)
///   - tokenizer: tokenizer (for EOS id, unknown token id, and detokenization)
///   - iterator: a token iterator conforming to ``TokenIteratorProtocol``
///   - wiredMemoryTicket: Optional wired memory ticket for policy-based coordination.
///   - tools: Optional tool schemas used to parse tool-call arguments into their declared types.
///   - toolCallPolicy: Recovery and validation rules for generated tool calls.
/// - Returns: An `AsyncStream` that emits `Generation` values and a `Task`
public func generateTask<Iterator: TokenIteratorProtocol>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming Iterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil,
    toolCallPolicy: ToolCallPolicy = .init()
) -> (AsyncStream<Generation>, Task<Void, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: textToolHandler(
            tokenizer: tokenizer, modelConfiguration: modelConfiguration, tools: tools,
            toolCallPolicy: toolCallPolicy)
    )
}

/// Receives what the prefill of a generation left, before the generation reads its first
/// token: the model state, and a reader of the drafter continuation at the end of the prompt.
///
/// The reader gives `nil` when the generation does not speculate with a resumable drafter. It
/// copies the drafter caches, thus a caller calls it only when it keeps a prompt cache
/// checkpoint at the end of the prompt, and only during the callback.
package typealias PreparedGenerationState =
    (LMOutput.State?, _ promptBoundaryContinuation: () -> MTPDrafterContinuation?) -> Void

/// Builds the token iterator of one token-recording generation and hands `preparedState` what
/// its prefill left.
///
/// - Parameters:
///   - input: the input for the language model.
///   - cache: optional ``KVCache``.
///   - state: optional model state saved with `cache`.
///   - parameters: the configuration options for token generation.
///   - context: the model context, including the model and its tokenizer.
///   - speculation: the MTP drafter the generation speculates with, or `nil` for plain decode.
///   - mtpContinuation: the drafter continuation saved with `cache`, or `nil`.
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``.
///   - preparedState: receives the model state the prefill left and the reader of the drafter
///     continuation at the end of the prompt (see ``PreparedGenerationState``).
/// - Returns: the iterator, after its prefill and before its first token is read.
/// - Throws: an error when the iterator cannot be built.
func makePreparedIterator(
    input: LMInput, cache: [KVCache]?, state: LMOutput.State?,
    parameters: GenerateParameters, context: ModelContext,
    speculation: MTPSpeculation?, mtpContinuation: MTPDrafterContinuation?,
    components: GenerationComponents, preparedState: PreparedGenerationState
) throws -> any TokenIteratorProtocol {
    let iterator = try MTPSpeculation.makeTokenIterator(
        speculation: speculation, input: input, cache: cache, state: state,
        mtpContinuation: mtpContinuation, parameters: parameters, model: context.model,
        components: components)
    preparedState(iterator.state) { MTPSpeculation.promptBoundaryContinuation(of: iterator) }
    return iterator
}

/// Package variant of
/// ``generate(input:cache:state:parameters:context:components:wiredMemoryTicket:tools:)``
/// that also answers the tokens the generation fed into `cache`.
///
/// The `MLXFoundationModels` executor keeps a token ledger of the prompt cache
/// each session carries, the same ledger ``ChatSession`` keeps, thus it needs
/// the tokens the generation sampled and not the decoded text alone.
///
/// - Parameters:
///   - input: the input for the language model.
///   - cache: optional ``KVCache``.
///   - state: optional model state saved with `cache`.
///   - parameters: the configuration options for token generation.
///   - context: the model context, including the model and its tokenizer.
///   - speculation: the MTP drafter the generation speculates with, or `nil`
///     for plain decode. A drafter that cannot draft for the model decodes
///     plainly too (``MTPSpeculation/makeTokenIterator(speculation:input:cache:state:mtpContinuation:parameters:model:components:)``).
///   - mtpContinuation: the drafter continuation saved with `cache`, or `nil`.
///     The generation resumes it when it is aligned with `cache` and made by the
///     drafter of `speculation`, and else starts the drafter at the cached
///     position.
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``.
///   - wiredMemoryTicket: optional wired memory ticket.
///   - tools: optional tool schemas used to parse tool-call arguments.
///   - preparedState: receives what the prefill left (see
///     ``PreparedGenerationState``). A caller that carries `cache` to a later
///     turn seeds that turn with the model state; a Qwen VL model keys its
///     M-RoPE anchor there and refuses a warm cache without it.
///   - drafterContinuation: receives, when the generation ends, the drafter
///     continuation aligned with what `cache` then holds, or `nil` when the
///     generation did not speculate with a resumable drafter.
/// - Returns: the stream of text and tool calls, and the task that generates
///   them. The value of the task is every generated token, in order.
/// - Throws: an error when the iterator cannot be built.
package func generateTaskRecordingTokens(
    input: LMInput, cache: [KVCache]? = nil, state: LMOutput.State? = nil,
    parameters: GenerateParameters, context: ModelContext,
    speculation: MTPSpeculation? = nil,
    mtpContinuation: MTPDrafterContinuation? = nil,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil,
    preparedState: PreparedGenerationState = { _, _ in },
    drafterContinuation: @escaping @Sendable (MTPDrafterContinuation?) -> Void = { _ in }
) throws -> (AsyncStream<Generation>, Task<[Int], Never>) {
    let iterator = try makePreparedIterator(
        input: input, cache: cache, state: state, parameters: parameters, context: context,
        speculation: speculation, mtpContinuation: mtpContinuation, components: components,
        preparedState: preparedState)
    return generateTextStream(
        iterator: iterator, input: input, parameters: parameters, context: context,
        wiredMemoryTicket: wiredMemoryTicket, tools: tools,
        tokenCollector: RecordingGeneratedTokenList(drafterContinuation: drafterContinuation))
}

/// Internal variant used by `ChatSession` to keep its token-prefix record in
/// lockstep with the KV cache and receive the iterator's finalized
/// continuation state.
struct RecordedGenerationResult {
    let generatedTokens: [Int]
    let state: LMOutput.State?
    let mtpDrafterContinuation: MTPDrafterContinuation?
}

/// Records the token ledger with the text/tool-call handler, through the composed-handler
/// overload below.
func generateTaskRecordingTokens<Iterator: TokenIteratorProtocol>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming Iterator,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil,
    toolCallPolicy: ToolCallPolicy = .init()
) -> (AsyncStream<Generation>, Task<SendableBox<RecordedGenerationResult>, Never>) {
    generateTaskRecordingTokens(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: textToolHandler(
            tokenizer: tokenizer, modelConfiguration: modelConfiguration, tools: tools,
            toolCallPolicy: toolCallPolicy))
}

/// Records the same token ledger with a composed output handler.
func generateTaskRecordingTokens<Handler: TokenLoopHandler>(
    promptTokenCount: Int,
    modelConfiguration: ModelConfiguration,
    tokenizer: Tokenizer,
    iterator: consuming any TokenIteratorProtocol,
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    handler: consuming Handler
) -> (AsyncStream<Handler.Output>, Task<SendableBox<RecordedGenerationResult>, Never>) {
    generateLoopTask(
        promptTokenCount: promptTokenCount,
        modelConfiguration: modelConfiguration,
        tokenizer: tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        tokenCollector: RecordingGeneratedTokens(),
        handler: handler)
}

/// Generates tokens asynchronously using MTP speculative decoding.
///
/// Parallel to ``generate(input:cache:state:parameters:context:draftModel:draftCache:numDraftTokens:components:wiredMemoryTicket:tools:)``
/// but for MTP drafters: the drafter shares K/V with the target model and
/// produces a block of `blockSize - 1` candidate tokens per round in a
/// single `draftBlock(...)` call. The drafter shares the target's
/// tokenizer (via `context.tokenizer`).
///
/// To continue `cache` in a later call with the drafter state of this one, call
/// ``generateTask(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:tools:)``,
/// whose task gives back the ``MTPDrafterContinuation``.
///
/// - Parameters:
///   - input: language model input for the main (verifier) model.
///   - cache: optional ``KVCache`` for the main model.
///   - state: optional model state saved with `cache`, as
///     ``PromptCacheSnapshot/state``. Pass it whenever `cache` came from a
///     snapshot: models that position from a carried anchor need it to place
///     later tokens correctly.
///   - parameters: generation parameters (sampling, max tokens, KV
///     quantization, etc.).
///   - context: model context for the main (verifier) model.
///   - mtpDrafter: the ``MTPDrafterModel``. The target is threaded through
///     ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)``
///     per round; drafter instances hold no target-derived state and are safe
///     to share across iterators. Speculative rounds are staged and committed
///     rather than written and rewound, so a sliding-window model speculates
///     for the whole stream rather than only while total context stays inside
///     the window. If the target ever stops emitting drafter state — a KV
///     cache that quantizes mid-stream, say — the iterator logs once, reports
///     ``GenerateCompletionInfo/passthroughReason``, and finishes the stream
///     with ordinary single-token generation.
///   - mtpContinuation: the drafter continuation that the generation which left
///     `cache` gave back, or `nil` to start the drafter at the cached position.
///   - blockSize: total tokens per round (up to `blockSize - 1` drafted plus the
///     bonus from the previous verify). Mirrors mlx-vlm's
///     `draft_block_size`. Default 4 matches mlx-vlm's example configs. A drafter
///     caps it at its ``MTPDrafterModel/maximumBlockSize``.
///   - minimumDraftProbability: the drafter top-1 probability below which a
///     chaining drafter (Qwen MTP) stops a round's chain early, as llama.cpp's
///     `--spec-draft-p-min`; 0 drafts the whole block every round.
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
///   - wiredMemoryTicket: optional wired memory ticket.
///   - tools: Optional tool schemas used to parse arguments and authorize function names.
/// - Returns: an `AsyncStream<Generation>` yielding chunks and tool calls.
/// - Throws: an error if the iterator initialization fails.
public func generate(
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
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) throws -> AsyncStream<Generation> {
    try generateTask(
        input: input, cache: cache, state: state, parameters: parameters, context: context,
        mtpDrafter: mtpDrafter, mtpContinuation: mtpContinuation, blockSize: blockSize,
        minimumDraftProbability: minimumDraftProbability, components: components,
        wiredMemoryTicket: wiredMemoryTicket, tools: tools
    ).0
}

/// Generates text and tool calls with MTP speculative decoding, and returns the
/// stream plus the `Task` that generates it.
///
/// The same generation as
/// ``generate(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:tools:)``.
/// The value of the task is the ``MTPDrafterContinuation`` the drafter left at
/// the end of the generation, aligned with what `cache` then holds. Hand it to
/// the next call that continues `cache`, so that call resumes the drafter
/// rather than starting it again at the cached position. The value is `nil`
/// when the drafter keeps no state across calls, or when the generation stopped
/// speculating.
///
/// - Parameters: the parameters of
///   ``generate(input:cache:state:parameters:context:mtpDrafter:mtpContinuation:blockSize:minimumDraftProbability:components:wiredMemoryTicket:tools:)``.
/// - Returns: the stream of text and tool calls, and the task that generates it.
/// - Throws: an error if the iterator initialization fails.
public func generateTask(
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
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) throws -> (AsyncStream<Generation>, Task<MTPDrafterContinuation?, Never>) {
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
    return generateTextStream(
        iterator: iterator, input: input, parameters: parameters, context: context,
        wiredMemoryTicket: wiredMemoryTicket, tools: tools,
        tokenCollector: MTPContinuationCollector())
}
