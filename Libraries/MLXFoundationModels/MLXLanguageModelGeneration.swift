// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
// `_version: 2` gates on the FoundationModels *framework* major version, which
// is 1.4.x on the macOS/iOS 26 SDK and 2.0.x on 27. The third-party-model
// surface this adapter uses (`LanguageModel`, `LanguageModelCapabilities`, the
// generic `LanguageModelSession(model:)` init) only exists on the 27 SDK, so
// this excludes the whole adapter from older SDKs where those symbols are
// absent. A plain `canImport(FoundationModels)` is insufficient — the module
// also ships in 26 — and `@available` cannot help, since it gates runtime
// availability, not the compile-time presence of a symbol in the SDK.
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import MLX
import MLXGuidedGeneration

// MARK: - Generation paths

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel.Executor {
    /// The identifier of one transcript entry of a turn. Each entry kind has
    /// a type of its own, thus a response, a reasoning and a tool-calls entry
    /// identifier cannot be swapped without a compile error.
    protocol EntryID: Sendable {
        /// - Parameter rawValue: The identifier the channel events carry.
        init(rawValue: String)
    }

    /// The transcript entry that the response text, the metadata and the
    /// usage of one turn go to; `rawValue` is the identifier the channel
    /// events carry.
    struct ResponseEntryID: EntryID { let rawValue: String }
    /// The transcript entry that the reasoning text of one turn goes to.
    struct ReasoningEntryID: EntryID { let rawValue: String }
    /// The transcript entry that the tool calls of one turn go to.
    struct ToolCallsEntryID: EntryID { let rawValue: String }

    /// Appends response text to the response entry.
    ///
    /// - Parameters:
    ///   - text: The text to append.
    ///   - entryID: The response entry.
    ///   - channel: The channel to send the event into.
    static func emitResponse(
        _ text: String, entryID: ResponseEntryID,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await emit(text: text, entryID: entryID.rawValue, destination: .response, into: channel)
    }

    /// Appends reasoning text to the reasoning entry.
    ///
    /// - Parameters:
    ///   - text: The text to append.
    ///   - entryID: The reasoning entry.
    ///   - channel: The channel to send the event into.
    static func emitReasoning(
        _ text: String, entryID: ReasoningEntryID,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await emit(text: text, entryID: entryID.rawValue, destination: .reasoning, into: channel)
    }

    /// Marks the response entry as incomplete output: the generation was
    /// cut off, or its output was dropped, before the answer completed.
    ///
    /// - Parameters:
    ///   - entryID: The response entry.
    ///   - channel: The channel to send the metadata into.
    static func emitIncompleteOutput(
        entryID: ResponseEntryID, into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await emitMetadata(["incompleteOutput": true], entryID: entryID.rawValue, into: channel)
    }

    /// Sends the usage of one pass to the response entry.
    ///
    /// - Parameters:
    ///   - input: The input half of the usage.
    ///   - output: The output half of the usage.
    ///   - entryID: The response entry.
    ///   - channel: The channel to send the usage into.
    static func emitUsage(
        input: LanguageModelExecutorGenerationChannel.Usage.Input,
        output: LanguageModelExecutorGenerationChannel.Usage.Output,
        entryID: ResponseEntryID,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await emitUsage(input: input, output: output, entryID: entryID.rawValue, into: channel)
    }

    /// Sends one tool call to the tool-calls entry.
    ///
    /// - Parameters:
    ///   - id: The tool-call identifier.
    ///   - name: The name of the tool.
    ///   - arguments: The arguments JSON of the call.
    ///   - entryID: The tool-calls entry.
    ///   - channel: The channel to send the tool call into.
    static func emitToolCall(
        id: String, name: String, arguments: String, entryID: ToolCallsEntryID,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await emitToolCall(
            id: id, name: name, arguments: arguments, entryID: entryID.rawValue,
            into: channel)
    }

    /// Runs the tool path of one generation turn — fresh turns and
    /// continuations alike.
    ///
    /// Allowed mode uses native generation so the model can answer or
    /// call a tool; required mode constrains generation to a real tool
    /// call.
    ///
    /// A tool body can start a second generation on this same model.
    /// This executor releases the model container BEFORE the SDK runs a
    /// tool body: the caller's `perform` closure sends the tool-call
    /// delta, then it returns, and each continuation round opens its
    /// own `perform`. Thus the container lock covers one generation. It
    /// does not cover a complete turn with the tool rounds of that
    /// turn. `ToolBodyContainerReentryTests` and
    /// `ToolBodyContainerReentryRealModelTests` guard this guarantee.
    ///
    /// - Parameters:
    ///   - request: The generation request for this turn
    ///   - messages: The chat messages the tool-aware prompt renders from
    ///   - enabledToolDefinitions: The tool definitions this turn exposes
    ///   - toolCallingMode: The resolved tool-calling mode
    ///   - schemaJSON: The encoded response schema, when the request carries one
    ///   - baselineInput: The prompt rendered without tool awareness
    ///   - resolved: The resolver-patched configuration
    ///   - declaresReasoning: Whether `.reasoning` was declared at init
    ///   - modelID: The model identifier the constraint caches key on
    ///   - requestedMaxTokens: The caller's token budget, when set
    ///   - samplingConfiguration: The translated sampling mode
    ///   - entryID: The transcript entry id for response events
    ///   - toolCallsEntryID: The transcript entry id for tool-call events
    ///   - reasoningEntryID: The transcript entry id for reasoning events
    ///   - promptCache: The session's prompt-cache slot
    ///   - speculation: The MTP drafter the turn speculates with, or nil
    ///   - context: The loaded model context
    ///   - channel: The channel to send response events into
    func runToolGeneration(
        request: LanguageModelExecutorGenerationRequest,
        messages: [Chat.Message],
        enabledToolDefinitions: [Transcript.ToolDefinition],
        toolCallingMode: GenerationOptions.ToolCallingMode,
        schemaJSON: String?,
        baselineInput: LMInput,
        resolved: ModelConfiguration,
        declaresReasoning: Bool,
        modelID: String,
        requestedMaxTokens: Int?,
        samplingConfiguration: MLXSamplingConfiguration?,
        entryID: ResponseEntryID,
        toolCallsEntryID: ToolCallsEntryID,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Re-render using the model's native tool-aware chat
        // template (Qwen/Llama/Phi/Gemma all ship one in their
        // tokenizer_config.json). This is what teaches the model
        // *what* tools exist and how to decide between them; the
        // grammar constraint below only enforces the *shape* of
        // whatever tool call it emits.
        // Think-then-call is gated to the enable_thinking
        // family (Qwen3/QwQ): their template both renders the tool
        // block AND honors `enable_thinking`. R1-style `.alwaysOn`
        // models are tool-blind (template ignores `tools:`), so
        // they fall through to the single-phase path unchanged;
        // thinking-disabled requests stay single-phase too.
        let thinkThenCallConfig: ReasoningConfig? = {
            guard declaresReasoning,
                let reasoningConfig = resolved.reasoningConfig,
                case .templateFlag = reasoningConfig.promptStrategy,
                Self.thinkingEnabled(
                    for: request.contextOptions.reasoningLevel) != false
            else { return nil }
            return reasoningConfig
        }()
        // Thread `enable_thinking` through the tool-aware template
        // so the prompt's thinking state matches how we drive
        // generation. For a toggleable model (`.templateFlag`, e.g.
        // Qwen3 whose `enable_thinking` defaults ON) the effective
        // value is:
        //   - reasoning declared: honor the requested level
        //     (default ON), and the think-then-call phase below
        //     lets the model reason before the grammar constrains it;
        //   - reasoning NOT declared: force thinking OFF, mirroring
        //     the unconstrained path's suppression (see
        //     `ReasoningPlan.suppressedInput`).
        // Forcing OFF here is load-bearing for both modes. Native
        // `.allowed` generation would otherwise surface undeclared
        // reasoning through response/tool parsing. In `.required`,
        // the grammar forces tool-call JSON from the first token, so
        // a thinking-primed model cannot emit its `<think>` block and
        // greedy decoding of unconstrained developer tool string
        // arguments can degenerate (Qwen: "1234567890...").
        // `.alwaysOn` models with reasoning undeclared were already
        // rejected by the capability gate above; `.none`/no-config
        // models take no context. A strategy that renders thinking off
        // as a closed block (Qwen 3.5) keeps the flag on and closes the
        // block after the render (`ThinkingRender`).
        let toolAwareRender: ThinkingRender
        if let config = resolved.reasoningConfig,
            case .templateFlag = config.promptStrategy
        {
            let enabled =
                declaresReasoning
                ? Self.thinkingEnabled(for: request.contextOptions.reasoningLevel)
                : false
            toolAwareRender = try ThinkingRender(config: config, thinkingEnabled: enabled)
        } else {
            toolAwareRender = ThinkingRender()
        }
        if ToolCallingModeResolution.usesAllowedBehavior(toolCallingMode) {
            try await runAllowedToolTurn(
                request: request,
                messages: messages,
                enabledToolDefinitions: enabledToolDefinitions,
                toolAwareRender: toolAwareRender,
                thinkThenCallConfig: thinkThenCallConfig,
                schemaJSON: schemaJSON,
                baselineInput: baselineInput,
                modelID: modelID,
                requestedMaxTokens: requestedMaxTokens,
                samplingConfiguration: samplingConfiguration,
                entryID: entryID,
                toolCallsEntryID: toolCallsEntryID,
                reasoningEntryID: reasoningEntryID,
                promptCache: promptCache,
                speculation: speculation,
                context: context,
                channel: channel)
            return
        }
        try await runRequiredToolTurn(
            request: request,
            messages: messages,
            enabledToolDefinitions: enabledToolDefinitions,
            toolAwareRender: toolAwareRender,
            thinkThenCallConfig: thinkThenCallConfig,
            modelID: modelID,
            requestedMaxTokens: requestedMaxTokens,
            samplingConfiguration: samplingConfiguration,
            entryID: entryID,
            toolCallsEntryID: toolCallsEntryID,
            reasoningEntryID: reasoningEntryID,
            promptCache: promptCache,
            speculation: speculation,
            context: context,
            channel: channel)
    }

    /// Runs allowed tool mode: native generation over the tool-aware
    /// prompt, so the model can answer in text or call a tool.
    ///
    /// The prompt goes through the model's UserInputProcessor (like the
    /// unconstrained and guided paths) instead of hand-building an
    /// LMInput from raw applyChatTemplate output: processors produce
    /// the token rank their model family requires (LLM processors emit
    /// [N]; VLM processors emit [1, N], and VLM `prepare` fatally
    /// aborts on 1-D input), and they carry image/video content through
    /// to the model.
    ///
    /// - Parameters:
    ///   - request: The generation request for this turn
    ///   - messages: The chat messages the tool-aware prompt renders from
    ///   - enabledToolDefinitions: The tool definitions this turn exposes
    ///   - toolAwareRender: The thinking state the tool-aware render carries
    ///   - thinkThenCallConfig: The reasoning config of the think-then-call phase, when active
    ///   - schemaJSON: The encoded response schema, when the request carries one
    ///   - baselineInput: The prompt rendered without tool awareness
    ///   - modelID: The model identifier the schema fallback keys on
    ///   - requestedMaxTokens: The caller's token budget, when set
    ///   - samplingConfiguration: The translated sampling mode
    ///   - entryID: The transcript entry id for response events
    ///   - toolCallsEntryID: The transcript entry id for tool-call events
    ///   - reasoningEntryID: The transcript entry id for reasoning events
    ///   - promptCache: The session's prompt-cache slot
    ///   - speculation: The MTP drafter the turn speculates with, or nil
    ///   - context: The loaded model context
    ///   - channel: The channel to send response events into
    func runAllowedToolTurn(
        request: LanguageModelExecutorGenerationRequest,
        messages: [Chat.Message],
        enabledToolDefinitions: [Transcript.ToolDefinition],
        toolAwareRender: ThinkingRender,
        thinkThenCallConfig: ReasoningConfig?,
        schemaJSON: String?,
        baselineInput: LMInput,
        modelID: String,
        requestedMaxTokens: Int?,
        samplingConfiguration: MLXSamplingConfiguration?,
        entryID: ResponseEntryID,
        toolCallsEntryID: ToolCallsEntryID,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let toolSpecs = try ToolCallingConversions.makeToolSpecs(
            from: enabledToolDefinitions)
        let toolAwareInput = try await toolAwareRender.prepare(
            messages: messages, tools: toolSpecs, context: context)
        let transcriptBoundary = await toolAwareRender.transcriptBoundary(
            of: toolAwareInput, messages: messages, tools: toolSpecs, context: context)
        let reasoning = thinkThenCallConfig.map {
            (
                config: $0,
                primedInside: Self.reasoningPrimedInside(
                    input: toolAwareInput,
                    config: $0,
                    tokenizer: context.tokenizer)
            )
        }
        let result = try await runAllowedToolGeneration(
            input: toolAwareInput,
            transcriptBoundary: transcriptBoundary,
            toolSpecs: toolSpecs,
            reasoning: reasoning,
            requestedMaxTokens: requestedMaxTokens,
            requestedTemperature: request.generationOptions.temperature,
            samplingConfiguration: samplingConfiguration,
            reasoningEntryID: reasoningEntryID,
            promptCache: promptCache,
            speculation: speculation,
            context: context,
            channel: channel)

        if result.endedInsideReasoning {
            await Self.emitIncompleteOutput(entryID: entryID, into: channel)
        } else if !result.toolCalls.isEmpty {
            for call in result.toolCalls {
                let argumentsData = try JSONEncoder().encode(
                    call.function.arguments)
                let arguments = String(
                    decoding: argumentsData, as: UTF8.self)
                await Self.emitToolCall(
                    id: call.id ?? UUID().uuidString,
                    name: call.function.name,
                    arguments: arguments,
                    entryID: toolCallsEntryID,
                    into: channel)
            }
        } else if let schemaJSON {
            try await runSchemaGeneration(
                schemaJSON: schemaJSON,
                input: baselineInput,
                modelID: modelID,
                requestedMaxTokens: requestedMaxTokens,
                entryID: entryID,
                promptCache: promptCache,
                speculation: speculation,
                context: context,
                channel: channel)
        } else {
            await Self.emitResponse(
                result.responseText, entryID: entryID, into: channel)
        }
        if schemaJSON == nil || !result.toolCalls.isEmpty
            || result.endedInsideReasoning
        {
            await emitAllowedUsage(
                result, entryID: entryID, promptCache: promptCache,
                channel: channel)
        }
        Stream.gpu.synchronize()
    }

    /// Runs required tool mode: guided generation constrained to a real
    /// tool call.
    ///
    /// Required mode is the only mode that reaches guided tool
    /// generation, and it uses developer definitions only.
    /// `GuidedGenerationLoop.run` builds its own key/value cache and
    /// accepts none from a caller, thus this pass carries nothing from
    /// an earlier turn.
    ///
    /// - Parameters:
    ///   - request: The generation request for this turn
    ///   - messages: The chat messages the tool-aware prompt renders from
    ///   - enabledToolDefinitions: The tool definitions the grammar constrains to
    ///   - toolAwareRender: The thinking state the tool-aware render carries
    ///   - thinkThenCallConfig: The reasoning config of the think-then-call phase, when active
    ///   - modelID: The model identifier the constraint caches key on
    ///   - requestedMaxTokens: The caller's token budget, when set
    ///   - samplingConfiguration: The translated sampling mode
    ///   - entryID: The transcript entry id for response events
    ///   - toolCallsEntryID: The transcript entry id for tool-call events
    ///   - reasoningEntryID: The transcript entry id for reasoning events
    ///   - promptCache: The session's prompt-cache slot
    ///   - speculation: The MTP drafter of the think-then-call phase and of
    ///     the grammar phase after it, or nil.
    ///   - context: The loaded model context
    ///   - channel: The channel to send response events into
    func runRequiredToolTurn(
        request: LanguageModelExecutorGenerationRequest,
        messages: [Chat.Message],
        enabledToolDefinitions: [Transcript.ToolDefinition],
        toolAwareRender: ThinkingRender,
        thinkThenCallConfig: ReasoningConfig?,
        modelID: String,
        requestedMaxTokens: Int?,
        samplingConfiguration: MLXSamplingConfiguration?,
        entryID: ResponseEntryID,
        toolCallsEntryID: ToolCallsEntryID,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        promptCache.carriesNoCache()
        let requiredToolDefinitions = enabledToolDefinitions
        let toolSpecs = try ToolCallingConversions.makeToolSpecs(
            from: requiredToolDefinitions)
        let toolAwareInput = try await toolAwareRender.prepare(
            messages: messages, tools: toolSpecs, context: context)

        let toolCallingGrammar =
            try SchemaConverter.encodeToolCallingGrammar(
                tools: requiredToolDefinitions
            )
        // The inner JSON envelope is still needed separately to
        // seed `CompletionReserve` -- the wrapper tokens
        // (`<tool_call>`, two `\n`s, `</tool_call>`) are small
        // and fixed, so padding the reserve with their
        // tokenized size adds noise rather than accuracy.
        let toolCallingEnvelopeJSON =
            try SchemaConverter.encodeToolCallingEnvelopeJSON(
                tools: requiredToolDefinitions
            )

        // Always partition into zones -- the grammar has
        // wiggle room (JSON whitespace before the outer
        // `}`, whitespace before `\n</tool_call>`) that
        // open-source models tend to exploit into infinite
        // loops when not pushed toward structural close.
        // Use the caller's budget when set, otherwise the
        // Executor's default.
        let maxTokens = requestedMaxTokens ?? Self.defaultMaxTokens
        let grammar = try await Self.guidedGrammar(
            kind: .structuralTag,
            source: toolCallingGrammar,
            reserveSchemaJSON: toolCallingEnvelopeJSON,
            maxTokens: maxTokens,
            modelID: modelID,
            context: context)

        // PHASE 1 (think-then-call): reason unconstrained until
        // `</think>`, retaining the token IDs to prefill into the
        // constrained phase below. Empty on the single-phase path.
        var reasoningTokenIDs: [Int] = []
        if let thinkThenCallConfig {
            let primedInside = Self.reasoningPrimedInside(
                input: toolAwareInput, config: thinkThenCallConfig,
                tokenizer: context.tokenizer)
            let phase1 = try await runToolCallReasoningPhase(
                input: toolAwareInput, config: thinkThenCallConfig,
                primedInside: primedInside, maxTokens: maxTokens,
                requestedTemperature: request.generationOptions
                    .temperature,
                samplingConfiguration: samplingConfiguration,
                reasoningEntryID: reasoningEntryID,
                responseEntryID: entryID,
                speculation: speculation,
                context: context, channel: channel)
            reasoningTokenIDs = phase1.tokenIDs
            if !phase1.closed {
                // Cut off mid-thought (budget exhausted before
                // `</think>`). Don't prefill a truncated thought
                // into the grammar — signal and finish. Phase 1
                // already synchronized the GPU on its way out.
                await Self.emitIncompleteOutput(entryID: entryID, into: channel)
                return
            }
        }

        // Phase 2 continues from the model's completed reasoning;
        // carry the raw IDs (no decode/re-encode) so the grammar
        // starts from the exact post-`</think>` state.
        let phase2Input =
            reasoningTokenIDs.isEmpty
            ? toolAwareInput
            : Self.continuationInput(
                from: toolAwareInput, appending: reasoningTokenIDs)
        // Shared budget (match the unconstrained path): the
        // envelope continues under the remaining budget, floored
        // at the completion reserve so it always has room to close
        // the tool call.
        let phase2MaxTokens =
            reasoningTokenIDs.isEmpty
            ? maxTokens
            : Swift.max(
                maxTokens - reasoningTokenIDs.count, grammar.completionReserve)

        var outputBuffer = ""
        let (generatedTokenCount, incomplete) = try Self.runGuidedLoop(
            input: phase2Input, grammar: grammar, maxTokens: phase2MaxTokens,
            speculation: speculation, context: context
        ) { outputBuffer += $0 }
        try Task.checkCancellation()

        GuidedGenerationDiagnosticSink.current?.recordBuffer(
            outputBuffer, incompleteOutput: incomplete)

        let emittedToolCall = await emitRequiredToolCallEvent(
            outputBuffer: outputBuffer,
            toolCallsEntryID: toolCallsEntryID,
            channel: channel
        )

        if let generatedTokenCount {
            // Output total spans both phases (reasoning + envelope);
            // the reasoning subset is the Phase-1 token count,
            // clamped ≤ total.
            let reasoningCount = reasoningTokenIDs.count
            let totalOutput = generatedTokenCount + reasoningCount
            await Self.emitUsage(
                input: Self.usageInput(
                    fedTokenCount: toolAwareInput.text.tokens.size,
                    promptCache: promptCache),
                output: .init(
                    totalTokenCount: totalOutput,
                    reasoningTokenCount: Swift.min(reasoningCount, totalOutput)),
                entryID: entryID, into: channel)
        }

        if incomplete || !emittedToolCall {
            await Self.emitIncompleteOutput(entryID: entryID, into: channel)
        }
    }

    struct AllowedToolGenerationResult {
        var responseText = ""
        var toolCalls: [MLXLMCommon.ToolCall] = []
        var rejectedToolCalls: [RejectedToolCall] = []
        var completionInfo: GenerateCompletionInfo?
        var reasoningTokenCount = 0
        var endedInsideReasoning = false
    }

    /// Drains the active generations of one `respond` at cancellation.
    ///
    /// The framework resumes a cancelled `LanguageModelSession.respond`
    /// before this executor's own teardown completes, so the caller
    /// could otherwise see its `CancellationError` while a forward pass
    /// still runs on the GPU — and a process exit in that window aborts
    /// on signal 6 or 11. The coordinator closes that window from the
    /// cancellation side: each generation registers as one section, and
    /// the cancellation handler cancels every registered generation and
    /// waits until every section left.
    ///
    /// The wait must not depend on the respond task making progress —
    /// a cancellation handler runs BEFORE the runtime resumes that
    /// task's suspended continuations, so a wait on the whole `respond`
    /// deadlocks against its own delivery. Every section therefore
    /// leaves through a path that needs no cancellation delivery: a
    /// token-producer task polls its own flag, which the direct
    /// `cancel()` here sets, and a watcher task reports its end; a
    /// guided-generation loop polls the respond task's flag, which the
    /// runtime sets before it runs this handler, and leaves on its own
    /// thread.
    ///
    /// An `NSCondition` guards the state; every access takes the lock,
    /// which is the synchronization invariant behind
    /// `@unchecked Sendable`. A condition rather than an actor, because
    /// a cancellation handler is synchronous and cannot await.
    // swiftlint:disable:next no_unchecked_sendable `condition` guards all the mutable state
    final class GenerationDrainCoordinator: @unchecked Sendable {

        /// Longest the canceller waits for quiescence.
        ///
        /// An active section ends within one decode step plus a stream
        /// synchronize. The generation loop checks cancellation before
        /// each step: one forward pass for plain decode, and one whole
        /// round for MTP decode — the drafter passes of the round and one
        /// verify pass, a few forward passes at most. That is under a
        /// second on a 1B model and within seconds on a 30B model. The
        /// bound stands far above both so a pathological hang cannot
        /// block the canceller forever.
        private static let quiescenceTimeout: TimeInterval = 60

        private let condition = NSCondition()
        private var activeSections = 0
        private var cancellationRequested = false
        private var pendingCancels: [() -> Void] = []

        /// Enters one generation section.
        ///
        /// Throws `CancellationError` when the respond task is already
        /// cancelled, so an abandoned `respond` can start no new GPU
        /// work after the canceller was released.
        func enterSection() throws {
            try Task.checkCancellation()
            condition.lock()
            activeSections += 1
            condition.unlock()
        }

        /// Registers the closure that cancels the section's generation.
        ///
        /// When the cancellation already ran, the closure runs at once,
        /// so a generation that started inside the cancellation race is
        /// still cancelled and its section still leaves.
        func attachCancel(_ cancel: @escaping @Sendable () -> Void) {
            condition.lock()
            let requested = cancellationRequested
            if !requested {
                pendingCancels.append(cancel)
            }
            condition.unlock()
            if requested {
                cancel()
            }
        }

        /// Leaves one generation section and wakes the canceller.
        func leaveSection() {
            condition.lock()
            activeSections -= 1
            condition.broadcast()
            condition.unlock()
        }

        /// Cancels every registered generation and waits for quiescence.
        ///
        /// Runs on the canceller's thread, inside the `cancel()` call
        /// that cancels the respond task, thus that call does not return
        /// — and the framework does not resume the caller — until every
        /// active section left or ``quiescenceTimeout`` passed.
        func cancelActiveAndAwaitQuiescence() {
            condition.lock()
            cancellationRequested = true
            let cancels = pendingCancels
            pendingCancels.removeAll()
            condition.unlock()
            for cancel in cancels {
                cancel()
            }
            let deadline = Date(timeIntervalSinceNow: Self.quiescenceTimeout)
            condition.lock()
            while activeSections > 0 {
                guard condition.wait(until: deadline) else { break }
            }
            condition.unlock()
        }
    }

    /// Starts a token-producer task inside a drain-coordinator section.
    ///
    /// The section is entered BEFORE `make` runs, so a cancellation that
    /// strikes between the two finds the section active and waits for
    /// it; ``GenerationDrainCoordinator/attachCancel(_:)`` then cancels
    /// the fresh producer at registration. The watcher `Task` is
    /// deliberately unstructured: it must outlive a `respond` the
    /// framework abandoned, it only awaits the producer and reports its
    /// end, and it ends the moment the producer does.
    ///
    /// - Parameter make: builds the stream and its producer task.
    /// - Returns: what `make` returned.
    /// - Throws: `CancellationError` when the respond task is already
    ///   cancelled, and whatever `make` throws.
    static func startTrackedGeneration<Element: Sendable, Value: Sendable>(
        _ make: () throws -> (AsyncStream<Element>, Task<Value, Never>)
    ) throws -> (AsyncStream<Element>, Task<Value, Never>) {
        guard let coordinator = drainCoordinator else { return try make() }
        try coordinator.enterSection()
        do {
            let (stream, task) = try make()
            coordinator.attachCancel { task.cancel() }
            Task {
                _ = await task.value
                coordinator.leaveSection()
            }
            return (stream, task)
        } catch {
            coordinator.leaveSection()
            throw error
        }
    }

    /// Runs one guided-generation pass inside a drain-coordinator section.
    ///
    /// A guided loop runs synchronously on the respond task and polls
    /// `Task.isCancelled` per token, so it leaves its section on its own
    /// thread with no cancellation delivery needed. The GPU stream is
    /// synchronized before the section leaves, thus a canceller that the
    /// coordinator releases finds the GPU quiet.
    ///
    /// The MTP rounds and fallbacks of the pass go into the MTP outcome of
    /// the call, when the model speculates.
    ///
    /// - Parameter body: the guided-generation pass.
    /// - Returns: what `body` returned.
    /// - Throws: `CancellationError` when the respond task is already
    ///   cancelled, and whatever `body` throws.
    static func withGuidedGenerationSection<Value>(
        _ body: () throws -> Value
    ) throws -> Value {
        guard let coordinator = drainCoordinator else {
            return try SpeculativeDecodingOutcome.recordingGuidedPass(body)
        }
        try coordinator.enterSection()
        defer {
            Stream.gpu.synchronize()
            coordinator.leaveSection()
        }
        return try SpeculativeDecodingOutcome.recordingGuidedPass(body)
    }

    /// Stops the generation task and waits until it stops.
    ///
    /// The task holds the GPU loop. A caller that leaves the token stream
    /// before its end must stop the task and then wait for it. A caller that
    /// does not wait leaves the loop in operation, and the next pass then
    /// shares the GPU with it.
    ///
    /// - Parameter task: the generation task to stop.
    func cancelAndDrain<Success>(_ task: Task<Success, Never>) async {
        task.cancel()
        _ = await task.value
    }

    /// Reads a generation stream and keeps its cache for the next turn.
    ///
    /// Both exits wait for the generation task, thus the GPU loop never
    /// stays in operation. A good exit gives the prompt cache the tokens
    /// that the task fed into the caches. The ledger of that cache is then
    /// the render plus those tokens, thus the next turn reuses the cache
    /// with no rewind. An error exit stops the task first, sends the error
    /// on, and keeps no cache.
    ///
    /// - Parameters:
    ///   - task: the generation task that feeds the caches.
    ///   - plan: the plan that the pass ran, or nil when the pass had no
    ///     plan.
    ///   - state: the model state the prefill of the pass left, which the
    ///     next turn seeds its iterator with.
    ///   - drafterState: receives the MTP drafter continuation of the
    ///     generation when it ends, which the next turn resumes.
    ///   - promptCache: the slot that keeps the cache for the next turn.
    ///   - body: reads the token stream of the same generation.
    /// - Throws: the error that `body` throws.
    func withPromptCacheCommit(
        task: Task<[Int], Never>,
        plan: ExecutorPromptCachePlan?,
        state: LMOutput.State?,
        drafterState: ExecutorDrafterStateReceiver,
        promptCache: ExecutorPromptCacheSlot,
        body: () async throws -> Void
    ) async rethrows {
        do {
            try await body()
        } catch {
            await cancelAndDrain(task)
            throw error
        }
        // The generation hands its drafter state over before the task
        // returns its tokens, thus the receiver holds it here.
        let generatedTokens = await task.value
        promptCache.commit(
            plan, generatedTokens: generatedTokens, state: state,
            drafterState: drafterState.drafterState)
    }

    /// Reads the cache of the session back from disk when it went there,
    /// and plans the pass that is about to run.
    ///
    /// - Parameters:
    ///   - promptCache: the slot of the session.
    ///   - input: the prepared input of the pass.
    ///   - parameters: the generation parameters of the pass.
    ///   - transcriptBoundary: the index of the prompt where the pass
    ///     takes its checkpoint, or nil.
    ///   - speculation: the MTP drafter of the pass, or nil. The plan
    ///     resumes the drafter state the cache keeps for that drafter.
    ///   - context: the loaded model context.
    /// - Returns: the plan, or nil when the pass carries no cache.
    /// - Throws: the error of the plan.
    static func planPromptCache(
        _ promptCache: ExecutorPromptCacheSlot, input: LMInput,
        parameters: GenerateParameters, transcriptBoundary: Int?,
        speculation: MTPSpeculation?, context: ModelContext
    ) throws -> ExecutorPromptCachePlan? {
        promptCache.restoreIfPending(
            model: context.model, parameters: parameters, speculation: speculation)
        return try promptCache.plan(
            input: input, model: context.model, parameters: parameters,
            protocolRules: promptCacheReuseRules(of: context),
            transcriptBoundary: transcriptBoundary, speculation: speculation,
            decodeTokens: context.tokenizer.decode(tokenIds:))
    }

    func runAllowedToolGeneration(
        input: LMInput,
        transcriptBoundary: Int?,
        toolSpecs: [[String: any Sendable]],
        reasoning: (config: ReasoningConfig, primedInside: Bool)?,
        requestedMaxTokens: Int?,
        requestedTemperature: Double?,
        samplingConfiguration: MLXSamplingConfiguration?,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws -> AllowedToolGenerationResult {
        let parameters = Self.makeParameters(
            maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
            requestedTemperature: requestedTemperature,
            samplingConfiguration: samplingConfiguration)
        let plan = try Self.planPromptCache(
            promptCache, input: input, parameters: parameters,
            transcriptBoundary: transcriptBoundary, speculation: speculation,
            context: context)
        let format = context.configuration.toolCallFormat ?? .json
        var router = AllowedToolOutputRouter(
            format: format,
            tools: toolSpecs,
            reasoning: reasoning)
        var protocolDecoder = format.makeProtocolTokenStreamDecoder(
            tokenizer: context.tokenizer,
            tools: toolSpecs,
            stopStrings: context.configuration.effectiveStopStrings)
        var detokenizer = context.tokenizer.makeStreamingDetokenizer()
        var result = AllowedToolGenerationResult()
        let prefill = ExecutorPrefillRecorder(promptCache: promptCache)
        let finalDrafterState = ExecutorDrafterStateReceiver()
        let (stream, task) = try Self.startTrackedGeneration {
            try generateProtocolTokensTask(
                input: plan?.input ?? input,
                cache: plan?.caches,
                state: plan?.state,
                parameters: parameters,
                context: context,
                decoder: protocolDecoder,
                speculation: speculation,
                mtpContinuation: plan?.drafterState,
                preparedState: prefill.record,
                drafterContinuation: finalDrafterState.receive)
        }

        try await withPromptCacheCommit(
            task: task, plan: plan, state: prefill.state,
            drafterState: finalDrafterState, promptCache: promptCache
        ) {
            generationLoop: for await generation in stream {
                try Task.checkCancellation()
                switch generation {
                case .token(let token):
                    if var decoder = protocolDecoder {
                        if decoder.isInsideReasoning {
                            result.reasoningTokenCount += 1
                        }
                        var reasoningText = ""
                        var shouldContinue = true
                        let decoderContinues = decoder.push(token) { event in
                            shouldContinue = consumeProtocolEvent(
                                event, result: &result, reasoningText: &reasoningText)
                            return shouldContinue
                        }
                        protocolDecoder = decoder
                        if !reasoningText.isEmpty {
                            await Self.emitReasoning(
                                reasoningText, entryID: reasoningEntryID, into: channel)
                            try Task.checkCancellation()
                        }
                        if !decoderContinues || !shouldContinue {
                            task.cancel()
                            break generationLoop
                        }
                    } else {
                        if router.isInsideReasoning {
                            result.reasoningTokenCount += 1
                        }
                        detokenizer.append(token: token)
                        if let chunk = detokenizer.next() {
                            let reasoningChunks = consumeAllowedEvents(
                                router.process(chunk), result: &result)
                            for text in reasoningChunks {
                                await Self.emitReasoning(
                                    text, entryID: reasoningEntryID, into: channel)
                                try Task.checkCancellation()
                            }
                        }
                    }
                case .info(let info):
                    Self.observeCompletion(info)
                    result.completionInfo = info
                }
            }
        }

        let finalReasoningText: String
        if var decoder = protocolDecoder {
            result.endedInsideReasoning = decoder.isInsideReasoning
            var reasoningText = ""
            _ = decoder.finish { event in
                consumeProtocolEvent(
                    event, result: &result, reasoningText: &reasoningText)
            }
            protocolDecoder = decoder
            finalReasoningText = reasoningText
        } else {
            let pending = detokenizer.finish().map { router.process($0) } ?? []
            let chunks = consumeAllowedEvents(pending + router.finish(), result: &result)
            finalReasoningText = chunks.joined()
            result.endedInsideReasoning = router.isInsideReasoning
        }
        if !finalReasoningText.isEmpty {
            await Self.emitReasoning(
                finalReasoningText, entryID: reasoningEntryID, into: channel)
        }
        if let rejection = result.rejectedToolCalls.first {
            throw RejectedToolCallError(rejection)
        }
        return result
    }

    func consumeProtocolEvent(
        _ event: TokenStreamEvent,
        result: inout AllowedToolGenerationResult,
        reasoningText: inout String
    ) -> Bool {
        switch event {
        case .reasoning(let text): reasoningText += text
        case .response(let text): result.responseText += text
        case .toolCall(let call): result.toolCalls.append(call)
        case .rejectedToolCall(let rejection): result.rejectedToolCalls.append(rejection)
        case .protocolError(let message): Self.protocolLogger.error("\(message)")
        case .stop: return false
        }
        return true
    }

    func consumeAllowedEvents(
        _ events: [AllowedToolOutputRouter.Event],
        result: inout AllowedToolGenerationResult
    ) -> [String] {
        var reasoningChunks: [String] = []
        for event in events {
            switch event {
            case .reasoning(let text):
                reasoningChunks.append(text)
            case .response(let text):
                result.responseText += text
            case .toolCall(let call):
                result.toolCalls.append(call)
            case .rejectedToolCall(let rejection):
                result.rejectedToolCalls.append(rejection)
            }
        }
        return reasoningChunks
    }

    func emitAllowedUsage(
        _ result: AllowedToolGenerationResult,
        entryID: ResponseEntryID,
        promptCache: ExecutorPromptCacheSlot,
        channel: LanguageModelExecutorGenerationChannel
    ) async {
        guard let info = result.completionInfo else { return }
        await Self.emitUsage(
            input: Self.usageInput(
                fedTokenCount: info.promptTokenCount, promptCache: promptCache),
            output: .init(
                totalTokenCount: Self.generatedTokenCount(of: info),
                reasoningTokenCount: min(
                    result.reasoningTokenCount,
                    info.generationTokenCount)),
            entryID: entryID,
            into: channel)
    }

    /// The grammar, the biases and the zone reserves of one guided pass.
    struct GuidedGrammar {
        /// The grammar tokenizer of the model.
        let grammarTokenizer: GrammarTokenizer
        /// The constraint the pass generates under.
        let constraint: GrammarConstraint
        /// The closing and whitespace biases of the model.
        let bias: TokenizerBias
        /// The reserve the closing zone of the pass starts at.
        let completionReserve: Int
        /// The reserve the hard closing zone of the pass starts at.
        let hardReserve: Int
    }

    /// Builds the grammar of one guided pass and sizes its reserves.
    ///
    /// Schema output and required tool calls build the same setup; they
    /// differ only in the constraint kind, its source, and the JSON the
    /// structural reserve is estimated from.
    ///
    /// - Parameters:
    ///   - kind: The kind of the constraint
    ///   - source: The grammar source of the constraint
    ///   - reserveSchemaJSON: The JSON schema the structural reserve is estimated from
    ///   - maxTokens: The token budget of the pass
    ///   - modelID: The model identifier the caches key on
    ///   - context: The loaded model context
    /// - Returns: The grammar and reserves of the pass.
    /// - Throws: The error of the grammar tokenizer or of the constraint.
    static func guidedGrammar(
        kind: ConstraintKind,
        source: String,
        reserveSchemaJSON: String,
        maxTokens: Int,
        modelID: String,
        context: ModelContext
    ) async throws -> GuidedGrammar {
        let grammarTokenizer = try await MLXLanguageModel.makeGrammarTokenizer(
            modelID: modelID,
            tokenizer: context.tokenizer)
        let constraint = try await MLXLanguageModel.makeConstraint(
            modelID: modelID,
            kind: kind,
            source: source,
            tokenizer: grammarTokenizer,
            hostTokenizer: context.tokenizer,
            fastForward: true)
        let bias = await MLXLanguageModel.makeTokenizerBias(
            modelID: modelID,
            tokenizer: context.tokenizer)
        let structuralReserve = CompletionReserve.estimate(
            schemaJSON: reserveSchemaJSON,
            tokenizer: context.tokenizer)
        return GuidedGrammar(
            grammarTokenizer: grammarTokenizer,
            constraint: constraint,
            bias: bias,
            completionReserve: Swift.max(
                structuralReserve * completionReserveStructuralMultiplier,
                maxTokens / completionReserveBudgetDivisor),
            hardReserve: structuralReserve * hardReserveStructuralMultiplier)
    }

    /// Runs one guided pass inside the guided-generation section.
    ///
    /// - Parameters:
    ///   - input: The prompt of the pass
    ///   - grammar: The grammar and reserves of the pass
    ///   - maxTokens: The token budget of the pass
    ///   - speculation: The MTP drafter the pass speculates with, or nil
    ///   - context: The loaded model context
    ///   - onText: Receives each piece of text the pass emits
    /// - Returns: The generated token count, nil when the output is
    ///   incomplete, and whether it is.
    /// - Throws: The errors of the loop other than incomplete output.
    static func runGuidedLoop(
        input: LMInput,
        grammar: GuidedGrammar,
        maxTokens: Int,
        speculation: MTPSpeculation?,
        context: ModelContext,
        onText: (String) -> Void
    ) throws -> (generatedTokenCount: Int?, incomplete: Bool) {
        do {
            let generatedTokenCount = try withGuidedGenerationSection {
                try GuidedGenerationLoop.run(
                    input: input,
                    context: context,
                    constraint: grammar.constraint,
                    maxTokens: maxTokens,
                    vocabSize: Int(grammar.grammarTokenizer.vocabSize),
                    completionReserve: grammar.completionReserve,
                    hardReserve: grammar.hardReserve,
                    closingBias: grammar.bias.closing,
                    whitespaceBias: grammar.bias.whitespace,
                    whitespaceTokenIDs: grammar.bias.whitespaceTokenIDs,
                    speculation: speculation
                ) { text in
                    onText(text)
                    GuidedGenerationDiagnosticSink.current?.recordEmit()
                    return !Task.isCancelled
                }
            }
            return (generatedTokenCount, false)
        } catch GuidedGenerationError.incompleteOutput {
            return (nil, true)
        }
    }

    /// Runs schema output: guided generation constrained to the JSON schema of the request.
    ///
    /// - Parameters:
    ///   - schemaJSON: The JSON schema the grammar constrains to
    ///   - input: The rendered prompt
    ///   - modelID: The model identifier the constraint caches key on
    ///   - requestedMaxTokens: The caller's token budget, when set
    ///   - entryID: The transcript entry id for response events
    ///   - promptCache: The session's prompt-cache slot
    ///   - speculation: The MTP drafter the grammar pass speculates with, or nil
    ///   - context: The loaded model context
    ///   - channel: The channel to send response events into
    func runSchemaGeneration(
        schemaJSON: String,
        input: LMInput,
        modelID: String,
        requestedMaxTokens: Int?,
        entryID: ResponseEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // `GuidedGenerationLoop.run` builds its own key/value cache and
        // accepts none from a caller, thus this pass carries nothing from an
        // earlier turn and reports nothing.
        promptCache.carriesNoCache()
        let maxTokens = requestedMaxTokens ?? Self.defaultMaxTokens
        let grammar = try await Self.guidedGrammar(
            kind: .json,
            source: schemaJSON,
            reserveSchemaJSON: schemaJSON,
            maxTokens: maxTokens,
            modelID: modelID,
            context: context)

        let (textStream, textContinuation) = AsyncStream<String>.makeStream()
        async let forwarder: Void = {
            for await text in textStream {
                await Self.emitResponse(
                    text, entryID: entryID, into: channel)
            }
        }()

        let (generatedTokenCount, incomplete) = try Self.runGuidedLoop(
            input: input, grammar: grammar, maxTokens: maxTokens,
            speculation: speculation, context: context
        ) { textContinuation.yield($0) }

        let cancellationError: Error?
        do {
            try Task.checkCancellation()
            cancellationError = nil
        } catch {
            cancellationError = error
        }
        textContinuation.finish()
        await forwarder
        if let cancellationError {
            throw cancellationError
        }

        if let generatedTokenCount {
            await Self.emitUsage(
                input: Self.usageInput(
                    fedTokenCount: input.text.tokens.size, promptCache: promptCache),
                output: .init(
                    totalTokenCount: generatedTokenCount,
                    reasoningTokenCount: 0),
                entryID: entryID,
                into: channel)
        }
        if incomplete {
            await Self.emitIncompleteOutput(entryID: entryID, into: channel)
        }
    }

    /// Unconstrained text generation. Used on the no-tools/no-schema
    /// path when the model has no reasoning config to route through.
    func runUnconstrained(
        input: LMInput,
        transcriptBoundary: Int?,
        requestedMaxTokens: Int?,
        requestedTemperature: Double?,
        samplingConfiguration: MLXSamplingConfiguration?,
        entryID: ResponseEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Use a finite default when the framework doesn't specify a
        // token limit; there's no grammar to stop the model naturally.
        let parameters = Self.makeParameters(
            maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
            requestedTemperature: requestedTemperature,
            samplingConfiguration: samplingConfiguration
        )
        let plan = try Self.planPromptCache(
            promptCache, input: input, parameters: parameters,
            transcriptBoundary: transcriptBoundary, speculation: speculation,
            context: context)

        // The token-recording form is what lets this pass leave a prompt
        // cache behind: the ledger of that cache names the render plus the
        // tokens generation fed into it, and the text stream alone does not
        // carry those token identifiers.
        let prefill = ExecutorPrefillRecorder(promptCache: promptCache)
        let finalDrafterState = ExecutorDrafterStateReceiver()
        let (stream, task) = try Self.startTrackedGeneration {
            try generateTaskRecordingTokens(
                input: plan?.input ?? input,
                cache: plan?.caches,
                state: plan?.state,
                parameters: parameters,
                context: context,
                speculation: speculation,
                mtpContinuation: plan?.drafterState,
                preparedState: prefill.record,
                drafterContinuation: finalDrafterState.receive)
        }

        try await withPromptCacheCommit(
            task: task, plan: plan, state: prefill.state,
            drafterState: finalDrafterState, promptCache: promptCache
        ) {
            for await generation in stream {
                try Task.checkCancellation()
                switch generation {
                case .chunk(let text):
                    await Self.emitResponse(text, entryID: entryID, into: channel)
                case .info(let info):
                    // MLX-LM emits one .info event at end-of-generation with
                    // authoritative scalar token counts (`promptTokenCount`
                    // is the prompt; `generationTokenCount` is the
                    // model-generated completion -- see GenerationResults.swift's
                    // `GenerateCompletionInfo` definition).
                    Self.observeCompletion(info)
                    await Self.emitUsage(
                        input: Self.usageInput(
                            fedTokenCount: info.promptTokenCount, promptCache: promptCache),
                        output: .init(
                            totalTokenCount: Self.generatedTokenCount(of: info),
                            reasoningTokenCount: 0),
                        entryID: entryID, into: channel)
                case .toolCall(_):
                    break
                case .rejectedToolCall(let rejection):
                    throw RejectedToolCallError(rejection)
                }
            }
        }
    }

    /// Dispatches the no-tools/no-schema path: reasoning routing when a
    /// config resolved, otherwise plain unconstrained text.
    ///
    /// `transcriptBoundary` is the checkpoint position of the input this
    /// method feeds: the reasoning input when `reasoningSetup` is set,
    /// and `fallbackInput` otherwise.
    func runTextGeneration(
        reasoningSetup: (input: LMInput, config: ReasoningConfig, primedInside: Bool)?,
        fallbackInput: LMInput,
        transcriptBoundary: Int?,
        requestedMaxTokens: Int?,
        requestedTemperature: Double?,
        samplingConfiguration: MLXSamplingConfiguration?,
        responseEntryID: ResponseEntryID,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if let reasoning = reasoningSetup {
            try await runReasoning(
                input: reasoning.input,
                transcriptBoundary: transcriptBoundary,
                reasoningConfig: reasoning.config,
                primedInside: reasoning.primedInside,
                requestedMaxTokens: requestedMaxTokens,
                requestedTemperature: requestedTemperature,
                samplingConfiguration: samplingConfiguration,
                responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID,
                promptCache: promptCache,
                speculation: speculation,
                context: context,
                channel: channel)
        } else {
            try await runUnconstrained(
                input: fallbackInput,
                transcriptBoundary: transcriptBoundary,
                requestedMaxTokens: requestedMaxTokens,
                requestedTemperature: requestedTemperature,
                samplingConfiguration: samplingConfiguration,
                entryID: responseEntryID,
                promptCache: promptCache,
                speculation: speculation,
                context: context,
                channel: channel)
        }
    }

    /// Reasoning-aware unconstrained generation.
    ///
    /// Routes thinking delimited by the model's reasoning markers to
    /// `.reasoning` events and the rest to `.response`, using a raw
    /// protocol-neutral token decoder when the format owns framing, or a
    /// self-owned `StreamingDetokenizer` for ordinary formats. The loop
    /// sees real token IDs for an accurate reasoning token count.
    func runReasoning(
        input: LMInput,
        transcriptBoundary: Int?,
        reasoningConfig: ReasoningConfig,
        primedInside: Bool,
        requestedMaxTokens: Int?,
        requestedTemperature: Double?,
        samplingConfiguration: MLXSamplingConfiguration?,
        responseEntryID: ResponseEntryID,
        reasoningEntryID: ReasoningEntryID,
        promptCache: ExecutorPromptCacheSlot,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let parameters = Self.makeParameters(
            maxTokens: requestedMaxTokens ?? Self.defaultMaxTokens,
            requestedTemperature: requestedTemperature,
            samplingConfiguration: samplingConfiguration
        )
        let plan = try Self.planPromptCache(
            promptCache, input: input, parameters: parameters,
            transcriptBoundary: transcriptBoundary, speculation: speculation,
            context: context)

        var emitter = ReasoningEventEmitter(
            config: reasoningConfig, primedInside: primedInside)
        let format = context.configuration.toolCallFormat ?? .json
        var protocolDecoder = format.makeProtocolTokenStreamDecoder(
            tokenizer: context.tokenizer,
            tools: nil,
            stopStrings: context.configuration.effectiveStopStrings)
        var detokenizer = context.tokenizer.makeStreamingDetokenizer()
        var reasoningTokenCount = 0
        var completionInfo: GenerateCompletionInfo?
        // Surfaced as a thrown error, as on the schema and allowed-tool
        // paths: this path passes no tools, so a tool-call-shaped output is
        // never executable.
        var decodedEvents = ReasoningDecodedEvents()
        let prefill = ExecutorPrefillRecorder(promptCache: promptCache)
        let finalDrafterState = ExecutorDrafterStateReceiver()
        let (stream, task) = try Self.startTrackedGeneration {
            try generateProtocolTokensTask(
                input: plan?.input ?? input,
                cache: plan?.caches,
                state: plan?.state,
                parameters: parameters,
                context: context,
                decoder: protocolDecoder,
                speculation: speculation,
                mtpContinuation: plan?.drafterState,
                preparedState: prefill.record,
                drafterContinuation: finalDrafterState.receive)
        }

        try await withPromptCacheCommit(
            task: task, plan: plan, state: prefill.state,
            drafterState: finalDrafterState, promptCache: promptCache
        ) {
            generationLoop: for await generation in stream {
                try Task.checkCancellation()
                switch generation {
                case .token(let token):
                    if var decoder = protocolDecoder {
                        if decoder.isInsideReasoning {
                            reasoningTokenCount += 1
                        }
                        let decoderContinues = decoder.push(token) { event in
                            Self.consumeDecodedEvent(event, into: &decodedEvents)
                        }
                        protocolDecoder = decoder
                        for segment in decodedEvents.takeSegments() {
                            await Self.send(
                                segment, responseEntryID: responseEntryID,
                                reasoningEntryID: reasoningEntryID, channel: channel)
                        }
                        if let rejection = decodedEvents.rejectedToolCall {
                            throw RejectedToolCallError(rejection)
                        }
                        if !decoderContinues || !decodedEvents.shouldContinue {
                            task.cancel()
                            break generationLoop
                        }
                    } else {
                        // One `.token` == one real token, so this is a true
                        // token count. Closing delimiters are deliberately
                        // attributed to reasoning until the emitter consumes
                        // them, then the final usage count is clamped.
                        if emitter.isInsideReasoning {
                            reasoningTokenCount += 1
                        }
                        detokenizer.append(token: token)
                        if let chunk = detokenizer.next() {
                            for segment in emitter.process(chunk) {
                                await Self.send(
                                    segment, responseEntryID: responseEntryID,
                                    reasoningEntryID: reasoningEntryID, channel: channel)
                            }
                        }
                    }
                case .info(let info):
                    Self.observeCompletion(info)
                    completionInfo = info
                }
            }
        }

        let endedInsideReasoning: Bool
        if var decoder = protocolDecoder {
            endedInsideReasoning = decoder.isInsideReasoning
            _ = decoder.finish { event in
                _ = Self.consumeDecodedEvent(event, into: &decodedEvents)
                return true
            }
            for segment in decodedEvents.takeSegments() {
                await Self.send(
                    segment, responseEntryID: responseEntryID,
                    reasoningEntryID: reasoningEntryID, channel: channel)
            }
            protocolDecoder = decoder
            if let rejection = decodedEvents.rejectedToolCall {
                throw RejectedToolCallError(rejection)
            }
        } else {
            let pending = detokenizer.finish().map { emitter.process($0) } ?? []
            for segment in pending + emitter.finalize() {
                await Self.send(
                    segment, responseEntryID: responseEntryID,
                    reasoningEntryID: reasoningEntryID, channel: channel)
            }
            endedInsideReasoning = emitter.isInsideReasoning
        }

        // If generation ended while still inside a thinking block, the model
        // was cut off mid-thought (e.g. it exhausted the token budget before
        // emitting `</think>`). Signal it so a consumer doesn't mistake an
        // empty or partial answer for the model's chosen response — mirrors
        // the guided path's `incompleteOutput` convention.
        if endedInsideReasoning {
            await Self.emitIncompleteOutput(entryID: responseEntryID, into: channel)
        }

        if let info = completionInfo {
            // Single source of truth for usage: one authoritative
            // `.updateUsage` (the framework's aggregator replaces wholesale,
            // so we must not also rely on per-delta auto-summing). The
            // reasoning count is clamped to never exceed the total.
            await Self.emitUsage(
                input: Self.usageInput(
                    fedTokenCount: info.promptTokenCount, promptCache: promptCache),
                output: .init(
                    totalTokenCount: Self.generatedTokenCount(of: info),
                    reasoningTokenCount: min(reasoningTokenCount, info.generationTokenCount)),
                entryID: responseEntryID, into: channel)
        }
    }

    /// Routes one scanned segment to the appropriate channel entry.
    static func send(
        _ segment: ReasoningEventEmitter.Segment,
        responseEntryID: ResponseEntryID,
        reasoningEntryID: ReasoningEntryID,
        channel: LanguageModelExecutorGenerationChannel
    ) async {
        switch segment {
        case .reasoning(let text):
            await Self.emitReasoning(text, entryID: reasoningEntryID, into: channel)
        case .response(let text):
            await Self.emitResponse(text, entryID: responseEntryID, into: channel)
        }
    }

    /// Prepares an `LMInput` for the unconstrained reasoning path with
    /// thinking explicitly on, off, or unspecified. Maps the package-
    /// internal `cannotDisableReasoning` to the framework's
    /// `unsupportedCapability` so always-on models surface a typed error
    /// before generation rather than leaking `<think>` into `.response`.
    ///
    /// - Parameters:
    ///   - messages: The chat messages the prompt renders from
    ///   - config: The reasoning protocol of the model
    ///   - thinkingEnabled: `true` / `false` to force thinking on / off, `nil` for the default
    ///   - context: The loaded model context whose processor renders the prompt
    ///   - cannotDisableMessage: The description of the error for a model that always reasons
    /// - Returns: The render that made the prompt, and the prepared prompt
    /// - Throws: `unsupportedCapability` when thinking cannot go off, or the error of the processor
    static func preparedInput(
        messages: [Chat.Message],
        config: ReasoningConfig,
        thinkingEnabled: Bool?,
        context: ModelContext,
        cannotDisableMessage: String
    ) async throws -> (render: ThinkingRender, input: LMInput) {
        let render: ThinkingRender
        do {
            render = try ThinkingRender(config: config, thinkingEnabled: thinkingEnabled)
        } catch ReasoningError.cannotDisableReasoning {
            throw LanguageModelError.unsupportedCapability(
                LanguageModelError.UnsupportedCapability(
                    capability: .reasoning,
                    debugDescription: cannotDisableMessage))
        }
        return (
            render, try await render.prepare(messages: messages, tools: nil, context: context)
        )
    }

    /// The events of a token-stream decoder on the no-tools reasoning path.
    struct ReasoningDecodedEvents {
        /// The text segments that wait to go to the channel.
        private(set) var segments: [ReasoningEventEmitter.Segment] = []

        /// The first tool-call-shaped output, which this path cannot execute.
        var rejectedToolCall: RejectedToolCall?

        /// Whether generation goes on after these events.
        var shouldContinue = true

        /// Queues one text segment for the channel.
        mutating func append(_ segment: ReasoningEventEmitter.Segment) {
            segments.append(segment)
        }

        /// Gives the queued segments, and empties the queue.
        ///
        /// - Returns: The segments, in the order the decoder gave them.
        mutating func takeSegments() -> [ReasoningEventEmitter.Segment] {
            defer { segments.removeAll() }
            return segments
        }
    }

    /// Records one event of the no-tools reasoning path.
    ///
    /// A tool call stops nothing here: this path declares no tools, a protocol decoder then
    /// accepts a tool frame the model wrote, and the frame has nowhere to go. Model output
    /// reaches this, so it is logged rather than asserted.
    ///
    /// - Parameters:
    ///   - event: The event of the decoder.
    ///   - events: The record of the pass.
    /// - Returns: Whether the decoder goes on.
    static func consumeDecodedEvent(
        _ event: TokenStreamEvent, into events: inout ReasoningDecodedEvents
    ) -> Bool {
        switch event {
        case .reasoning(let text): events.append(.reasoning(text))
        case .response(let text): events.append(.response(text))
        case .toolCall(let call):
            protocolLogger.error(
                "Dropping a tool call to \(call.function.name) on a turn with no tools")
        case .rejectedToolCall(let rejection):
            events.rejectedToolCall = rejection
            events.shouldContinue = false
        case .protocolError(let message):
            protocolLogger.error("\(message)")
        case .stop:
            events.shouldContinue = false
        }
        return events.shouldContinue
    }

    /// Maps a requested reasoning level to a thinking on/off/unspecified
    /// flag. `nil` (no opinion) defers to the strategy's default; any
    /// concrete level means "think" (v1 does not modulate depth); only the
    /// package convention `.custom("no_think")` means "off".
    static func thinkingEnabled(for level: ContextOptions.ReasoningLevel?) -> Bool? {
        guard let level else { return nil }
        switch level {
        case .light, .moderate, .deep:
            return true
        case .custom(let value):
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            return normalized == "no_think" ? false : true
        @unknown default:
            // A future level we don't recognize → default to thinking on.
            assertionFailure("Unrecognized reasoning level: \(level)")
            logger.error("Unrecognized reasoning level \(level); thinking stays on")
            return true
        }
    }

    /// Decodes the rendered prompt's tail and asks whether it ends inside an
    /// open reasoning block (some model families prefill the opening
    /// delimiter).
    /// Build the Phase-2 continuation input: the tool-aware prompt with the
    /// completed reasoning token IDs appended along the sequence axis.
    ///
    /// The prompt tokens keep whatever rank the model's processor produced
    /// ([N] from LLM processors, [1, N] from VLM processors — VLM `prepare`
    /// requires the batched form), and processed image/video content is
    /// carried through so a VLM's Phase-2 prefill still sees its pixels.
    static func continuationInput(
        from input: LMInput, appending tokenIDs: [Int]
    ) -> LMInput {
        let promptTokens = input.text.tokens
        var appended = MLXArray(tokenIDs.map { Int32($0) })
            .asType(promptTokens.dtype)
        if promptTokens.ndim == LMInput.Text.batchedTokenRank {
            appended = appended[.newAxis, 0...]
        }
        return LMInput(
            text: .init(tokens: concatenated([promptTokens, appended], axis: -1)),
            image: input.image,
            video: input.video)
    }

    static func reasoningPrimedInside(
        input: LMInput, config: ReasoningConfig, tokenizer: any Tokenizer
    ) -> Bool {
        ReasoningEventEmitter.promptEndsInsideReasoning(
            promptTokens: input.text.tokens.asArray(Int.self), config: config,
            tokenizer: tokenizer)
    }

    /// Think-then-call Phase 1: generate reasoning unconstrained until
    /// the model closes its thinking block, routing reasoning text to
    /// `.reasoning` events and retaining the raw token IDs to prefill into the
    /// constrained Phase 2.
    ///
    /// Uses the `Task`-returning `generateTokensTask` so the GPU loop is
    /// cancelled and drained at the phase boundary — without that, Phase 2's
    /// prefill could overlap Phase 1's in-flight forward pass on the shared
    /// `Stream` and trip a Metal command-buffer assertion.
    ///
    /// Returns the accumulated token IDs and whether `</think>` actually
    /// closed. If it did not (budget exhausted mid-thought), the caller must
    /// skip Phase 2 rather than prefill a truncated thought into the grammar.
    func runToolCallReasoningPhase(
        input: LMInput,
        config: ReasoningConfig,
        primedInside: Bool,
        maxTokens: Int,
        requestedTemperature: Double?,
        samplingConfiguration: MLXSamplingConfiguration?,
        reasoningEntryID: ReasoningEntryID,
        responseEntryID: ResponseEntryID,
        speculation: MTPSpeculation?,
        context: ModelContext,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws -> (tokenIDs: [Int], closed: Bool) {
        let parameters = Self.makeParameters(
            maxTokens: maxTokens,
            requestedTemperature: requestedTemperature,
            samplingConfiguration: samplingConfiguration
        )
        var collector = ReasoningTokenCollector(
            config: config, primedInside: primedInside, tokenizer: context.tokenizer
        )

        // Phase 1 speculates from a cache of its own. Its drafter state ends
        // with its task: phase 2 feeds the prompt and the reasoning tokens to
        // the guided loop's own new cache and runs no drafter, thus it needs
        // none of the state this phase leaves.
        let (stream, task) = try Self.startTrackedGeneration {
            try generateTokensTask(
                input: input, parameters: parameters, context: context,
                speculation: speculation)
        }
        var closed = false
        do {
            for await generation in stream {
                try Task.checkCancellation()
                if case .info(let info) = generation {
                    Self.observeCompletion(info)
                }
                guard !closed, case .token(let token) = generation else { continue }
                for segment in collector.ingest(token) {
                    await Self.send(
                        segment, responseEntryID: responseEntryID,
                        reasoningEntryID: reasoningEntryID, channel: channel)
                }
                if collector.shouldStopAfterReasoning {
                    GuidedGenerationDiagnosticSink.current?.recordToolReasoningClose()
                    closed = true
                    // Stop the generation, and read the stream to its end, which
                    // reports the completion of this phase.
                    task.cancel()
                }
            }
        } catch {
            // Drain the generation task before propagating, but do NOT sync
            // here: respond's outer `catch` is the single GPU-sync point for
            // this exit path. Keep one clean GPU sync per exit path —
            // cascading syncs across nested catches can race the Metal
            // command-buffer state during teardown.
            await cancelAndDrain(task)
            throw error
        }
        // Drain the generation task before Phase 2 reuses the Stream.
        await cancelAndDrain(task)
        Stream.gpu.synchronize()

        for segment in collector.finalize() {
            await Self.send(
                segment, responseEntryID: responseEntryID,
                reasoningEntryID: reasoningEntryID, channel: channel)
        }
        return (collector.reasoningTokenIDs, closed)
    }

    /// Parses a required-mode tool-calling envelope JSON object and emits
    /// its developer tool call.
    ///
    /// The output buffer is expected to be a JSON object matching the
    /// shape `{"name": <tool-name>, "arguments": <args>}`. Grammars from
    /// `SchemaConverter.encodeToolCallingGrammar` guarantee either that
    /// shape directly (bare JSON) or that shape wrapped in Qwen's
    /// `<tool_call>\n...\n</tool_call>` special-token delimiters --
    /// `unwrapToolCallMarkers` below strips the wrapper if present. The
    /// guided path emits a single `.toolCallDelta` with the arguments JSON
    /// and a freshly minted toolCallID.
    ///
    /// Required mode never degrades malformed or partial output into a
    /// response event.
    ///
    /// - Parameters:
    ///   - outputBuffer: The output of the guided generation.
    ///   - toolCallsEntryID: The entry of the tool call.
    ///   - channel: The channel that receives the tool call.
    /// - Returns: `true` when the tool call was emitted, and `false` when the output was
    ///   dropped, so the caller can mark the entry as incomplete output.
    func emitRequiredToolCallEvent(
        outputBuffer: String,
        toolCallsEntryID: ToolCallsEntryID,
        channel: LanguageModelExecutorGenerationChannel
    ) async -> Bool {
        let unwrapped = Self.unwrapToolCallMarkers(outputBuffer)
        let data = Data(unwrapped.utf8)
        guard
            let envelope = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let name = envelope["name"] as? String
        else {
            GuidedGenerationDiagnosticSink.current?.recordParse(
                parsedAsToolCall: false, parsedName: nil)
            Self.logger.error(
                "Dropping a required tool call: \(outputBuffer.utf8.count) bytes of output are no tool-call envelope"
            )
            return false
        }

        GuidedGenerationDiagnosticSink.current?.recordParse(
            parsedAsToolCall: true, parsedName: name)

        // The arguments of a tool whose schema is a scalar -- a string, a
        // number, a Boolean -- are not an object. Without
        // `.fragmentsAllowed`, `JSONSerialization` raises an Objective-C
        // exception for such a value, which `try?` cannot catch, and the
        // process stops.
        guard
            let arguments = envelope["arguments"],
            let argumentsData = try? JSONSerialization.data(
                withJSONObject: arguments, options: [.fragmentsAllowed]),
            let argumentsJSON = String(data: argumentsData, encoding: .utf8)
        else {
            assertionFailure("tool-call envelope for \(name) has no usable arguments")
            Self.logger.error("Dropping a tool call to \(name): no usable arguments")
            return false
        }
        await Self.emitToolCall(
            id: UUID().uuidString,
            name: name,
            arguments: argumentsJSON,
            entryID: toolCallsEntryID,
            into: channel)
        return true
    }

    /// Strips Qwen-style `<tool_call>\n...\n</tool_call>` wrapper markers
    /// if present, returning the inner JSON text. Untouched if the buffer
    /// doesn't start with a wrapper -- the `bare_call` grammar alternative
    /// is valid output and parses directly.
    ///
    /// The inner newlines around the JSON come from the Qwen training
    /// format; we're tolerant of whitespace on either side of the markers
    /// so that tokenizer decoding quirks (extra spaces, missing newlines)
    /// don't cause the JSON parse to fail.
    static func unwrapToolCallMarkers(_ buffer: String) -> String {
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        let openMarker = "<tool_call>"
        let closeMarker = "</tool_call>"
        guard trimmed.hasPrefix(openMarker) else { return buffer }
        let afterOpen = trimmed.dropFirst(openMarker.count)
        let inner: Substring
        if let closeRange = afterOpen.range(of: closeMarker, options: .backwards) {
            inner = afterOpen[afterOpen.startIndex ..< closeRange.lowerBound]
        } else {
            inner = afterOpen
        }
        return inner.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel.Executor.EntryID {
    /// An identifier for a new entry: a fresh UUID.
    init() { self.init(rawValue: UUID().uuidString) }
}
