// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift @ 4546a5d720e7013adffdbddd728c6106e4f9e637
// Manual transcription; no git ancestry.
//
// DeepSeek-V4 ships NO `chat_template`. The published `tokenizer_config.json`
// and `tokenizer.json` of `deepseek-ai/DeepSeek-V4-Flash` @
// 60d8d70770c6776ff598c94bb586a859a38244f1 have no such key, thus
// `Tokenizer.applyChatTemplate` cannot make a prompt for this model. DeepSeek
// distributes the prompt builder as a Python file in the same repository, at
// `encoding/encoding_dsv4.py` (27908 bytes, sha256
// bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0). This file
// is a transcription of that Python file, and it is the only path from a loaded
// DeepSeek-V4 model to a usable prompt.
//
// WHERE EACH LITERAL COMES FROM
//
// Each token string below is the value of the same name in
// `encoding/encoding_dsv4.py`, and each one is also an entry of the
// `added_tokens` array of `tokenizer.json` in the same model repository. Thus
// the reference and the published tokenizer agree, and each marker is one
// token that a detokenizer never splits:
//
//   Python name              tokenizer.json id   `special`
//   bos_token                0                   true
//   eos_token                1                   true
//   USER_SP_TOKEN            128803              false
//   ASSISTANT_SP_TOKEN       128804              false
//   thinking_start_token     128821              false
//   thinking_end_token       128822              false
//   LATEST_REMINDER_SP_TOKEN 128828              false
//   DS_TASK_SP_TOKENS         128829…128845      false
//
// The delimiter inside every marker is FULLWIDTH VERTICAL LINE U+FF5C, and the
// two separators inside the sentence markers are LOWER ONE EIGHTH BLOCK U+2581.
// Neither is the ASCII `|` or `_` that they look like.
//
// ONE DISAGREEMENT WITH THE CARD, RECORDED HERE AND ON THE TASK
//
// The task card says that `eos_token_id == 1` comes from
// `tokenizer_config.json`. That file holds no `eos_token_id` key at all. It
// holds `eos_token`, whose `content` is the string below, and
// `generation_config.json` holds `eos_token_id: 1`. `Tokenizer.eosTokenId` in
// this repository reads the `eos_token` string and looks it up in the
// vocabulary, where `tokenizer.json` gives it the id 1. Thus the value the card
// asks for is correct, and its stated source is not. This file follows the
// tokenizer.
//
// THE TOOL HALF
//
// The DSML tool format is the second half of the reference. `｜DSML｜` is one
// more marker built from the same FULLWIDTH VERTICAL LINE, token id 128825, and
// it carries NO angle brackets of its own — each template puts the brackets
// around it. A tool result is plain ASCII, `<tool_result>…</tool_result>`.
//
// Tool schemas and the arguments of a call go through `PythonStyleJSON`, which
// keeps the order of the members of a JSON object and leaves the solidus alone.
// `JSONSerialization` does neither, thus it cannot make the published bytes.
//
// TWO PLACES THE PYTHON HAS NO DEFINED BEHAVIOUR
//
// 1. Arguments that parse to a JSON value which is not an object make the
//    Python raise `AttributeError` from `.items()`. This file puts them through
//    the same one-parameter path the Python uses for arguments that do not
//    parse at all.
// 2. `render_message` raises `NotImplementedError` for the `tool` role, which
//    it can reach only when a caller skips `merge_tool_messages`.
//    ``encode(messages:thinkingMode:reasoningEffort:dropsEarlierReasoning:context:addsBeginOfSentence:)``
//    always merges first, thus a `tool` turn never reaches the renderer. The
//    switch stays exhaustive and renders such a turn the way the merge would.
//
// TWO PLACES THIS FILE WRITES THE TURN THE MODEL WROTE
//
// The Python reads an assistant turn from an API caller, thus its `reasoning`
// and its `content` are always two separate fields and its `content` never
// holds the text of a tool call. A turn that comes back from a LIVE generation
// is not shaped that way, because the model writes one stream of text and
// `MLXLMCommon` keeps the whole of it as the content:
//
// 1. DeepSeek-V4 declares no `reasoningConfig`, thus no decoder splits the
//    reasoning out and the content holds `reasoning` + `</think>` + the answer.
//    ``holdsItsOwnReasoning(_:)`` finds such a content and writes no second
//    close in front of it.
// 2. The model writes a blank line between its answer and its block of calls,
//    and the tool-call reader keeps that blank line with the answer.
//    ``contentBeforeToolCalls(of:)`` takes those newlines away, because
//    ``toolCallsBlock(_:)`` writes the same blank line again.
//
// Both rules keep the render of a turn equal to the text the model wrote. A
// live prompt cache needs that: its token ledger is the render of the round
// before PLUS the tokens the model wrote, thus a render that writes those
// tokens differently makes the whole cache useless. Card ^v7z7v99 measured
// that loss on the published checkpoint, and
// `Tests/MLXLMTests/DeepSeekV4ToolEncodingTests.swift` holds the rule.

/// Builds a DeepSeek-V4 prompt from a conversation.
///
/// DeepSeek-V4 ships no chat template, so this encoder, and not
/// `Tokenizer.applyChatTemplate`, makes the prompt for that model family.
///
/// ```swift
/// let encoder = DeepSeekV4ChatEncoder()
/// let prompt = encoder.encode(
///     messages: [.system(content: "You are helpful."), .user(content: "Hello")],
///     thinkingMode: .thinking)
/// ```
public struct DeepSeekV4ChatEncoder: Sendable {

    /// Creates an encoder. The encoder holds no state.
    public init() {}

    // MARK: - Types

    /// Which of the two DeepSeek-V4 generation modes the prompt asks for.
    public enum ThinkingMode: String, Sendable {
        /// The model answers directly, after a closed `</think>` tail.
        case chat
        /// The model reasons first, after an open `<think>` tail.
        case thinking
    }

    /// How much deliberation the prompt asks the model for.
    public enum ReasoningEffort: String, Sendable {
        /// The default amount of deliberation. Adds no preface.
        case high
        /// The largest amount of deliberation. Adds ``reasoningEffortMaxPreface``.
        case max
    }

    /// The role of one message in a DeepSeek-V4 conversation.
    public enum Role: String, Sendable {
        /// The system prompt. Renders with no marker of its own.
        case system
        /// A developer instruction. Renders behind the user marker.
        case developer
        /// A user turn.
        case user
        /// An assistant turn.
        case assistant
        /// The answer of one tool. It never renders on its own: the encoder
        /// folds it into the user turn before it as a `<tool_result>` block.
        case tool
        /// A reminder that the caller puts in front of the last user turn.
        case latestReminder = "latest_reminder"
    }

    /// One tool that the prompt offers the model.
    ///
    /// The reference reads an OpenAI tool list and keeps the `function` member
    /// of each entry, thus this type carries that member and nothing around it.
    public struct Tool: Sendable {
        /// The `function` object of an OpenAI tool definition, as JSON text.
        ///
        /// The encoder writes it again on one line, in the order of the members
        /// of this text. Text that is not JSON goes into the prompt unchanged.
        public var functionSchemaJSON: String

        /// Creates a tool.
        /// - Parameter functionSchemaJSON: the JSON text of the `function`
        ///   object of an OpenAI tool definition.
        public init(functionSchemaJSON: String) {
            self.functionSchemaJSON = functionSchemaJSON
        }
    }

    /// One call that an assistant turn makes to a tool.
    public struct ToolCall: Sendable {
        /// The identifier that ties this call to the answer that comes back.
        ///
        /// An empty identifier takes no part in the ordering rule of
        /// ``DeepSeekV4ChatEncoder``.
        public var id: String
        /// The name of the tool to call.
        public var name: String
        /// The arguments of the call, as JSON text.
        ///
        /// The encoder makes one DSML parameter for each member of this object,
        /// in the order of the text. Text that is not a JSON object becomes one
        /// parameter named `arguments`, exactly as the reference does.
        public var argumentsJSON: String

        /// Creates a tool call.
        /// - Parameters:
        ///   - id: the identifier that ties the call to its answer.
        ///   - name: the name of the tool to call.
        ///   - argumentsJSON: the arguments of the call, as JSON text.
        public init(id: String = "", name: String, argumentsJSON: String) {
            self.id = id
            self.name = name
            self.argumentsJSON = argumentsJSON
        }
    }

    /// One piece of a user turn.
    ///
    /// The reference builds this list in `merge_tool_messages`, thus a caller
    /// never writes one: it sends ``Message/user(content:task:)`` and
    /// ``Message/toolResult(content:toolCallID:)`` turns, and the encoder folds
    /// them.
    enum ContentBlock: Equatable, Sendable {
        /// The text of a user turn.
        case text(String)
        /// The answer of one tool.
        case toolResult(toolUseID: String, content: String)

        /// Whether this block is the answer of a tool.
        var isToolResult: Bool {
            guard case .toolResult = self else { return false }
            return true
        }

        /// The identifier of the call this block answers, or `nil` for text.
        var toolUseID: String? {
            guard case .toolResult(let identifier, _) = self else { return nil }
            return identifier
        }
    }

    /// One of the internal classification tasks that DeepSeek-V4 answers with a
    /// task token in place of an ordinary assistant turn.
    public enum QuickInstructionTask: String, Sendable, CaseIterable {
        /// Choose the next action.
        case action
        /// Write a search query.
        case query
        /// Name the authority of a source.
        case authority
        /// Name the domain of a source.
        case domain
        /// Write the title of a source.
        case title
        /// Read a URL.
        case readURL = "read_url"
    }

    /// One message of a DeepSeek-V4 conversation.
    ///
    /// Build a message with one of the role factories —
    /// ``system(content:tools:responseFormatJSON:)``, ``user(content:task:)``,
    /// ``developer(content:tools:responseFormatJSON:)``,
    /// ``latestReminder(content:)``,
    /// ``assistant(content:reasoning:toolCalls:endsWithEndOfSentence:)`` or
    /// ``toolResult(content:toolCallID:)``.
    public struct Message: Sendable {
        /// The role of the sender.
        public var role: Role
        /// The text of the message.
        public var content: String
        /// The chain of thought of an assistant turn.
        ///
        /// Only an assistant message reads this value. ``ThinkingMode/chat``
        /// never renders it, and ``encode(messages:thinkingMode:reasoningEffort:dropsEarlierReasoning:context:addsBeginOfSentence:)``
        /// removes it from every turn before the last user turn when
        /// `dropsEarlierReasoning` is `true`.
        public var reasoning: String?
        /// The classification task that this message asks the model to answer.
        ///
        /// A message with a task renders a task token in place of the ordinary
        /// generation tail.
        public var task: QuickInstructionTask?
        /// Whether the message ends with the end-of-sentence marker.
        ///
        /// Only an assistant message reads this value. `false` leaves the turn
        /// open, which primes the model to continue that same turn.
        public var endsWithEndOfSentence: Bool
        /// The tools that the prompt offers the model.
        ///
        /// Only a system or a developer message renders them. One tool anywhere
        /// in the conversation also turns the `dropsEarlierReasoning` rule off,
        /// because a mid-trajectory agent turn needs its whole chain of
        /// thought.
        public var tools: [Tool]
        /// The schema that the answer must follow, as JSON text.
        ///
        /// Only a system or a developer message renders it, after the tools.
        public var responseFormatJSON: String?
        /// The calls that an assistant turn makes to its tools.
        public var toolCalls: [ToolCall]
        /// The identifier of the call that a tool turn answers.
        public var toolCallID: String
        /// The pieces of a user turn, or `nil` before the merge step builds
        /// them.
        ///
        /// `merge_tool_messages` in the reference owns this list, thus the
        /// encoder fills it and a caller never does.
        var contentBlocks: [ContentBlock]?

        /// Creates a message.
        ///
        /// - Parameters:
        ///   - role: the role of the sender.
        ///   - content: the text of the message.
        ///   - reasoning: the chain of thought of an assistant turn.
        ///   - task: the classification task this message asks for.
        ///   - endsWithEndOfSentence: whether an assistant turn is closed.
        ///   - tools: the tools the prompt offers the model.
        ///   - responseFormatJSON: the schema the answer must follow.
        ///   - toolCalls: the calls that an assistant turn makes.
        ///   - toolCallID: the call that a tool turn answers.
        public init(
            role: Role,
            content: String,
            reasoning: String? = nil,
            task: QuickInstructionTask? = nil,
            endsWithEndOfSentence: Bool = true,
            tools: [Tool] = [],
            responseFormatJSON: String? = nil,
            toolCalls: [ToolCall] = [],
            toolCallID: String = ""
        ) {
            self.role = role
            self.content = content
            self.reasoning = reasoning
            self.task = task
            self.endsWithEndOfSentence = endsWithEndOfSentence
            self.tools = tools
            self.responseFormatJSON = responseFormatJSON
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
        }

        /// A system message.
        /// - Parameters:
        ///   - content: the text of the system prompt.
        ///   - tools: the tools the prompt offers the model.
        ///   - responseFormatJSON: the schema the answer must follow.
        /// - Returns: the message.
        public static func system(
            content: String, tools: [Tool] = [], responseFormatJSON: String? = nil
        ) -> Self {
            Self(
                role: .system, content: content, tools: tools,
                responseFormatJSON: responseFormatJSON)
        }

        /// A developer message.
        /// - Parameters:
        ///   - content: the text of the instruction.
        ///   - tools: the tools the prompt offers the model.
        ///   - responseFormatJSON: the schema the answer must follow.
        /// - Returns: the message.
        public static func developer(
            content: String, tools: [Tool] = [], responseFormatJSON: String? = nil
        ) -> Self {
            Self(
                role: .developer, content: content, tools: tools,
                responseFormatJSON: responseFormatJSON)
        }

        /// A user message.
        /// - Parameters:
        ///   - content: the text of the turn.
        ///   - task: the classification task this turn asks for.
        /// - Returns: the message.
        public static func user(content: String, task: QuickInstructionTask? = nil) -> Self {
            Self(role: .user, content: content, task: task)
        }

        /// A reminder message that renders in front of the last user turn.
        /// - Parameter content: the text of the reminder.
        /// - Returns: the message.
        public static func latestReminder(content: String) -> Self {
            Self(role: .latestReminder, content: content)
        }

        /// An assistant message.
        /// - Parameters:
        ///   - content: the answer of the turn.
        ///   - reasoning: the chain of thought of the turn.
        ///   - toolCalls: the calls that the turn makes to its tools.
        ///   - endsWithEndOfSentence: whether the turn is closed.
        /// - Returns: the message.
        public static func assistant(
            content: String,
            reasoning: String? = nil,
            toolCalls: [ToolCall] = [],
            endsWithEndOfSentence: Bool = true
        ) -> Self {
            Self(
                role: .assistant, content: content, reasoning: reasoning,
                endsWithEndOfSentence: endsWithEndOfSentence, toolCalls: toolCalls)
        }

        /// The answer of one tool.
        ///
        /// The encoder folds it into the user turn before it, thus it never
        /// renders a marker of its own.
        ///
        /// - Parameters:
        ///   - content: the text that the tool gave back.
        ///   - toolCallID: the call that this answer belongs to.
        /// - Returns: the message.
        public static func toolResult(content: String, toolCallID: String = "") -> Self {
            Self(role: .tool, content: content, toolCallID: toolCallID)
        }
    }

    // MARK: - Special tokens

    /// The literal markers that a DeepSeek-V4 prompt is made of.
    ///
    /// Each one is a single entry of the `added_tokens` array of the published
    /// `tokenizer.json`, thus each one survives detokenization unsplit.
    public enum SpecialToken {
        /// The delimiter inside each marker: FULLWIDTH VERTICAL LINE U+FF5C.
        private static let delimiter = "\u{FF5C}"
        /// The separator inside the sentence markers: LOWER ONE EIGHTH BLOCK
        /// U+2581.
        private static let wordSeparator = "\u{2581}"

        /// Wraps a name in the two delimiters and the angle brackets.
        /// - Parameter name: the name between the delimiters.
        /// - Returns: the marker.
        private static func marker(_ name: String) -> String {
            "<" + delimiter + name + delimiter + ">"
        }

        /// The marker that opens a conversation. Token id 0.
        public static let beginOfSentence = marker(
            "begin" + wordSeparator + "of" + wordSeparator + "sentence")
        /// The marker that closes an assistant turn. Token id 1.
        public static let endOfSentence = marker(
            "end" + wordSeparator + "of" + wordSeparator + "sentence")
        /// The marker that opens a user turn.
        public static let user = marker("User")
        /// The marker that opens an assistant turn.
        public static let assistant = marker("Assistant")
        /// The marker that opens a reminder turn.
        public static let latestReminder = marker("latest_reminder")
        /// The marker that opens a reasoning block.
        public static let thinkStart = "<think>"
        /// The marker that closes a reasoning block.
        public static let thinkEnd = "</think>"
        /// The marker that opens and closes every DSML tag. Token id 128825.
        ///
        /// It carries no angle brackets of its own: each template puts them
        /// around it, thus a block of calls opens `<｜DSML｜tool_calls>`.
        public static let dsml = delimiter + "DSML" + delimiter

        /// The marker of one classification task.
        /// - Parameter task: the task to name.
        /// - Returns: the marker.
        public static func task(_ task: QuickInstructionTask) -> String {
            marker(task.rawValue)
        }

        /// Every marker that ``DeepSeekV4ChatEncoder`` writes.
        ///
        /// ``DeepSeekV4Tokenization`` reads this list: each of these markers
        /// is one token of the published vocabulary, thus each one stands
        /// outside the pre-tokenizer.
        public static var allMarkers: [String] {
            [
                beginOfSentence, endOfSentence, user, assistant, latestReminder,
                thinkStart, thinkEnd, dsml,
            ] + QuickInstructionTask.allCases.map(task)
        }
    }

    /// The preface that ``ReasoningEffort/max`` puts in front of the first
    /// message.
    ///
    /// This is a transcription of `REASONING_EFFORT_MAX` in
    /// `encoding/encoding_dsv4.py`. The model saw these words during training,
    /// thus a change to them shifts the distribution of the prompt.
    public static let reasoningEffortMaxPreface = """
        Reasoning Effort: Absolute maximum with no shortcuts permitted.
        You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
        Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.


        """

    // MARK: - Encoding

    /// Renders a conversation into a DeepSeek-V4 prompt.
    ///
    /// The result ends with the generation tail — the assistant marker and
    /// either `<think>` or `</think>` — when the last message is a user or a
    /// developer message. It ends with a task marker when that message names a
    /// task, and it ends with the message itself when the last message is an
    /// assistant turn.
    ///
    /// - Parameters:
    ///   - messages: the turns to render.
    ///   - thinkingMode: which of the two generation modes to prime.
    ///   - reasoningEffort: how much deliberation to ask for. Only
    ///     ``ReasoningEffort/max`` in ``ThinkingMode/thinking`` changes the
    ///     prompt.
    ///   - dropsEarlierReasoning: whether to remove the chain of thought of
    ///     every turn before the last user turn.
    ///   - context: turns that a cached prefix already holds. They take part in
    ///     the indexing rules but they are not in the result.
    ///   - addsBeginOfSentence: whether to open the prompt with
    ///     ``SpecialToken/beginOfSentence``. A non-empty `context` never opens
    ///     the prompt again.
    /// - Returns: the prompt.
    public func encode(
        messages: [Message],
        thinkingMode: ThinkingMode,
        reasoningEffort: ReasoningEffort? = nil,
        dropsEarlierReasoning: Bool = true,
        context: [Message] = [],
        addsBeginOfSentence: Bool = true
    ) -> String {
        // The reference sorts the tool answers of the whole conversation, the
        // unmerged context included, and then keeps only the tail that the
        // caller asked for.
        let renderedMessages = Array(
            Self.sortingToolResultsByCallOrder(context + Self.mergingToolMessages(messages))
                .dropFirst(context.count))
        let renderedContext: [Message] =
            context.isEmpty
            ? [] : Self.sortingToolResultsByCallOrder(Self.mergingToolMessages(context))
        var prompt =
            (addsBeginOfSentence && renderedContext.isEmpty)
            ? SpecialToken.beginOfSentence : ""

        // One tool anywhere in the conversation turns the drop rule off,
        // because a mid-trajectory agent turn needs its whole chain of thought.
        let full = renderedContext + renderedMessages
        let dropsReasoning = dropsEarlierReasoning && full.allSatisfy { $0.tools.isEmpty }
        let turns: [Message]
        let renderedCount: Int
        if thinkingMode == .thinking && dropsReasoning {
            turns = Self.removingEarlierReasoning(full)
            renderedCount = turns.count - Self.removingEarlierReasoning(renderedContext).count
        } else {
            turns = full
            renderedCount = renderedMessages.count
        }
        let contextCount = turns.count - renderedCount

        for offset in 0 ..< renderedCount {
            prompt += Self.render(
                at: offset + contextCount, in: turns, thinkingMode: thinkingMode,
                dropsReasoning: dropsReasoning, reasoningEffort: reasoningEffort)
        }
        return prompt
    }

    // MARK: - One message

    /// Renders the message at one index, with the generation tail that follows
    /// it.
    ///
    /// - Parameters:
    ///   - index: which message to render.
    ///   - messages: every turn, context first.
    ///   - thinkingMode: which of the two generation modes to prime.
    ///   - dropsReasoning: whether earlier reasoning is gone.
    ///   - reasoningEffort: how much deliberation to ask for.
    /// - Returns: the rendering of that one message.
    private static func render(
        at index: Int,
        in messages: [Message],
        thinkingMode: ThinkingMode,
        dropsReasoning: Bool,
        reasoningEffort: ReasoningEffort?
    ) -> String {
        let message = messages[index]
        let lastUserIndex = lastUserOrDeveloperIndex(in: messages)
        var out = ""

        if index == 0 && thinkingMode == .thinking && reasoningEffort == .max {
            out += reasoningEffortMaxPreface
        }
        out += body(
            of: message, precededByTask: index > 0 && messages[index - 1].task != nil,
            isAfterLastUser: index > lastUserIndex, thinkingMode: thinkingMode,
            dropsReasoning: dropsReasoning)

        // The tail belongs to the last message of a user or developer turn. A
        // reminder counts as part of that turn, thus it does not end it.
        let nextRole: Role? = index + 1 < messages.count ? messages[index + 1].role : nil
        if let nextRole, nextRole != .assistant, nextRole != .latestReminder {
            return out
        }
        out += tail(
            of: message, isAtOrAfterLastUser: index >= lastUserIndex,
            thinkingMode: thinkingMode, dropsReasoning: dropsReasoning)
        return out
    }

    /// Renders the message itself, without the generation tail.
    ///
    /// - Parameters:
    ///   - message: the message to render.
    ///   - precededByTask: whether the message before this one names a task,
    ///     which makes an assistant turn a task answer with no reasoning block.
    ///   - isAfterLastUser: whether this message follows the last user turn.
    ///   - thinkingMode: which of the two generation modes to prime.
    ///   - dropsReasoning: whether earlier reasoning is gone.
    /// - Returns: the rendering of the message.
    private static func body(
        of message: Message,
        precededByTask: Bool,
        isAfterLastUser: Bool,
        thinkingMode: ThinkingMode,
        dropsReasoning: Bool
    ) -> String {
        switch message.role {
        case .system:
            return message.content + toolsAndResponseFormat(of: message)
        case .developer:
            return SpecialToken.user + message.content + toolsAndResponseFormat(of: message)
        case .user:
            return SpecialToken.user + userBody(of: message)
        case .tool:
            // The merge step turns every tool turn into a user turn, thus this
            // arm renders what that merge would have rendered.
            return SpecialToken.user + toolResultBlock(message.content)
        case .latestReminder:
            return SpecialToken.latestReminder + message.content
        case .assistant:
            var out = ""
            if thinkingMode == .thinking && !precededByTask
                && (!dropsReasoning || isAfterLastUser)
                && !holdsItsOwnReasoning(message)
            {
                out += (message.reasoning ?? "") + SpecialToken.thinkEnd
            }
            out += contentBeforeToolCalls(of: message) + toolCallsBlock(message.toolCalls)
            if message.endsWithEndOfSentence {
                out += SpecialToken.endOfSentence
            }
            return out
        }
    }

    /// Tells whether the content of an assistant turn already holds its own
    /// closed reasoning block.
    ///
    /// `ChatSession` keeps the whole generated text of a turn as the content of
    /// that turn, because DeepSeek-V4 declares no `reasoningConfig` and thus no
    /// decoder splits the reasoning out of the token stream. Such a content
    /// already holds ``SpecialToken/thinkEnd``, thus one more in front of it
    /// makes a turn the model never wrote.
    ///
    /// - Parameter message: the assistant turn to read.
    /// - Returns: `true` when the turn carries no separate reasoning and its
    ///   content holds the close of a reasoning block.
    private static func holdsItsOwnReasoning(_ message: Message) -> Bool {
        message.reasoning == nil && message.content.contains(SpecialToken.thinkEnd)
    }

    /// The content of an assistant turn, without the newlines that belong to
    /// the block of calls after it.
    ///
    /// The model writes a blank line between its answer and its block of calls.
    /// The tool-call reader keeps that blank line with the answer, because the
    /// block leaves the text stream as a structured call, thus the content of
    /// the turn ends with the same two newlines that ``toolCallsBlock(_:)``
    /// writes. Together they make four newlines where the model wrote two.
    ///
    /// - Parameter message: the assistant turn to read.
    /// - Returns: the content to render in front of the block of calls.
    private static func contentBeforeToolCalls(of message: Message) -> String {
        guard !message.toolCalls.isEmpty else { return message.content }
        var content = message.content
        while content.hasSuffix("\n") {
            content.removeLast()
        }
        return content
    }

    /// Renders the tail that primes the model to generate.
    ///
    /// - Parameters:
    ///   - message: the last message of the turn.
    ///   - isAtOrAfterLastUser: whether this message is the last user turn or
    ///     follows it.
    ///   - thinkingMode: which of the two generation modes to prime.
    ///   - dropsReasoning: whether earlier reasoning is gone.
    /// - Returns: the tail, which is empty after an assistant or a system turn
    ///   that names no task.
    private static func tail(
        of message: Message,
        isAtOrAfterLastUser: Bool,
        thinkingMode: ThinkingMode,
        dropsReasoning: Bool
    ) -> String {
        if let task = message.task {
            // Only `action` asks the model for an assistant turn. The other
            // tasks read their answer straight after the task marker.
            guard task == .action else { return SpecialToken.task(task) }
            let thinkTail =
                thinkingMode == .thinking ? SpecialToken.thinkStart : SpecialToken.thinkEnd
            return SpecialToken.assistant + thinkTail + SpecialToken.task(task)
        }
        guard message.role == .user || message.role == .developer else { return "" }
        let opensReasoning =
            thinkingMode == .thinking && (!dropsReasoning || isAtOrAfterLastUser)
        return SpecialToken.assistant
            + (opensReasoning ? SpecialToken.thinkStart : SpecialToken.thinkEnd)
    }

    // MARK: - Multi-turn rules

    /// Removes the reasoning that a new turn must not see again.
    ///
    /// An assistant turn before the last user turn keeps its answer and loses
    /// its chain of thought. A developer turn before the last user turn goes
    /// away whole. A system, user or reminder turn always stays.
    ///
    /// - Parameter messages: the turns to filter.
    /// - Returns: the turns that the prompt renders.
    private static func removingEarlierReasoning(_ messages: [Message]) -> [Message] {
        let lastUserIndex = lastUserOrDeveloperIndex(in: messages)
        var result: [Message] = []
        for (index, message) in messages.enumerated() {
            let alwaysKept =
                message.role == .user || message.role == .system
                || message.role == .tool || message.role == .latestReminder
            if alwaysKept || index >= lastUserIndex {
                result.append(message)
            } else if message.role == .assistant {
                var withoutReasoning = message
                withoutReasoning.reasoning = nil
                result.append(withoutReasoning)
            }
        }
        return result
    }

    /// Folds every tool turn, and every user turn that follows another, into
    /// one user turn made of content blocks.
    ///
    /// The reference builds that block list in `merge_tool_messages`. Text
    /// blocks join with a blank line, which gives one user marker for a whole
    /// run of user turns. A turn that names a task never absorbs the text turn
    /// after it, but it does absorb a tool answer.
    ///
    /// - Parameter messages: the turns to merge.
    /// - Returns: the merged turns.
    private static func mergingToolMessages(_ messages: [Message]) -> [Message] {
        var merged: [Message] = []
        for message in messages {
            switch message.role {
            case .tool:
                append(
                    .toolResult(toolUseID: message.toolCallID, content: message.content),
                    foldingIntoTaskTurn: true,
                    otherwiseStarting: Message(role: .user, content: ""), to: &merged)
            case .user:
                append(
                    .text(message.content), foldingIntoTaskTurn: false,
                    otherwiseStarting: message, to: &merged)
            case .system, .developer, .assistant, .latestReminder:
                merged.append(message)
            }
        }
        return merged
    }

    /// Adds one block to the user turn at the end of the list, or starts a new
    /// user turn to hold it.
    ///
    /// - Parameters:
    ///   - block: the block to add.
    ///   - foldingIntoTaskTurn: whether the block may join a user turn that
    ///     names a task. Only a tool answer may.
    ///   - newTurn: the turn to start when the block cannot join.
    ///   - merged: the turns built so far.
    private static func append(
        _ block: ContentBlock,
        foldingIntoTaskTurn: Bool,
        otherwiseStarting newTurn: Message,
        to merged: inout [Message]
    ) {
        if var last = merged.last, last.role == .user, last.contentBlocks != nil,
            foldingIntoTaskTurn || last.task == nil
        {
            last.contentBlocks?.append(block)
            merged[merged.count - 1] = last
            return
        }
        var turn = newTurn
        turn.contentBlocks = [block]
        merged.append(turn)
    }

    /// Puts the tool answers of each user turn back into the order of the calls
    /// that asked for them.
    ///
    /// A tool answers when it is ready, thus the order the answers come back in
    /// is not the order the assistant asked in. The reference sorts by the
    /// position of the matching call in the assistant turn before, and an
    /// answer whose identifier no call names keeps the place it had.
    ///
    /// - Parameter messages: the merged turns.
    /// - Returns: the turns, with each block list sorted.
    private static func sortingToolResultsByCallOrder(_ messages: [Message]) -> [Message] {
        var callOrder: [String: Int] = [:]
        var result = messages
        for index in result.indices {
            let message = result[index]
            if message.role == .assistant, !message.toolCalls.isEmpty {
                callOrder = [:]
                for (position, call) in message.toolCalls.enumerated() where !call.id.isEmpty {
                    callOrder[call.id] = position
                }
                continue
            }
            guard message.role == .user, let blocks = message.contentBlocks else { continue }
            result[index].contentBlocks = reordering(blocks, by: callOrder)
        }
        return result
    }

    /// Sorts the tool answers of one block list and leaves every other block
    /// where it is.
    ///
    /// - Parameters:
    ///   - blocks: the blocks of one user turn.
    ///   - callOrder: the place of each call, by identifier.
    /// - Returns: the blocks, sorted.
    private static func reordering(
        _ blocks: [ContentBlock], by callOrder: [String: Int]
    ) -> [ContentBlock] {
        let answers = blocks.filter(\.isToolResult)
        guard answers.count > 1, !callOrder.isEmpty else { return blocks }
        let sorted =
            answers.enumerated()
            .sorted { left, right in
                let leftOrder = callOrder[left.element.toolUseID ?? ""] ?? 0
                let rightOrder = callOrder[right.element.toolUseID ?? ""] ?? 0
                return leftOrder == rightOrder
                    ? left.offset < right.offset : leftOrder < rightOrder
            }
            .map(\.element)

        var result: [ContentBlock] = []
        var next = 0
        for block in blocks {
            guard block.isToolResult else {
                result.append(block)
                continue
            }
            result.append(sorted[next])
            next += 1
        }
        return result
    }

    // MARK: - The tool half

    /// The tools and the response format that a system or a developer message
    /// carries, each after a blank line.
    ///
    /// - Parameter message: the message to read.
    /// - Returns: the text to put after that message, empty when it carries
    ///   neither.
    private static func toolsAndResponseFormat(of message: Message) -> String {
        var out = ""
        if !message.tools.isEmpty {
            out += "\n\n" + renderedTools(message.tools)
        }
        if let schema = message.responseFormatJSON {
            out += "\n\n" + responseFormatPreface + pythonStyleJSON(schema)
        }
        return out
    }

    /// The words that introduce a response format schema.
    private static let responseFormatPreface = """
        ## Response Format:

        You MUST strictly adhere to the following schema to reply:

        """

    /// The `## Tools` section that a message carrying tools renders.
    ///
    /// This is a transcription of `TOOLS_TEMPLATE` in
    /// `encoding/encoding_dsv4.py`. The model saw these words during training,
    /// thus a change to them shifts the distribution of the prompt.
    ///
    /// - Parameter tools: the tools to offer the model.
    /// - Returns: the section.
    private static func renderedTools(_ tools: [Tool]) -> String {
        let schemas = tools.map { pythonStyleJSON($0.functionSchemaJSON) }.joined(separator: "\n")
        return """
            ## Tools

            You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(SpecialToken.dsml)tool_calls>" block like the following:

            <\(SpecialToken.dsml)tool_calls>
            <\(SpecialToken.dsml)invoke name="$TOOL_NAME">
            <\(SpecialToken.dsml)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(SpecialToken.dsml)parameter>
            ...
            </\(SpecialToken.dsml)invoke>
            <\(SpecialToken.dsml)invoke name="$TOOL_NAME2">
            ...
            </\(SpecialToken.dsml)invoke>
            </\(SpecialToken.dsml)tool_calls>

            String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

            If thinking_mode is enabled (triggered by \(SpecialToken.thinkStart)), you MUST output your complete reasoning inside \(SpecialToken.thinkStart)...\(SpecialToken.thinkEnd) BEFORE any tool calls or final response.

            Otherwise, output directly after \(SpecialToken.thinkEnd) with tool calls or final response.

            ### Available Tool Schemas

            \(schemas)

            You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.

            """
    }

    /// The name of the DSML tag that wraps the calls of one turn.
    private static let toolCallsBlockName = "tool_calls"

    /// The DSML block of calls that closes an assistant turn.
    ///
    /// - Parameter calls: the calls that the turn makes.
    /// - Returns: the block, empty when the turn makes no call.
    private static func toolCallsBlock(_ calls: [ToolCall]) -> String {
        guard !calls.isEmpty else { return "" }
        let invocations = calls.map(invocation(of:)).joined(separator: "\n")
        return "\n\n<\(SpecialToken.dsml)\(toolCallsBlockName)>\n" + invocations
            + "\n</\(SpecialToken.dsml)\(toolCallsBlockName)>"
    }

    /// One DSML invocation.
    /// - Parameter call: the call to render.
    /// - Returns: the invocation.
    private static func invocation(of call: ToolCall) -> String {
        "<\(SpecialToken.dsml)invoke name=\"\(call.name)\">\n" + parameters(of: call)
            + "\n</\(SpecialToken.dsml)invoke>"
    }

    /// The DSML parameters of one call, one for each member of its arguments.
    /// - Parameter call: the call to read.
    /// - Returns: the parameters, one on each line.
    private static func parameters(of call: ToolCall) -> String {
        arguments(of: call).map(parameter(_:)).joined(separator: "\n")
    }

    /// One DSML parameter.
    ///
    /// A string argument goes in as it is, under `string="true"`. Every other
    /// kind goes in as JSON, under `string="false"`.
    ///
    /// - Parameter member: the argument to render.
    /// - Returns: the parameter.
    private static func parameter(_ member: PythonStyleJSON.Member) -> String {
        let isString: Bool
        let value: String
        if case .string(let text) = member.value {
            isString = true
            value = text
        } else {
            isString = false
            value = member.value.pythonStyleText
        }
        return "<\(SpecialToken.dsml)parameter name=\"\(member.key)\" "
            + "string=\"\(isString)\">\(value)</\(SpecialToken.dsml)parameter>"
    }

    /// The members of the arguments of one call.
    ///
    /// Arguments that are not a JSON object become one member named
    /// `arguments`, which is what the reference does when `json.loads` fails.
    ///
    /// - Parameter call: the call to read.
    /// - Returns: the members, in the order of the text.
    private static func arguments(of call: ToolCall) -> [PythonStyleJSON.Member] {
        guard case .object(let members)? = PythonStyleJSON.parse(call.argumentsJSON) else {
            return [PythonStyleJSON.Member(key: "arguments", value: .string(call.argumentsJSON))]
        }
        return members
    }

    /// The pieces of a user turn, joined by a blank line.
    /// - Parameter message: the user turn to render.
    /// - Returns: the text of the turn.
    private static func userBody(of message: Message) -> String {
        guard let blocks = message.contentBlocks, !blocks.isEmpty else { return message.content }
        return blocks.map(rendered(_:)).joined(separator: "\n\n")
    }

    /// Renders one piece of a user turn.
    /// - Parameter block: the piece to render.
    /// - Returns: the text.
    private static func rendered(_ block: ContentBlock) -> String {
        switch block {
        case .text(let text):
            return text
        case .toolResult(_, let content):
            return toolResultBlock(content)
        }
    }

    /// Wraps the answer of one tool.
    /// - Parameter content: the text that the tool gave back.
    /// - Returns: the wrapped answer.
    private static func toolResultBlock(_ content: String) -> String {
        "<tool_result>" + content + "</tool_result>"
    }

    /// Writes JSON text again the way Python's
    /// `json.dumps(value, ensure_ascii=False)` writes it.
    ///
    /// - Parameter text: the JSON text to rewrite.
    /// - Returns: the one-line form, or `text` itself when it is not JSON.
    private static func pythonStyleJSON(_ text: String) -> String {
        PythonStyleJSON.parse(text)?.pythonStyleText ?? text
    }

    /// Finds the turn that the reasoning rules measure against.
    ///
    /// - Parameter messages: the turns to search.
    /// - Returns: the index of the last user or developer turn, or `-1` when
    ///   there is none.
    private static func lastUserOrDeveloperIndex(in messages: [Message]) -> Int {
        let found = messages.lastIndex { $0.role == .user || $0.role == .developer }
        return found ?? -1
    }
}
