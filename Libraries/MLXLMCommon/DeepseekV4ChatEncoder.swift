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
// EXTENSION POINT FOR THE TOOL-ENCODING FOLLOW-ON TASK
//
// This file renders the non-tool part of the reference. Five named places below
// carry a `Tool encoding:` note that states what the follow-on task adds there.
// Nothing else in this file has to move for that work.

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
        /// A reminder that the caller puts in front of the last user turn.
        case latestReminder = "latest_reminder"
    }

    /// One of the internal classification tasks that DeepSeek-V4 answers with a
    /// task token in place of an ordinary assistant turn.
    public enum QuickInstructionTask: String, Sendable {
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
    /// Build a message with one of the role factories — ``system(content:)``,
    /// ``user(content:task:)``, ``developer(content:)``,
    /// ``latestReminder(content:)`` or
    /// ``assistant(content:reasoning:endsWithEndOfSentence:)``.
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

        /// Creates a message.
        ///
        /// - Parameters:
        ///   - role: the role of the sender.
        ///   - content: the text of the message.
        ///   - reasoning: the chain of thought of an assistant turn.
        ///   - task: the classification task this message asks for.
        ///   - endsWithEndOfSentence: whether an assistant turn is closed.
        public init(
            role: Role,
            content: String,
            reasoning: String? = nil,
            task: QuickInstructionTask? = nil,
            endsWithEndOfSentence: Bool = true
        ) {
            self.role = role
            self.content = content
            self.reasoning = reasoning
            self.task = task
            self.endsWithEndOfSentence = endsWithEndOfSentence
        }

        /// A system message.
        /// - Parameter content: the text of the system prompt.
        /// - Returns: the message.
        public static func system(content: String) -> Self {
            Self(role: .system, content: content)
        }

        /// A developer message.
        /// - Parameter content: the text of the instruction.
        /// - Returns: the message.
        public static func developer(content: String) -> Self {
            Self(role: .developer, content: content)
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
        ///   - endsWithEndOfSentence: whether the turn is closed.
        /// - Returns: the message.
        public static func assistant(
            content: String,
            reasoning: String? = nil,
            endsWithEndOfSentence: Bool = true
        ) -> Self {
            Self(
                role: .assistant, content: content, reasoning: reasoning,
                endsWithEndOfSentence: endsWithEndOfSentence)
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

        /// The marker of one classification task.
        /// - Parameter task: the task to name.
        /// - Returns: the marker.
        public static func task(_ task: QuickInstructionTask) -> String {
            marker(task.rawValue)
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
        let renderedMessages = Self.mergeConsecutiveUserMessages(messages)
        let renderedContext = Self.mergeConsecutiveUserMessages(context)
        var prompt =
            (addsBeginOfSentence && renderedContext.isEmpty)
            ? SpecialToken.beginOfSentence : ""

        // Tool encoding: the reference turns `drop_thinking` off whenever any
        // message still carries `tools`, because a mid-trajectory agent turn
        // needs its whole chain of thought. The follow-on task adds the `tools`
        // field and computes the value below from `dropsEarlierReasoning` and
        // that field together.
        let full = renderedContext + renderedMessages
        let turns: [Message]
        let renderedCount: Int
        if thinkingMode == .thinking && dropsEarlierReasoning {
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
                dropsReasoning: dropsEarlierReasoning, reasoningEffort: reasoningEffort)
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
            // Tool encoding: the reference appends the tool schemas and the
            // response format to a system message that carries them.
            return message.content
        case .developer:
            // Tool encoding: the reference appends the tool schemas and the
            // response format to a developer message that carries them.
            return SpecialToken.user + message.content
        case .user:
            // Tool encoding: the reference renders `content_blocks` here, which
            // hold the `<tool_result>` answers that a tool turn merges in.
            return SpecialToken.user + message.content
        case .latestReminder:
            return SpecialToken.latestReminder + message.content
        case .assistant:
            var out = ""
            if thinkingMode == .thinking && !precededByTask
                && (!dropsReasoning || isAfterLastUser)
            {
                out += (message.reasoning ?? "") + SpecialToken.thinkEnd
            }
            out += message.content
            // Tool encoding: the reference appends the DSML `tool_calls` block
            // of an assistant turn here, before the end-of-sentence marker.
            if message.endsWithEndOfSentence {
                out += SpecialToken.endOfSentence
            }
            return out
        }
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
                || message.role == .latestReminder
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

    /// Joins user turns that follow one another into one turn.
    ///
    /// The reference builds one `content_blocks` list for them and joins the
    /// blocks with a blank line, which gives one user marker for the whole run.
    /// A turn that names a task never absorbs the turn after it.
    ///
    /// - Parameter messages: the turns to merge.
    /// - Returns: the merged turns.
    private static func mergeConsecutiveUserMessages(_ messages: [Message]) -> [Message] {
        // Tool encoding: the reference also folds a `tool` turn into the user
        // turn before it, as a `<tool_result>` block of the same list.
        var merged: [Message] = []
        for message in messages {
            guard message.role == .user, var last = merged.last, last.role == .user,
                last.task == nil
            else {
                merged.append(message)
                continue
            }
            last.content += "\n\n" + message.content
            merged[merged.count - 1] = last
        }
        return merged
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
