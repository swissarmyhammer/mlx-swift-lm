// Copyright © 2025 Apple Inc.

/// Namespace for structured chat message types used to represent
/// conversations independent of any specific model's raw message format.
///
/// See ``Chat/Message`` for the structured message type and
/// ``MessageGenerator`` for how these are converted into a model-specific
/// raw representation.
public enum Chat {
    /// A single structured chat message: a role, text content, optional
    /// media attachments, and optional tool-call metadata.
    ///
    /// Use the ``system(_:images:videos:)``, ``assistant(_:images:videos:toolCalls:)``,
    /// ``user(_:images:videos:audios:)``, and ``tool(_:images:id:)`` factory
    /// methods to construct role-specific messages, or the memberwise
    /// ``init(role:content:images:videos:audios:tool:)`` directly.
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        ///
        /// Note: media attachments are intentionally not included in the raw
        /// dictionary produced by ``MessageGenerator/generate(message:)`` --
        /// see that method's documentation for how media reaches model input
        /// via a separate, role-agnostic channel.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        ///
        /// See ``images`` for how media attachments are handled relative to
        /// ``MessageGenerator/generate(message:)``.
        public var videos: [UserInput.Video]

        /// Array of audio data associated with the message.
        ///
        /// See ``images`` for how media attachments are handled relative to
        /// ``MessageGenerator/generate(message:)``.
        public var audios: [UserInput.Audio]

        /// Tool-call metadata associated with this message.
        public var tool: Tool?

        /// Tool-call metadata attached to a message: either the tool calls
        /// emitted by an assistant message, or the id of the tool call that
        /// a tool-result message is answering.
        public struct Tool: Sendable {
            fileprivate enum Storage: Sendable {
                case calls([ToolCall])
                case result(id: String)
            }

            fileprivate let storage: Storage

            private init(storage: Storage) {
                self.storage = storage
            }

            /// Tool calls emitted by an assistant message.
            public static func calls(_ calls: [ToolCall]) -> Self {
                Self(storage: .calls(calls))
            }

            /// Id of the assistant tool call answered by a tool message.
            public static func result(id: String) -> Self {
                Self(storage: .result(id: id))
            }
        }

        /// Creates a structured chat message.
        ///
        /// - Parameters:
        ///   - role: The role of the message sender.
        ///   - content: The text content of the message.
        ///   - images: Image attachments associated with the message.
        ///   - videos: Video attachments associated with the message.
        ///   - audios: Audio attachments associated with the message.
        ///   - tool: Tool-call metadata associated with the message, if any.
        public init(
            role: Role, content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = [],
            tool: Tool? = nil
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
            self.audios = audios
            self.tool = tool
        }

        /// Shared factory helper backing the role-specific factory methods
        /// (``system(_:images:videos:)``, ``assistant(_:images:videos:toolCalls:)``,
        /// ``user(_:images:videos:audios:)``, ``tool(_:images:id:)``). Each
        /// public factory delegates here, passing only the parameters
        /// relevant to its own external signature.
        private static func create(
            role: Role, content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = [],
            tool: Tool? = nil
        ) -> Self {
            Self(role: role, content: content, images: images, videos: videos, audios: audios, tool: tool)
        }

        /// Creates a system message that provides instructions or context to the model.
        ///
        /// - Parameters:
        ///   - content: The system instruction text.
        ///   - images: Image attachments associated with the message.
        ///   - videos: Video attachments associated with the message.
        /// - Returns: A new ``Message`` with the ``Role/system`` role.
        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            create(role: .system, content: content, images: images, videos: videos)
        }

        /// Creates an assistant message, optionally carrying tool calls the model requested.
        ///
        /// - Parameters:
        ///   - content: The assistant's response text.
        ///   - images: Image attachments associated with the message.
        ///   - videos: Video attachments associated with the message.
        ///   - toolCalls: Tool calls emitted by the assistant, if any.
        /// - Returns: A new ``Message`` with the ``Role/assistant`` role.
        public static func assistant(
            _ content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            toolCalls: [ToolCall]? = nil
        ) -> Self {
            create(
                role: .assistant, content: content, images: images, videos: videos,
                tool: toolCalls.map { .calls($0) })
        }

        /// Creates a user message.
        ///
        /// - Parameters:
        ///   - content: The user's text content.
        ///   - images: Image attachments associated with the message.
        ///   - videos: Video attachments associated with the message.
        ///   - audios: Audio attachments associated with the message.
        /// - Returns: A new ``Message`` with the ``Role/user`` role.
        public static func user(
            _ content: String,
            images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            audios: [UserInput.Audio] = []
        ) -> Self {
            create(role: .user, content: content, images: images, videos: videos, audios: audios)
        }

        /// Creates a tool-result message containing the output of an executed tool.
        ///
        /// - Parameters:
        ///   - content: The tool's text (or JSON-rendered structured) result.
        ///   - images: Image attachments the tool's result carried, if any.
        ///   - id: Correlates this result with the tool call that produced it.
        public static func tool(
            _ content: String, images: [UserInput.Image] = [], id: String? = nil
        ) -> Self {
            create(role: .tool, content: content, images: images, tool: id.map { .result(id: $0) })
        }

        /// The role of a message's sender within a chat conversation.
        public enum Role: String, Sendable {
            /// A message from the end user.
            case user

            /// A message from the assistant (model).
            case assistant

            /// A system prompt or instruction message.
            case system

            /// A message containing the result of an executed tool call.
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat/Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    ///
    /// - Parameter input: The input to convert into raw messages.
    /// - Returns: Raw messages, aka ``Message``.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    ///
    /// - Parameter messages: The structured chat messages to convert.
    /// - Returns: Array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    ///
    /// - Parameter message: The structured chat message to convert.
    /// - Returns: `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    /// Default implementation that produces a raw dictionary containing
    /// `role` and `content`, plus `tool_calls`/`tool_call_id` when tool
    /// metadata is present (see ``addToolMetadata(to:for:)``).
    ///
    /// Note: media attachments (``Chat/Message/images``, ``Chat/Message/videos``,
    /// ``Chat/Message/audios``) are intentionally *not* included in this
    /// dictionary for any role. Media is instead extracted role-agnostically
    /// directly from the `Chat.Message` array by `UserInput.init(chat:processing:tools:additionalContext:)`,
    /// bypassing this dictionary entirely, so chat-template rendering here
    /// stays text/tool-call-only by design.
    ///
    /// - Parameter message: The structured chat message to convert.
    /// - Returns: `[String: any Sendable]`, aka ``Message``.
    public func generate(message: Chat.Message) -> Message {
        var dictionary: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]

        addToolMetadata(to: &dictionary, for: message)

        return dictionary
    }

    /// Adds tool-call metadata from a structured message to a raw message dictionary.
    public func addToolMetadata(to dictionary: inout Message, for message: Chat.Message) {
        switch message.tool?.storage {
        case .calls(let calls):
            dictionary["tool_calls"] = calls.map { toolCall -> [String: any Sendable] in
                var entry: [String: any Sendable] = [
                    "type": "function",
                    "function": [
                        "name": toolCall.function.name,
                        "arguments": toolCall.function.argumentsObject,
                    ] as [String: any Sendable],
                ]
                if let id = toolCall.id {
                    entry["id"] = id
                }
                return entry
            }
        case .result(let id):
            dictionary["tool_call_id"] = id
        case nil:
            break
        }
    }

    /// Default implementation that converts each structured chat message to
    /// its raw dictionary form via ``generate(message:)``.
    ///
    /// - Parameter messages: The structured chat messages to convert.
    /// - Returns: Array of `[String: any Sendable]` aka ``Message``
    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    /// Default implementation that dispatches on the input's prompt kind:
    /// plain text becomes a single generated user message, `.messages` are
    /// passed through unchanged (already in raw form), and `.chat` messages
    /// are converted via ``generate(messages:)``.
    ///
    /// - Parameter input: The input to convert into raw messages.
    /// - Returns: Raw messages, aka ``Message``.
    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces `role` and
/// `content`, plus `tool_call_id` and `tool_calls` when present.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    /// Creates a default message generator.
    public init() {}
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    /// Creates a message generator that omits `system` role messages.
    public init() {}

    /// Converts structured chat messages into raw dictionaries, filtering
    /// out any message with the ``Chat/Message/Role/system`` role before
    /// converting via ``MessageGenerator/generate(message:)``.
    ///
    /// - Parameter messages: The structured chat messages to convert.
    /// - Returns: Array of `[String: any Sendable]` aka ``Message``, excluding system messages.
    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}
