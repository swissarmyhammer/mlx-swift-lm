// Copyright © 2025 Apple Inc.

import Foundation

public struct ToolCall: Hashable, Codable, Sendable {
    /// Represents the function details for a tool call
    public struct Function: Hashable, Codable, Sendable {
        /// The name of the function
        public let name: String

        /// The arguments passed to the function
        public let arguments: [String: JSONValue]

        /// The arguments as the model itself wrote them, as JSON text.
        ///
        /// ``arguments`` is a Swift `Dictionary`, thus it keeps no order and a
        /// render of the same call must invent one. A parser that reads the
        /// arguments in the order of the output of the model puts that order
        /// here, and a chat template that renders the call again writes the
        /// members in the order of this text. The text also keeps the digits of
        /// each number as the model wrote them, which a parsed number loses.
        ///
        /// A caller that has no such text gives `nil`, and the render falls
        /// back to the fixed order that ``arguments`` gets.
        ///
        /// The text carries the same values as ``arguments``, thus it takes no
        /// part in equality: two calls with the same name and the same
        /// arguments ask the same thing of the same tool, and the spelling of
        /// one render does not change what they ask.
        public let argumentsJSON: String?

        /// The key that carries ``argumentsJSON`` in the raw message dictionary
        /// that ``MessageGenerator/addToolMetadata(to:for:)`` writes.
        public static let argumentsJSONKey = "arguments_json"

        /// Creates the function details of a tool call.
        ///
        /// - Parameters:
        ///   - name: The name of the function.
        ///   - arguments: The arguments passed to the function.
        ///   - argumentsJSON: The arguments as the model itself wrote them, as
        ///     JSON text. See ``argumentsJSON``.
        public init(
            name: String, arguments: [String: JSONValue], argumentsJSON: String? = nil
        ) {
            self.name = name
            self.arguments = arguments
            self.argumentsJSON = argumentsJSON
        }

        /// Creates the function details of a tool call from Swift values.
        ///
        /// - Parameters:
        ///   - name: The name of the function.
        ///   - arguments: The arguments passed to the function, as Swift values.
        ///   - argumentsJSON: The arguments as the model itself wrote them, as
        ///     JSON text. See ``argumentsJSON``.
        public init(
            name: String, arguments: [String: any Sendable], argumentsJSON: String? = nil
        ) {
            self.init(
                name: name, arguments: arguments.mapValues { JSONValue.from($0) },
                argumentsJSON: argumentsJSON)
        }

        var argumentsObject: [String: any Sendable] {
            arguments.mapValues { $0.sendableValue }
        }

        /// Compares two calls by the name and the arguments.
        ///
        /// ``argumentsJSON`` takes no part, because it is one spelling of the
        /// same arguments.
        ///
        /// - Parameters:
        ///   - lhs: The first function to compare.
        ///   - rhs: The second function to compare.
        /// - Returns: Whether the two ask the same thing of the same tool.
        public static func == (lhs: Self, rhs: Self) -> Bool {
            lhs.name == rhs.name && lhs.arguments == rhs.arguments
        }

        /// Hashes the members that equality compares.
        ///
        /// - Parameter hasher: The hasher to feed.
        public func hash(into hasher: inout Hasher) {
            hasher.combine(name)
            hasher.combine(arguments)
        }
    }

    /// The function to be called
    public let function: Function

    /// Optional id used to correlate a tool call with its tool result.
    public let id: String?

    public init(function: Function, id: String? = nil) {
        self.function = function
        self.id = id
    }
}

extension ToolCall {
    public func execute<Input, Output>(with tool: Tool<Input, Output>) async throws -> Output {
        // Check that the tool name matches the function name
        guard tool.name == function.name else {
            throw ToolError.nameMismatch(toolName: tool.name, functionName: function.name)
        }

        // Convert the JSONValue arguments dictionary to a JSON-encoded Data object
        let jsonObject = function.arguments.mapValues { $0.anyValue }
        let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)

        // Decode the Input type from the JSON data
        let input = try JSONDecoder().decode(Input.self, from: jsonData)

        // Execute the tool's handler with the decoded input
        return try await tool.handler(input)
    }
}

// Define Tool-related errors
public enum ToolError: Error, LocalizedError {
    case nameMismatch(toolName: String, functionName: String)

    public var errorDescription: String? {
        switch self {
        case .nameMismatch(let toolName, let functionName):
            return "Tool name mismatch: expected '\(toolName)' but got '\(functionName)'"
        }
    }
}
