// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// The weather tool loop that the weight-free Inkling 2 tool-loop tests script.
///
/// It holds one tool that reads the weather of a city, the call a model writes for it, the call
/// the published template renders in its place, and the turns of the model, frame by frame, in
/// the order the real checkpoint writes them: a thinking frame, one tool frame for each call,
/// then `<|content_model_end_sampling|>`.
///
/// The `ChatSession` tests of `MLXLMTests` and the executor tests of
/// `MLXFoundationModelsTests` share it, thus both script the same wire format. Each suite names
/// its own texts and keeps its own assertions.
public struct InklingWeatherToolLoop: Sendable {

    /// The name of the tool.
    public static let toolName = "get_weather"

    /// The description of the tool.
    public static let toolDescription = "Get the current weather for a city"

    /// The name of the one argument of the tool.
    public static let cityArgument = "city"

    /// The weather of each city the tool knows.
    public static let weather = ["Rome": "Sunny, 21 C", "Paris": "Rainy, 12 C"]

    /// The tool a `ChatSession` declares.
    public static let toolSpec: ToolSpec = [
        "type": "function",
        "function": [
            "name": toolName,
            "description": toolDescription,
            "parameters": [
                "type": "object",
                "properties": [cityArgument: ["type": "string"] as [String: any Sendable]],
                "required": [cityArgument],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// The call of the tool for `city` as the model writes it: JSON with spaces.
    ///
    /// - Parameter city: the city of the call.
    /// - Returns: the text of the payload.
    public static func writtenCall(_ city: String) -> String {
        #"{"name": "\#(toolName)", "args": {"\#(cityArgument)": "\#(city)"}}"#
    }

    /// The call of the tool for `city` as the published template renders it: compact JSON with
    /// sorted keys. The render of ``InklingFrameTokenizer`` must write exactly this text, or the
    /// call renders as the unknown token.
    ///
    /// - Parameter city: the city of the call.
    /// - Returns: the text of the payload.
    public static func renderedCall(_ city: String) -> String {
        #"{"name":"\#(toolName)","args":{"\#(cityArgument)":"\#(city)"}}"#
    }

    /// The reasoning the model writes before its calls.
    public let reasoning: String

    /// The answer the model writes after the tool results.
    public let answer: String

    /// The tokenizer whose vocabulary holds every text of the conversation.
    public let tokenizer: InklingFrameTokenizer

    /// Makes the loop of a suite.
    ///
    /// - Parameters:
    ///   - reasoning: the reasoning the model writes before its calls.
    ///   - answer: the answer the model writes after the tool results.
    ///   - otherPieces: the other texts of the suite, such as its question, one token each.
    public init(reasoning: String, answer: String, otherPieces: [String]) {
        let cities = Self.weather.keys.sorted()
        self.reasoning = reasoning
        self.answer = answer
        self.tokenizer = InklingFrameTokenizer(
            pieces: otherPieces + [reasoning, answer, Self.toolName]
                + cities.flatMap { [Self.writtenCall($0), Self.renderedCall($0)] }
                + cities.compactMap { Self.weather[$0] })
    }

    /// The tokens of `markers`.
    ///
    /// - Parameter markers: the control markers, in order.
    /// - Returns: their tokens.
    public func tokens(_ markers: InklingFrameTokenizer.Marker...) -> [Int] {
        markers.map(tokenizer.token)
    }

    /// The thinking frame the model writes.
    public var thinkingFrame: [Int] {
        tokens(.messageModel, .contentThinking) + [tokenizer.token(reasoning)]
            + tokens(.endMessage)
    }

    /// The tool frame the model writes for `city`.
    ///
    /// - Parameter city: the city of the call.
    /// - Returns: the tokens of the frame.
    public func callFrame(_ city: String) -> [Int] {
        tokens(.messageModel) + [tokenizer.token(Self.toolName)] + tokens(.contentToolJSON)
            + [tokenizer.token(Self.writtenCall(city))] + tokens(.endMessage)
    }

    /// The tool result frame the template renders for `city`.
    ///
    /// - Parameter city: the city of the call the result answers.
    /// - Returns: the tokens of the frame.
    public func resultFrame(_ city: String) -> [Int] {
        tokens(.messageTool) + [tokenizer.token(Self.toolName)] + tokens(.contentText)
            + [tokenizer.token(Self.weather[city] ?? "")] + tokens(.endMessage)
    }

    /// The tokens the model writes in the turn that calls the tool for each of `cities`.
    ///
    /// - Parameter cities: the city of each call, in order.
    /// - Returns: the tokens of the turn.
    public func callingTurn(_ cities: [String]) -> [Int] {
        thinkingFrame + cities.flatMap(callFrame) + tokens(.endSampling)
    }

    /// The tokens the model writes in the turn that answers from the tool results.
    public var answeringTurn: [Int] {
        tokens(.messageModel, .contentText) + [tokenizer.token(answer)]
            + tokens(.endMessage, .endSampling)
    }
}
