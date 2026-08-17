// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Ported from osaurus-ai/vmlx-swift-lm
//   Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift @ 4546a5d720e7013adffdbddd728c6106e4f9e637
// Manual transcription; no git ancestry.
//
// Every expected string below is a hand transcription of the output of
// DeepSeek's own Python encoder, `encoding/encoding_dsv4.py`, in the model
// repository `deepseek-ai/DeepSeek-V4-Flash` @
// 60d8d70770c6776ff598c94bb586a859a38244f1 (sha256 of that file is
// bdbd57c132a1b372…). No expected string comes from ``DeepSeekV4ChatEncoder``,
// thus a defect in the encoder cannot write itself into the expectation.
//
// ``expectedFixtureOne`` and ``expectedFixtureThree`` are character for
// character transcriptions of the published golden files
// `encoding/tests/test_output_1.txt` (2281 characters) and
// `encoding/tests/test_output_3.txt` (3021 characters) in the same model
// repository. The other seven conversations were rendered by running that
// Python file over the message lists this file declares.
//
// Each tool schema goes in as the indented JSON of the published input file,
// and each expectation holds the one-line JSON that Python writes. Thus the
// tests measure the reader that keeps the key order and the writer that leaves
// the solidus alone, and not a copy of the text from one place to another.
//
// The special tokens in each expected string use the FULLWIDTH VERTICAL LINE
// U+FF5C and the LOWER ONE EIGHTH BLOCK U+2581, not the ASCII `|` and `_`.
// ``testDSMLTokenUsesFullwidthVerticalLine`` examines those code points on
// their own, because the two look almost the same on a screen.

import Foundation
import MLXLMCommon
import XCTest

/// Examines the tool half of ``DeepSeekV4ChatEncoder`` against the output of
/// the Python encoder that DeepSeek ships in the DeepSeek-V4-Flash model
/// repository.
final class DeepSeekV4ToolEncodingTests: XCTestCase {

    private let encoder = DeepSeekV4ChatEncoder()

    // MARK: - The published golden fixture 1

    func testPublishedFixtureOneRendersByteIdentically() {
        let prompt = encoder.encode(
            messages: [
                .system(
                    content: "You are a helpful assistant.",
                    tools: [Self.weatherTool, Self.webSearchTool]),
                .user(content: "What's the weather in Beijing?"),
                .assistant(
                    content: "",
                    reasoning:
                        "The user wants to know the weather in Beijing. I should use the get_weather tool.",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_001", name: "get_weather",
                            argumentsJSON: "{\"location\": \"Beijing\", \"unit\": \"celsius\"}")
                    ]),
                .toolResult(
                    content: "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}",
                    toolCallID: "call_001"),
                .assistant(
                    content:
                        "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
                    reasoning: "Got the weather data. Let me format a nice response."),
            ], thinkingMode: .thinking)
        XCTAssertEqual(prompt, Self.expectedFixtureOne)
    }

    /// The `get_weather` schema of the published input `test_input_1.json`.
    private static let weatherTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "get_weather",
              "description": "Get the weather for a specific location",
              "parameters": {
                "type": "object",
                "properties": {
                  "location": {
                    "type": "string",
                    "description": "The city name"
                  },
                  "unit": {
                    "type": "string",
                    "enum": [
                      "celsius",
                      "fahrenheit"
                    ],
                    "description": "Temperature unit"
                  }
                },
                "required": [
                  "location"
                ]
              }
            }
            """)

    /// The `search` schema of the published input `test_input_1.json`.
    private static let webSearchTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "search",
              "description": "Search the web for information",
              "parameters": {
                "type": "object",
                "properties": {
                  "query": {
                    "type": "string",
                    "description": "Search query"
                  },
                  "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                  }
                },
                "required": [
                  "query"
                ]
              }
            }
            """)

    /// The published golden output `encoding/tests/test_output_1.txt`.
    private static let expectedFixtureOne =
        "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"get_weather\", \"description\": \"Get the weather for a specific location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The city name\"}, \"unit\": {\"type\": \"string\", \"enum\": [\"celsius\", \"fahrenheit\"], \"description\": \"Temperature unit\"}}, \"required\": [\"location\"]}}\n{\"name\": \"search\", \"description\": \"Search the web for information\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"Search query\"}, \"num_results\": {\"type\": \"integer\", \"description\": \"Number of results to return\"}}, \"required\": [\"query\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>What's the weather in Beijing?<｜Assistant｜><think>The user wants to know the weather in Beijing. I should use the get_weather tool.</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}</tool_result><｜Assistant｜><think>Got the weather data. Let me format a nice response.</think>The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.<｜end▁of▁sentence｜>"

    // MARK: - The published golden fixture 3

    func testPublishedFixtureThreeRendersByteIdentically() {
        let prompt = encoder.encode(
            messages: [
                .system(content: "该助手为DeepSeek，由深度求索公司创造。"),
                .latestReminder(content: "2026-02-21,星期六,广州,App,中文"),
                .developer(
                    content:
                        "小柴胡冲剂和布洛芬能一起吃吗？\n\nCITATION FORMAT: 【{cursor_id}†L{start_line_id}(-L{end_line_id})?】",
                    tools: [Self.citedSearchTool, Self.openTool, Self.findTool]),
                .assistant(
                    content: "", reasoning: "用户想知道小柴胡冲剂和布洛芬能否一起服用。",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            name: "search",
                            argumentsJSON: "{\"queries\": \"小柴胡冲剂 布洛芬 相互作用 一起吃\"}")
                    ]),
                .toolResult(content: "[0]"),
                .assistant(content: "请及时就医。", reasoning: "现在开始组织回答。"),
            ], thinkingMode: .thinking)
        XCTAssertEqual(prompt, Self.expectedFixtureThree)
    }

    /// The `search` schema of the published input `test_input_3.json`.
    private static let citedSearchTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "search",
              "description": "Web search. Split multiple queries with '||'.",
              "parameters": {
                "type": "object",
                "properties": {
                  "queries": {
                    "type": "string",
                    "description": "query1||query2"
                  }
                },
                "required": [
                  "queries"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
              }
            }
            """)

    /// The `open` schema of the published input `test_input_3.json`.
    private static let openTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "open",
              "description": "Batch open IDs (format 【{id}†...】) or URLs.",
              "parameters": {
                "type": "object",
                "properties": {
                  "open_list": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "id": {
                          "description": "ID or URL",
                          "anyOf": [
                            {
                              "type": "integer"
                            },
                            {
                              "type": "string"
                            }
                          ],
                          "default": -1
                        },
                        "cursor": {
                          "type": "integer",
                          "description": "",
                          "default": -1
                        },
                        "loc": {
                          "type": "integer",
                          "description": "Start line",
                          "default": -1
                        },
                        "num_lines": {
                          "type": "integer",
                          "description": "",
                          "default": -1
                        },
                        "view_source": {
                          "type": "boolean",
                          "description": "",
                          "default": false
                        }
                      },
                      "additionalProperties": false
                    },
                    "description": ""
                  }
                },
                "required": [
                  "open_list"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
              }
            }
            """)

    /// The `find` schema of the published input `test_input_3.json`.
    private static let findTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "find",
              "description": "Find exact text pattern in pages.",
              "parameters": {
                "type": "object",
                "properties": {
                  "find_list": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "pattern": {
                          "type": "string",
                          "description": ""
                        },
                        "cursor": {
                          "type": "integer",
                          "description": "",
                          "default": -1
                        }
                      },
                      "required": [
                        "pattern"
                      ],
                      "additionalProperties": false
                    },
                    "description": ""
                  }
                },
                "required": [
                  "find_list"
                ],
                "additionalProperties": false,
                "$schema": "http://json-schema.org/draft-07/schema#"
              }
            }
            """)

    /// The published golden output `encoding/tests/test_output_3.txt`.
    private static let expectedFixtureThree =
        "<｜begin▁of▁sentence｜>该助手为DeepSeek，由深度求索公司创造。<｜latest_reminder｜>2026-02-21,星期六,广州,App,中文<｜User｜>小柴胡冲剂和布洛芬能一起吃吗？\n\nCITATION FORMAT: 【{cursor_id}†L{start_line_id}(-L{end_line_id})?】\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"search\", \"description\": \"Web search. Split multiple queries with '||'.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"queries\": {\"type\": \"string\", \"description\": \"query1||query2\"}}, \"required\": [\"queries\"], \"additionalProperties\": false, \"$schema\": \"http://json-schema.org/draft-07/schema#\"}}\n{\"name\": \"open\", \"description\": \"Batch open IDs (format 【{id}†...】) or URLs.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"open_list\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"id\": {\"description\": \"ID or URL\", \"anyOf\": [{\"type\": \"integer\"}, {\"type\": \"string\"}], \"default\": -1}, \"cursor\": {\"type\": \"integer\", \"description\": \"\", \"default\": -1}, \"loc\": {\"type\": \"integer\", \"description\": \"Start line\", \"default\": -1}, \"num_lines\": {\"type\": \"integer\", \"description\": \"\", \"default\": -1}, \"view_source\": {\"type\": \"boolean\", \"description\": \"\", \"default\": false}}, \"additionalProperties\": false}, \"description\": \"\"}}, \"required\": [\"open_list\"], \"additionalProperties\": false, \"$schema\": \"http://json-schema.org/draft-07/schema#\"}}\n{\"name\": \"find\", \"description\": \"Find exact text pattern in pages.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"find_list\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"pattern\": {\"type\": \"string\", \"description\": \"\"}, \"cursor\": {\"type\": \"integer\", \"description\": \"\", \"default\": -1}}, \"required\": [\"pattern\"], \"additionalProperties\": false}, \"description\": \"\"}}, \"required\": [\"find_list\"], \"additionalProperties\": false, \"$schema\": \"http://json-schema.org/draft-07/schema#\"}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜Assistant｜><think>用户想知道小柴胡冲剂和布洛芬能否一起服用。</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"queries\" string=\"true\">小柴胡冲剂 布洛芬 相互作用 一起吃</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>[0]</tool_result><｜Assistant｜><think>现在开始组织回答。</think>请及时就医。<｜end▁of▁sentence｜>"

    // MARK: - The order of the tool results

    func testToolResultsSortByCallOrderAndNotByReturnOrder() {
        let prompt = encoder.encode(
            messages: [
                .system(content: "S"),
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_A", name: "alpha", argumentsJSON: "{\"n\": \"1\"}"),
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_B", name: "beta", argumentsJSON: "{\"n\": \"2\"}"),
                    ]),
                .toolResult(content: "B result", toolCallID: "call_B"),
                .toolResult(content: "A result", toolCallID: "call_A"),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>S<｜User｜>Q<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"alpha\">\n<｜DSML｜parameter name=\"n\" string=\"true\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"beta\">\n<｜DSML｜parameter name=\"n\" string=\"true\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>A result</tool_result>\n\n<tool_result>B result</tool_result><｜Assistant｜><think>"
        )
    }

    // MARK: - Tools switch the drop rule off

    func testToolsPresentKeepsTheReasoningOfEveryEarlierTurn() {
        let prompt = encoder.encode(
            messages: Self.twoTurnConversation(tools: [Self.echoTool]),
            thinkingMode: .thinking, dropsEarlierReasoning: true)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>S\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name\": \"echo\", \"description\": \"Echo the text back.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"text\": {\"type\": \"string\", \"description\": \"The text\"}}, \"required\": [\"text\"]}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>One<｜Assistant｜><think>First reasoning.</think>First answer.<｜end▁of▁sentence｜><｜User｜>Two<｜Assistant｜><think>"
        )
    }

    func testToolsAbsentDropsTheReasoningOfEveryEarlierTurn() {
        let prompt = encoder.encode(
            messages: Self.twoTurnConversation(tools: []),
            thinkingMode: .thinking, dropsEarlierReasoning: true)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>S<｜User｜>One<｜Assistant｜></think>First answer.<｜end▁of▁sentence｜><｜User｜>Two<｜Assistant｜><think>"
        )
    }

    /// The conversation that the two drop-rule tests share, with and without a
    /// tool on the system message.
    /// - Parameter tools: the tools the system message carries.
    /// - Returns: the four turns.
    private static func twoTurnConversation(tools: [DeepSeekV4ChatEncoder.Tool])
        -> [DeepSeekV4ChatEncoder.Message]
    {
        [
            .system(content: "S", tools: tools),
            .user(content: "One"),
            .assistant(content: "First answer.", reasoning: "First reasoning."),
            .user(content: "Two"),
        ]
    }

    /// A one-parameter schema, small enough to read inside an expectation.
    private static let echoTool = DeepSeekV4ChatEncoder.Tool(
        functionSchemaJSON: """
            {
              "name": "echo",
              "description": "Echo the text back.",
              "parameters": {
                "type": "object",
                "properties": {
                  "text": {
                    "type": "string",
                    "description": "The text"
                  }
                },
                "required": [
                  "text"
                ]
              }
            }
            """)

    // MARK: - The arguments of one call

    func testNonStringArgumentsRenderAsJSONWithStringFalse() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_1", name: "mixed",
                            argumentsJSON:
                                "{\"name\": \"a/b\", \"count\": 7, \"flag\": false, \"list\": [1, \"two\"], \"nested\": {\"k\": -1}, \"nothing\": null}"
                        )
                    ]),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜><think>R</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"mixed\">\n<｜DSML｜parameter name=\"name\" string=\"true\">a/b</｜DSML｜parameter>\n<｜DSML｜parameter name=\"count\" string=\"false\">7</｜DSML｜parameter>\n<｜DSML｜parameter name=\"flag\" string=\"false\">false</｜DSML｜parameter>\n<｜DSML｜parameter name=\"list\" string=\"false\">[1, \"two\"]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"nested\" string=\"false\">{\"k\": -1}</｜DSML｜parameter>\n<｜DSML｜parameter name=\"nothing\" string=\"false\">null</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        )
    }

    func testArgumentsThatAreNotJSONRenderAsOneStringParameter() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_1", name: "broken", argumentsJSON: "not json")
                    ]),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜><think>R</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"broken\">\n<｜DSML｜parameter name=\"arguments\" string=\"true\">not json</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        )
    }

    func testTwoCallsInOneAssistantTurnRenderInTheOrderTheyWereMade() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_1", name: "alpha", argumentsJSON: "{\"n\": \"1\"}"),
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_2", name: "beta", argumentsJSON: "{\"n\": \"2\"}"),
                    ]),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜><think>R</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"alpha\">\n<｜DSML｜parameter name=\"n\" string=\"true\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"beta\">\n<｜DSML｜parameter name=\"n\" string=\"true\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
        )
    }

    // MARK: - The response format

    func testResponseFormatRendersAfterTheSystemMessage() {
        let prompt = encoder.encode(
            messages: [
                .system(
                    content: "S",
                    responseFormatJSON: """
                        {
                          "type": "json_object",
                          "path": "a/b"
                        }
                        """),
                .user(content: "Q"),
            ], thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>S\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{\"type\": \"json_object\", \"path\": \"a/b\"}<｜User｜>Q<｜Assistant｜></think>"
        )
    }

    // MARK: - The user turn that a tool result joins

    func testATextBlockFollowsAToolResultAfterABlankLine() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(
                    content: "", reasoning: "R",
                    toolCalls: [
                        DeepSeekV4ChatEncoder.ToolCall(
                            id: "call_1", name: "alpha", argumentsJSON: "{\"n\": \"1\"}")
                    ]),
                .toolResult(content: "result", toolCallID: "call_1"),
                .user(content: "And now this."),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"alpha\">\n<｜DSML｜parameter name=\"n\" string=\"true\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>result</tool_result>\n\nAnd now this.<｜Assistant｜><think>"
        )
    }

    // MARK: - The turn the model wrote

    /// The two turns that open the agent conversation of the cache tests.
    private static let agentPrompt: [DeepSeekV4ChatEncoder.Message] = [
        .system(content: "S", tools: [echoTool]),
        .user(content: "Q"),
    ]

    /// The text the model writes in front of its block of calls.
    ///
    /// `ChatSession` keeps this text as the content of the assistant turn. It
    /// holds the reasoning and the close of the reasoning, because DeepSeek-V4
    /// declares no `reasoningConfig` and thus no decoder splits the reasoning
    /// out. It also holds the blank line that the DSML syntax asks for in front
    /// of the block, because the block itself leaves the text stream as a
    /// structured call.
    private static let generatedContent = "Reasoning.</think>I will look.\n\n"

    /// The block of calls that the model writes after ``generatedContent``.
    private static let generatedToolCallsBlock = """
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="echo">
        <｜DSML｜parameter name="text" string="true">hi</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """

    /// The one call of ``generatedToolCallsBlock``, as the parser reads it back.
    private static let echoCall = DeepSeekV4ChatEncoder.ToolCall(
        id: "call_1", name: "echo", argumentsJSON: "{\"text\": \"hi\"}")

    /// Everything the model writes after the generation tail of round 1.
    private static var generatedTurn: String {
        generatedContent + generatedToolCallsBlock
            + DeepSeekV4ChatEncoder.SpecialToken.endOfSentence
    }

    /// The render of the tool round holds the render of round 1 and then the
    /// text the model wrote.
    ///
    /// This is the property a live prompt cache needs. The ledger of
    /// `ChatSession` is the render of round 1 plus the tokens the model wrote,
    /// thus `ExtendCachedPrefixRule` splices the tool result onto the cache
    /// only while the next render writes those same tokens again.
    ///
    /// Card ^v7z7v99 measured the failure on the published checkpoint: the tool
    /// round fed every one of its 3906 tokens and skipped none, because the
    /// render wrote a second `</think>` and two more newlines than the model.
    func testToolRoundRenderExtendsRoundOneAndTheTextTheModelWrote() {
        let roundOne = encoder.encode(messages: Self.agentPrompt, thinkingMode: .thinking)
        let toolRound = encoder.encode(
            messages: Self.agentPrompt + [
                .assistant(content: Self.generatedContent, toolCalls: [Self.echoCall]),
                .toolResult(content: "result", toolCallID: "call_1"),
            ], thinkingMode: .thinking)
        XCTAssertTrue(
            toolRound.hasPrefix(roundOne + Self.generatedTurn),
            """
            the tool round render must hold the render of round 1 and then the text the \
            model wrote. It holds: <<<\(toolRound)>>>
            """)
    }

    /// An assistant turn whose content already holds the close of its reasoning
    /// renders that close once.
    func testAssistantContentThatHoldsItsOwnThinkEndRendersOneThinkEnd() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(content: "Reasoning.</think>Answer."),
            ], thinkingMode: .thinking)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜><think>Reasoning.</think>Answer."
                + "<｜end▁of▁sentence｜>")
    }

    /// The block of calls carries the blank line in front of it, thus a content
    /// that ends with that same blank line renders one blank line and not two.
    func testTheBlankLineInFrontOfTheToolCallsBlockRendersOnce() {
        let prompt = encoder.encode(
            messages: [
                .user(content: "Q"),
                .assistant(content: "I will look.\n\n", toolCalls: [Self.echoCall]),
            ], thinkingMode: .chat)
        XCTAssertEqual(
            prompt,
            "<｜begin▁of▁sentence｜><｜User｜>Q<｜Assistant｜></think>I will look.\n\n"
                + Self.generatedToolCallsBlock + "<｜end▁of▁sentence｜>")
    }

    // MARK: - Special token spelling

    func testDSMLTokenUsesFullwidthVerticalLine() {
        let fullwidth: Unicode.Scalar = "\u{FF5C}"
        XCTAssertEqual(
            Array(DeepSeekV4ChatEncoder.SpecialToken.dsml.unicodeScalars),
            [fullwidth, "D", "S", "M", "L", fullwidth],
            "the DSML token must not use the ASCII vertical line")
    }
}
