// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// WHY THIS TYPE EXISTS
//
// `DeepSeekV4ChatEncoder` must write a tool schema and the arguments of a tool
// call exactly as DeepSeek's own Python encoder writes them, because the model
// saw those bytes during training. The Python calls
// `json.dumps(value, ensure_ascii=False)` on a value that `json.loads` made,
// and that gives two properties Foundation does not give:
//
//   1. The members of an object keep the order of the source text.
//      `JSONSerialization` gives a `Dictionary`, whose order is undefined.
//   2. The solidus stays a solidus. `JSONSerialization` writes `http:\/\/…`
//      where Python writes `http://…`, and the published golden fixture
//      `encoding/tests/test_output_3.txt` holds the Python spelling.
//
// `JSONValue` in `Tool/Value.swift` carries `[String: JSONValue]`, thus it has
// the same order defect and cannot be reused here.
//
// ONE KNOWN DIVERGENCE FROM PYTHON
//
// A number keeps the text of the source instead of a parsed value, so a round
// trip never loses a digit of a large integer. Python instead re-writes the
// parsed number, thus Python turns `1e5` into `100000.0` and `1.0` into `1.0`.
// Every number of every published fixture is a plain integer, where the two
// agree.

import Foundation

/// A JSON value that keeps the order of the members of each object.
enum PythonStyleJSON: Equatable, Sendable {
    /// The `null` literal.
    case null
    /// A `true` or `false` literal.
    case bool(Bool)
    /// A number, held as the text of the source.
    case number(String)
    /// A string, with every escape of the source already decoded.
    case string(String)
    /// An array, in the order of the source.
    case array([PythonStyleJSON])
    /// An object, in the order of the source.
    case object([Member])

    /// One member of a JSON object.
    struct Member: Equatable, Sendable {
        /// The name of the member.
        let key: String
        /// The value of the member.
        let value: PythonStyleJSON
    }
}

// MARK: - Reading

extension PythonStyleJSON {

    /// Reads one JSON document.
    ///
    /// - Parameter text: the JSON text to read.
    /// - Returns: the value, or `nil` when `text` is not one whole JSON
    ///   document.
    static func parse(_ text: String) -> PythonStyleJSON? {
        var reader = JSONReader(text: text)
        guard let value = reader.readValue() else { return nil }
        reader.skipWhitespace()
        return reader.isAtEnd ? value : nil
    }
}

/// Walks the scalars of a JSON document once, left to right.
private struct JSONReader {

    /// The scalars of the whole document.
    private let scalars: [Unicode.Scalar]
    /// The scalar to read next.
    private var index = 0

    /// Creates a reader over one document.
    /// - Parameter text: the JSON text to read.
    init(text: String) {
        scalars = Array(text.unicodeScalars)
    }

    /// Whether every scalar is read.
    var isAtEnd: Bool { index >= scalars.count }

    /// Steps over the space, tab, carriage return and newline that JSON allows
    /// between tokens.
    mutating func skipWhitespace() {
        while let scalar = peek(),
            scalar == " " || scalar == "\t" || scalar == "\r" || scalar == "\n"
        {
            index += 1
        }
    }

    /// Reads one value of any kind.
    /// - Returns: the value, or `nil` when the text at this point is not a
    ///   value.
    mutating func readValue() -> PythonStyleJSON? {
        skipWhitespace()
        guard let scalar = peek() else { return nil }
        switch scalar {
        case "{": return readObject()
        case "[": return readArray()
        case "\"": return readString().map { .string($0) }
        case "t": return readKeyword("true", value: .bool(true))
        case "f": return readKeyword("false", value: .bool(false))
        case "n": return readKeyword("null", value: .null)
        default: return readNumber().map { .number($0) }
        }
    }

    // MARK: - One kind of value

    /// Reads an object, opening brace included.
    /// - Returns: the object, or `nil` when it is malformed.
    private mutating func readObject() -> PythonStyleJSON? {
        index += 1
        var members: [PythonStyleJSON.Member] = []
        skipWhitespace()
        if match("}") { return .object(members) }
        while true {
            skipWhitespace()
            guard let key = readString() else { return nil }
            skipWhitespace()
            guard match(":"), let value = readValue() else { return nil }
            members.append(PythonStyleJSON.Member(key: key, value: value))
            skipWhitespace()
            if match(",") { continue }
            return match("}") ? .object(members) : nil
        }
    }

    /// Reads an array, opening bracket included.
    /// - Returns: the array, or `nil` when it is malformed.
    private mutating func readArray() -> PythonStyleJSON? {
        index += 1
        var elements: [PythonStyleJSON] = []
        skipWhitespace()
        if match("]") { return .array(elements) }
        while true {
            guard let value = readValue() else { return nil }
            elements.append(value)
            skipWhitespace()
            if match(",") { continue }
            return match("]") ? .array(elements) : nil
        }
    }

    /// Reads a string and decodes every escape it holds.
    /// - Returns: the decoded string, or `nil` when it is malformed.
    private mutating func readString() -> String? {
        guard peek() == "\"" else { return nil }
        let start = index
        index += 1
        while index < scalars.count {
            let scalar = scalars[index]
            index += 1
            if scalar == "\\" {
                guard index < scalars.count else { return nil }
                index += 1
                continue
            }
            if scalar == "\"" {
                return Self.decoded(text(from: start))
            }
        }
        return nil
    }

    /// Reads a number and keeps the text of the source.
    /// - Returns: the text of the number, or `nil` when it is malformed.
    private mutating func readNumber() -> String? {
        let start = index
        while let scalar = peek(), Self.isNumberScalar(scalar) { index += 1 }
        guard index > start else { return nil }
        let number = text(from: start)
        return Double(number) == nil ? nil : number
    }

    /// Reads one of the three JSON keywords.
    /// - Parameters:
    ///   - keyword: the word to expect.
    ///   - value: the value the word stands for.
    /// - Returns: the value, or `nil` when the word is not there.
    private mutating func readKeyword(_ keyword: String, value: PythonStyleJSON)
        -> PythonStyleJSON?
    {
        let expected = Array(keyword.unicodeScalars)
        let end = index + expected.count
        guard end <= scalars.count, Array(scalars[index ..< end]) == expected else { return nil }
        index = end
        return value
    }

    // MARK: - Scalars

    /// The scalar to read next, or `nil` at the end.
    private func peek() -> Unicode.Scalar? {
        index < scalars.count ? scalars[index] : nil
    }

    /// Steps over one expected scalar.
    /// - Parameter scalar: the scalar to expect.
    /// - Returns: whether the scalar was there.
    private mutating func match(_ scalar: Unicode.Scalar) -> Bool {
        guard peek() == scalar else { return false }
        index += 1
        return true
    }

    /// The text between one point and the point already read.
    /// - Parameter start: the first scalar to take.
    /// - Returns: the text.
    private func text(from start: Int) -> String {
        String(String.UnicodeScalarView(scalars[start ..< index]))
    }

    /// Whether a scalar can be part of a JSON number.
    /// - Parameter scalar: the scalar to test.
    /// - Returns: whether it belongs to a number.
    private static func isNumberScalar(_ scalar: Unicode.Scalar) -> Bool {
        ("0" ... "9").contains(scalar) || scalar == "-" || scalar == "+" || scalar == "."
            || scalar == "e" || scalar == "E"
    }

    /// Decodes a whole JSON string token, quotation marks included.
    ///
    /// `JSONSerialization` owns the escape rules, the `\u` escape and the
    /// surrogate pair among them, thus this reader does not repeat them.
    ///
    /// - Parameter token: the string token, quotation marks included.
    /// - Returns: the decoded string, or `nil` when the token is malformed.
    private static func decoded(_ token: String) -> String? {
        let value = try? JSONSerialization.jsonObject(
            with: Data(token.utf8), options: [.fragmentsAllowed])
        return value as? String
    }
}

// MARK: - Reading a Swift value

// A `ToolSpec` and the arguments of a ``ToolCall`` reach this file as a Swift
// `Dictionary`, which keeps no order at all. `JSONSerialization` writes such a
// dictionary in its hash order, thus the same tool schema comes out in a
// different member order in each process, and the `## Tools` section of the
// prompt stops matching the shape the model saw during training. The
// initializer below writes one fixed order instead.
//
// WHERE THE ORDER COMES FROM
//
// ``PythonStyleJSON/publishedMemberOrder`` is the order of the two published
// tool inputs `encoding/tests/test_input_1.json` and
// `encoding/tests/test_input_3.json` of `deepseek-ai/DeepSeek-V4-Flash`. Each
// object of those two files, and of the golden outputs beside them, comes back
// byte for byte under that one order.
//
// WHAT THE ORDER CANNOT GIVE BACK
//
// A name the list does not hold sorts by name. Two such names carry real order
// in the published files — the members of a `properties` object, and the
// arguments of one call — and a Swift `Dictionary` lost that order before this
// file saw the value. Sorting gives the same bytes on every run, which the hash
// order never did, and it gives the published bytes when the published order is
// the sorted order.

extension PythonStyleJSON {

    /// The member names of the published DeepSeek-V4 tool schemas, in the order
    /// the published files write them.
    private static let publishedMemberOrder = [
        "name", "type", "enum", "items", "description", "anyOf", "parameters",
        "properties", "required", "additionalProperties", "$schema", "default",
    ]

    /// Builds a value from the Swift form that a tool specification and the
    /// arguments of a tool call carry.
    ///
    /// The members of each object take ``publishedMemberOrder`` first and the
    /// order of their names after it, because a Swift `Dictionary` keeps no
    /// order of its own.
    ///
    /// - Parameter value: the value to read.
    init(sendable value: any Sendable) {
        switch value {
        case is NSNull:
            self = .null
        case let flag as Bool:
            self = .bool(flag)
        case let number as Int:
            self = .number(String(number))
        case let number as Double:
            self = .number(String(number))
        case let text as String:
            self = .string(text)
        case let elements as [any Sendable]:
            self = .array(elements.map { PythonStyleJSON(sendable: $0) })
        case let members as [String: any Sendable]:
            self = .object(Self.orderedMembers(of: members))
        default:
            self = .string(String(describing: value))
        }
    }

    /// Puts the members of one object in the order this file writes them.
    /// - Parameter members: the members, by name.
    /// - Returns: the members, ordered.
    private static func orderedMembers(of members: [String: any Sendable]) -> [Member] {
        members.keys.sorted(by: precedes).compactMap { key in
            members[key].map { Member(key: key, value: PythonStyleJSON(sendable: $0)) }
        }
    }

    /// Whether one member name comes before another.
    /// - Parameters:
    ///   - left: the name to place.
    ///   - right: the name to place it against.
    /// - Returns: whether `left` comes first.
    private static func precedes(_ left: String, _ right: String) -> Bool {
        let leftPlace = place(of: left)
        let rightPlace = place(of: right)
        return leftPlace == rightPlace ? left < right : leftPlace < rightPlace
    }

    /// The place of one member name in ``publishedMemberOrder``.
    /// - Parameter name: the name to look up.
    /// - Returns: the place, or the length of the list for a name it does not
    ///   hold.
    private static func place(of name: String) -> Int {
        publishedMemberOrder.firstIndex(of: name) ?? publishedMemberOrder.count
    }
}

// MARK: - Writing

extension PythonStyleJSON {

    /// The value written the way `json.dumps(value, ensure_ascii=False)` writes
    /// it: one line, `", "` between items and `": "` after a key.
    var pythonStyleText: String {
        switch self {
        case .null:
            return "null"
        case .bool(let flag):
            return flag ? "true" : "false"
        case .number(let text):
            return text
        case .string(let text):
            return Self.quoted(text)
        case .array(let elements):
            let written = elements.map(\.pythonStyleText)
            return "[" + written.joined(separator: Self.itemSeparator) + "]"
        case .object(let members):
            let written = members.map {
                Self.quoted($0.key) + Self.keySeparator + $0.value.pythonStyleText
            }
            return "{" + written.joined(separator: Self.itemSeparator) + "}"
        }
    }

    /// What Python puts between two items of an array or an object.
    private static let itemSeparator = ", "
    /// What Python puts between the key and the value of a member.
    private static let keySeparator = ": "
    /// The first scalar that Python writes as itself. Everything below it is a
    /// control character and takes an escape.
    private static let firstPrintableScalarValue: UInt32 = 0x20

    /// Wraps a string in quotation marks and escapes what Python escapes.
    /// - Parameter text: the string to write.
    /// - Returns: the JSON string token.
    private static func quoted(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            out += escaped(scalar)
        }
        return out + "\""
    }

    /// The escape Python writes for each scalar that has a short escape.
    private static let escapeMap: [Unicode.Scalar: String] = [
        "\"": "\\\"",
        "\\": "\\\\",
        "\u{08}": "\\b",
        "\u{09}": "\\t",
        "\u{0A}": "\\n",
        "\u{0C}": "\\f",
        "\u{0D}": "\\r",
    ]

    /// Writes one scalar the way Python writes it inside a string.
    ///
    /// `ensure_ascii=False` leaves every scalar at or above the space as
    /// itself, the quotation mark and the reverse solidus excepted. The
    /// solidus is NOT escaped, which is where Foundation and Python differ.
    /// A control character without a short escape takes the `\u00XX` form.
    ///
    /// - Parameter scalar: the scalar to write.
    /// - Returns: the text that stands for it.
    private static func escaped(_ scalar: Unicode.Scalar) -> String {
        if let escape = escapeMap[scalar] { return escape }
        guard scalar.value < firstPrintableScalarValue else { return String(scalar) }
        return String(format: "\\u%04x", scalar.value)
    }
}
