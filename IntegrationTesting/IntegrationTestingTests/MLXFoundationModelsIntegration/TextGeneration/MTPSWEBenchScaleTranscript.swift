// Copyright © 2026 Apple Inc.
//
// The synthetic agent transcript of `MTPSWEBenchScaleTests`: a system prompt with a `runCode`
// tool, a bug report, and a long history of `runCode` calls that read Python modules of a
// repository and run its tests, in the shape of an ACP agent's SWE-bench session.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

/// The one argument of the `runCode` tool.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
struct SWEBenchRunCodeArguments {
    @Guide(description: "A bash command or a Python snippet to run at the repository root")
    var code: String
}

/// The synthetic agent transcript: realistic code and test output, each module and each test run
/// different from the others, thus the prompt is no run of repeated tokens.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum SWEBenchTranscript {

    /// The name of the tool the agent calls.
    static let toolName = "runCode"

    /// The tool the agent calls for every look at the repository.
    static let tool = Transcript.ToolDefinition(
        name: toolName,
        description: """
            Run a bash command or a Python snippet in the sandbox at the repository root, and \
            return its exit code, its standard output and its standard error.
            """,
        parameters: SWEBenchRunCodeArguments.generationSchema)

    /// The system prompt of the agent.
    static let systemPrompt = """
        You are a software engineering agent working in a Python repository checked out at \
        /workspace/repo. You fix the issue the user reports with the smallest correct change. \
        You inspect the repository only through the runCode tool: call it with one bash command \
        or one Python snippet at a time, read its output, and decide the next step from what you \
        read. Never guess the content of a file you have not printed. Keep the public API of \
        every module unchanged, keep the style of the surrounding code, and add or update tests \
        for every behavior you change. When the user asks you to write code in your reply, write \
        it in full in the reply instead of calling the tool.
        """

    /// The issue the agent works on.
    static let issue = """
        Issue #4127: totals are off by one cent. Several `total_*` helpers in the services under \
        src/ round their sums half to even instead of half up, so an order whose line items end in \
        half a cent comes out one cent short, and the tests in tests/ that check the totals of \
        such orders fail. Find every helper with this bug, explain the cause, and fix it without \
        changing the public functions. Start by reading the modules and running their tests.
        """

    /// Every `testRunInterval`th history round runs the tests of a module instead of reading one.
    private static let testRunInterval = 5

    /// The index of the first module a follow-up tool call prints, past every history module.
    private static let followUpModuleIndex = 10_000

    /// The modulus of every number a module states, so that each stays a small literal.
    private static let literalModulus = 997

    /// The multiplier that spreads the module index over each word list (Knuth's).
    private static let wordMixMultiplier: UInt = 2_654_435_761

    /// The stride that sets the word lists apart from one another.
    private static let wordSlotStride: UInt = 40_503

    /// The word lists, and the numbers, that each module draws from.
    private enum Slot: UInt {
        case domain = 1, package, verb, decimalField, textField, countField, limit, threshold
    }

    // The words a module draws its names from, one list for each kind of name.
    private static let domains = [
        "invoice", "shipment", "ledger_entry", "customer", "payment", "warehouse_bin", "coupon",
        "refund", "catalog_item", "subscription", "tax_rate", "carrier_quote", "audit_event",
        "price_rule", "return_label", "supplier", "gift_card", "payout",
    ]
    private static let packages = [
        "billing", "logistics", "accounting", "storefront", "fulfillment", "compliance",
        "pricing", "procurement", "payments", "reporting",
    ]
    private static let verbs = [
        "validate", "normalize", "merge", "serialize", "reconcile", "schedule", "archive",
        "summarize", "split", "dedupe", "rank", "expire", "allocate",
    ]
    private static let decimalFields = [
        "amount", "weight_kg", "discount", "unit_price", "tax", "balance", "fee", "credit",
    ]
    private static let textFields = [
        "region", "currency", "status", "owner", "sku", "channel", "warehouse", "tier",
    ]
    private static let countFields = [
        "quantity", "priority", "batch_size", "retry_count", "age_days", "line_count", "seats",
    ]

    /// The value module `index` draws for `slot`.
    private static func mixed(_ index: Int, _ slot: Slot) -> UInt {
        UInt(index) &* wordMixMultiplier &+ slot.rawValue &* wordSlotStride
    }

    /// The word module `index` draws from `words` for `slot`.
    private static func word(_ words: [String], _ index: Int, _ slot: Slot) -> String {
        words[Int(mixed(index, slot) % UInt(words.count))]
    }

    /// The number module `index` states for `slot`.
    private static func literal(_ index: Int, _ slot: Slot) -> Int {
        Int(mixed(index, slot) % UInt(literalModulus))
    }

    /// The names one module uses.
    private struct Module {
        let index: Int
        let domain: String
        let package: String
        let verb: String
        let amount: String
        let label: String
        let count: String

        init(_ index: Int) {
            self.index = index
            domain = word(domains, index, .domain)
            package = word(packages, index, .package)
            verb = word(verbs, index, .verb)
            amount = word(decimalFields, index, .decimalField)
            label = word(textFields, index, .textField)
            count = word(countFields, index, .countField)
        }

        var record: String {
            domain.split(separator: "_").map(\.capitalized).joined() + "Record"
        }
        var path: String { "src/\(package)/\(domain)_\(verb)_\(index).py" }
        var testPath: String { "tests/\(package)/test_\(domain)_\(index).py" }
    }

    /// The Python source of `module`.
    private static func source(of module: Module) -> String {
        let limit = literal(module.index, .limit)
        let threshold = literal(module.index, .threshold)
        let limitName = module.domain.uppercased() + "_LIMIT"
        let minimumName = "MINIMUM_" + module.count.uppercased()
        return #"""
            """\#(module.verb.capitalized) helpers for the \#(module.domain) rows of the \#(module.package) service."""
            from __future__ import annotations

            import dataclasses
            from collections import defaultdict
            from decimal import ROUND_HALF_EVEN, Decimal
            from typing import Iterable, Mapping

            \#(limitName) = \#(limit)
            \#(minimumName) = \#(threshold % (limit + 1))


            @dataclasses.dataclass(frozen=True)
            class \#(module.record):
                """One \#(module.domain) row as the \#(module.package) API returns it."""

                \#(module.amount): Decimal
                \#(module.label): str
                \#(module.count): int = 0

                def is_eligible(self) -> bool:
                    """Whether the row holds at least \#(minimumName) \#(module.count)."""
                    return self.\#(module.count) >= \#(minimumName)


            def \#(module.verb)_\#(module.domain)s(
                rows: Iterable[\#(module.record)], *, limit: int = \#(limitName)
            ) -> list[\#(module.record)]:
                """\#(module.verb.capitalized) the rows: drop those without a \#(module.label), keep at most ``limit``."""
                kept = [row for row in rows if row.\#(module.label).strip()]
                kept.sort(key=lambda row: (row.\#(module.amount), -row.\#(module.count)))
                return kept[:limit]


            def total_\#(module.amount)(rows: Iterable[\#(module.record)]) -> Decimal:
                """The sum of \#(module.amount) over ``rows``, rounded to cents."""
                total = sum((row.\#(module.amount) for row in rows), Decimal("0"))
                return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


            def group_by_\#(module.label)(rows: Iterable[\#(module.record)]) -> Mapping[str, list[\#(module.record)]]:
                """The eligible rows keyed by their lowercased \#(module.label), in input order."""
                groups: dict[str, list[\#(module.record)]] = defaultdict(list)
                for row in rows:
                    if row.is_eligible():
                        groups[row.\#(module.label).lower()].append(row)
                return dict(groups)
            """#
    }

    /// The pytest output of a test run of `module`, with the rounding failure of the issue.
    private static func testRun(of module: Module) -> String {
        let euros = literal(module.index, .limit)
        let cents = literal(module.index, .threshold)
        let test = "test_total_\(module.amount)_rounds_half_up"
        return #"""
            ============================= test session starts ==============================
            platform darwin -- Python 3.12.4, pytest-8.3.2, pluggy-1.5.0
            rootdir: /workspace/repo
            collected 4 items

            \#(module.testPath) ..F.                                        [100%]

            =================================== FAILURES ===================================
            ________________________ \#(test) ________________________

                def \#(test)():
                    rows = [
                        \#(module.record)(Decimal("\#(euros).005"), "north", \#(cents)),
                        \#(module.record)(Decimal("\#(cents).010"), "south", \#(euros)),
                    ]
            >       assert total_\#(module.amount)(rows) == Decimal("\#(euros + cents).02")
            E       AssertionError: assert Decimal('\#(euros + cents).01') == Decimal('\#(euros + cents).02')

            \#(module.testPath):\#(cents): AssertionError
            =========================== short test summary info ============================
            FAILED \#(module.testPath)::\#(test) - AssertionError
            ========================= 1 failed, 3 passed in 0.\#(euros)s ==========================
            """#
    }

    /// A text segment.
    static func segment(_ content: String) -> Transcript.Segment {
        .text(Transcript.TextSegment(content: content))
    }

    /// A user turn.
    static func prompt(_ content: String) -> Transcript.Entry {
        .prompt(Transcript.Prompt(segments: [segment(content)], responseFormat: nil))
    }

    /// A new instructions entry. Its identifier names a new session of the prompt cache.
    ///
    /// - Parameter declaresTool: Whether the instructions declare the `runCode` tool.
    /// - Returns: The entry.
    static func instructions(declaresTool: Bool = true) -> Transcript.Entry {
        .instructions(
            Transcript.Instructions(
                segments: [segment(systemPrompt)], toolDefinitions: declaresTool ? [tool] : []))
    }

    /// The text the tool gives back for a run that exited with `exitCode` and printed `output`.
    static func toolOutputText(exitCode: Int, output: String) -> String {
        "exit_code: \(exitCode)\nstdout:\n\(output)\nstderr:\n"
    }

    /// The output of the follow-up tool call of turn `turnIndex`: a module the history never read.
    static func followUpToolOutput(turnIndex: Int) -> String {
        toolOutputText(
            exitCode: 0, output: source(of: Module(followUpModuleIndex + turnIndex)))
    }

    /// The `runCode` command and its output of history round `index`.
    private static func exchange(_ index: Int) -> (command: String, output: String) {
        let module = Module(index)
        guard index % testRunInterval == testRunInterval - 1 else {
            return (
                "sed -n '1,200p' \(module.path)",
                toolOutputText(exitCode: 0, output: source(of: module))
            )
        }
        return (
            "python -m pytest -q \(module.testPath)",
            toolOutputText(exitCode: 1, output: testRun(of: module))
        )
    }

    /// The `runCode` command and its output of history round `index`, as one text: the content
    /// a benchmark that tokenizes the history without the chat template reads.
    static func exchangeText(_ index: Int) -> String {
        let (command, output) = exchange(index)
        return command + "\n" + output
    }

    /// The tool call and the tool output of history round `index`.
    private static func round(_ index: Int) throws -> [Transcript.Entry] {
        let (command, output) = exchange(index)
        let callID = "call_history_\(index)"
        let arguments = String(
            decoding: try JSONEncoder().encode(["code": command]), as: UTF8.self)
        let call = Transcript.ToolCall(
            id: callID, toolName: toolName, arguments: try GeneratedContent(json: arguments))
        return [
            .toolCalls(Transcript.ToolCalls(id: "toolcalls_history_\(index)", [call])),
            .toolOutput(
                Transcript.ToolOutput(id: callID, toolName: toolName, segments: [segment(output)])),
        ]
    }

    /// The bug report and as many history rounds as reach `minimumTokenCount` tokens of content,
    /// without the instructions entry, which each session makes its own.
    ///
    /// - Parameters:
    ///   - minimumTokenCount: The tokens the content of the history must reach.
    ///   - maximumRoundCount: The most rounds the history may hold.
    ///   - countTokens: The tokens of one text, through the tokenizer of the target.
    /// - Returns: The entries.
    /// - Throws: When `maximumRoundCount` rounds do not reach `minimumTokenCount`.
    static func history(
        minimumTokenCount: Int, maximumRoundCount: Int, countTokens: (String) -> Int
    ) throws -> [Transcript.Entry] {
        let roundTokens = (0 ..< maximumRoundCount).map { index in
            let (command, output) = exchange(index)
            return countTokens(command) + countTokens(output)
        }
        let reached = roundTokens.reduce(into: [countTokens(issue)]) {
            $0.append(($0.last ?? 0) + $1)
        }
        let roundCount = try #require(
            reached.firstIndex { $0 >= minimumTokenCount },
            "\(maximumRoundCount) history rounds stay under \(minimumTokenCount) tokens")
        return [prompt(issue)] + (try (0 ..< roundCount).flatMap(round))
    }
}

#endif
