// Copyright © 2026 Apple Inc.
//
// The real-weights measurement of card ^z5xrzg6: the model writes
// `</｜DSML｜inv>` where the DSML syntax states `</｜DSML｜invoke>`, thus the
// identifier of the piece `oke` (5406) never reaches the tool-call parser.
//
// The card ACCEPTS that loss rather than corrects it. `DSMLToolCallParser` now
// accepts the short tag, thus this suite records the divergence rather than
// reports a defect: it asserts that the greedy answer carries a closing tag the
// parser reads, whether the checkpoint writes the short tag or the whole one.
//
// `DeepseekV4IntegrationTests.aShortToolPromptEmitsOneDSMLToolCall` runs the
// same conversation through `ChatSession` and reports the TEXT. Text cannot
// answer the two questions this suite keeps a record of:
//
//   1. Does the same run give the same identifiers every time? A greedy decode
//      is deterministic, thus a difference between two runs would name a
//      non-deterministic path rather than a stable property of the checkpoint.
//   2. How far below the winner does the absent identifier stand? The gap is
//      the size of the divergence, thus a run that measures a smaller gap tells
//      the next reader that the checkpoint, or the numerics, moved.
//
// This suite drives the model directly rather than through `TokenIterator`,
// because `TokenIterator` answers identifiers and never logits. It reads the
// same prompt, the same greedy rule and the same token budget.
//
// The identifiers of the closing tag come from the published tokenizer of the
// checkpoint, thus this file states no vocabulary number of its own.
//
// Every `print` of the run stays. It is the record of the accepted divergence,
// thus a future reader gets the identifiers, the candidates and the logits back
// from one run of the suite.
//
// The checkpoint holds 141 GiB. This suite awaits the one shared load of
// `DeepseekV4SharedCheckpoint.swift`, thus a process loads it at most once.
// Run this suite ALONE:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/DeepseekV4ToolCallTokenDiagnosticTests`
//
// `swift test` is BLIND to this file. No SwiftPM target holds
// `IntegrationTesting/`, thus `swift build --build-tests` stays at exit 0 with
// a type error in it. Use the `xcodebuild build-for-testing` command of
// `DeepseekV4SharedCheckpoint.swift` as the compile evidence for any change.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

// MARK: - Constants

/// The per-test time limit of this suite, in minutes. Each test loads the
/// checkpoint once and then decodes two generations of a few dozen tokens.
private let suiteTimeLimitMinutes = 240

/// The number of identifiers one greedy run may produce. A DSML block for one
/// call with one parameter is far shorter than this.
private let generationStepLimit = 128

/// The number of candidates a step report names.
private let reportedCandidateCount = 5

/// The number of identifiers of the closing tag that stand before the piece
/// this suite tracks: `</`, the DSML marker, and `inv`.
private let piecesBeforeTheTrackedOne = 3

/// Prefix that makes every measurement line greppable in a run log.
private let measurementPrefix = "DSV4 TOKEN:"

// MARK: - One decode step

/// What one greedy decode step chose, and where the tracked identifier stood.
private struct GreedyStep {
    /// The identifier the step chose.
    let identifier: Int
    /// The candidates of the highest logits, best first.
    let candidates: [(identifier: Int, logit: Float)]
    /// The logit of the identifier this run tracks.
    let trackedLogit: Float
    /// The number of identifiers whose logit stands above the tracked one.
    /// Zero means the tracked identifier won the step.
    let trackedRank: Int

    /// The logit of the identifier the step chose.
    var chosenLogit: Float { candidates.first?.logit ?? .nan }

    /// How far the tracked identifier stands below the winner.
    var trackedGap: Float { chosenLogit - trackedLogit }
}

/// One line of a step report.
///
/// - Parameters:
///   - step: the step to report.
///   - label: what the step is, for the head of the line.
/// - Returns: the line.
private func report(_ step: GreedyStep, label: String) -> String {
    let candidates = step.candidates
        .map { "\($0.identifier) at \($0.logit)" }
        .joined(separator: ", ")
    return "\(measurementPrefix) \(label): chose \(step.identifier); "
        + "candidates [\(candidates)]; tracked identifier at logit \(step.trackedLogit), "
        + "rank \(step.trackedRank), \(step.trackedGap) below the winner"
}

// MARK: - The greedy run

/// Runs one greedy generation over `promptIdentifiers` and records each step.
///
/// The run reads the model directly. It keeps one cache, feeds the whole
/// prompt in one call, and then feeds the identifier it chose, which is the
/// shape `TokenIterator` decodes in.
///
/// - Parameters:
///   - model: the loaded DeepSeek-V4 model.
///   - promptIdentifiers: the rendered prompt.
///   - tracked: the identifier whose logit each step records.
/// - Returns: one entry for each decode step, in order.
/// - Throws: any error the cache builder raises.
private func greedyRun(
    model: DeepSeekV4Model, promptIdentifiers: [Int], tracking tracked: Int
) throws -> [GreedyStep] {
    let cache = try model.newCache(parameters: nil)
    var block = MLXArray(promptIdentifiers.map(Int32.init))
        .reshaped(1, promptIdentifiers.count)
    var steps: [GreedyStep] = []
    for _ in 0 ..< generationStepLimit {
        let logits = model(block, cache: cache)[0, -1].asType(.float32)
        let step = measure(logits: logits, tracking: tracked, settling: cache)
        steps.append(step)
        block = MLXArray([Int32(step.identifier)]).reshaped(1, 1)
    }
    return steps
}

/// Reads one logit row into a step report.
///
/// - Parameters:
///   - logits: the logit of each identifier, in float32.
///   - tracked: the identifier whose logit the report records.
///   - cache: the caches to settle beside the report, so that the lazy graph
///     stays bounded over a long run.
/// - Returns: the step report.
private func measure(
    logits: MLXArray, tracking tracked: Int, settling cache: [KVCache]
) -> GreedyStep {
    let ranked = argPartition(-logits, kth: reportedCandidateCount - 1)[
        0 ..< reportedCandidateCount]
    let trackedLogit = logits[tracked]
    let higher = (logits .> trackedLogit).sum()
    eval([logits, ranked, trackedLogit, higher] + cache.flatMap { $0.state })

    let candidates = ranked.asArray(Int32.self)
        .map { (identifier: Int($0), logit: logits[Int($0)].item(Float.self)) }
        .sorted { $0.logit > $1.logit }
    return GreedyStep(
        identifier: candidates[0].identifier,
        candidates: candidates,
        trackedLogit: trackedLogit.item(Float.self),
        trackedRank: Int(higher.item(Int32.self)))
}

/// The index at which `run` holds `pattern`.
///
/// - Parameters:
///   - pattern: the run of identifiers to find.
///   - run: the identifiers to search.
/// - Returns: the index of the first identifier of the first match, or `nil`
///   when the run holds no match.
private func firstIndex(of pattern: [Int], in run: [Int]) -> Int? {
    guard !pattern.isEmpty, run.count >= pattern.count else { return nil }
    for start in 0 ... (run.count - pattern.count)
    where Array(run[start ..< (start + pattern.count)]) == pattern {
        return start
    }
    return nil
}

// MARK: - The suite

/// The real-weights record of the identifier the tool round loses.
///
/// Card ^z5xrzg6 ACCEPTS the loss: `DSMLToolCallParser` reads the short closing
/// tag `</｜DSML｜inv>` beside the whole one, thus the round completes although
/// identifier 5406 never comes. This suite is the record of that divergence. It
/// keeps every measurement print, and it asserts what the round needs: two
/// greedy runs that agree, and a closing tag the parser reads.
///
/// See the file header for the gating rules and the run command.
@Suite(.serialized, .timeLimit(.minutes(suiteTimeLimitMinutes)))
struct DeepseekV4ToolCallTokenDiagnosticTests {

    /// The greedy run writes a closing tag that the DSML parser reads, and it
    /// writes the same identifiers every time.
    ///
    /// The measurement the run prints holds the record of the card: the
    /// identifiers the model wrote, whether two runs agree, and how far the
    /// absent identifier stood below the winner at the step that lost it.
    @Test func theGreedyRunWritesAClosingTagTheParserReads() async throws {
        guard
            let container = await deepseekV4ContainerOrSkip(
                testName: "theGreedyRunWritesAClosingTagTheParserReads")
        else { return }

        let closeTag = "</\(DeepSeekV4ChatEncoder.SpecialToken.dsml)invoke>"
        let shortCloseTag = "</\(DeepSeekV4ChatEncoder.SpecialToken.dsml)inv>"
        let promptIdentifiers = try await renderPromptTokens(
            container,
            messages: [.system(stockAgentInstructions), .user(stockToolUserPrompt)],
            tools: [stockToolSpec],
            additionalContext: ["thinking": false])

        let (closeTagIdentifiers, first, second, text) = try await container.perform { context in
            let identifiers = DeepSeekV4Tokenization(vocabulary: context.tokenizer)
                .identifiers(of: closeTag)
            let tracked = identifiers[piecesBeforeTheTrackedOne]
            guard let model = context.model as? DeepSeekV4Model else {
                throw IntegrationTestFailure(
                    "the loaded model must be a DeepSeekV4Model, and it is "
                        + "\(type(of: context.model))")
            }
            let first = try greedyRun(
                model: model, promptIdentifiers: promptIdentifiers, tracking: tracked)
            let second = try greedyRun(
                model: model, promptIdentifiers: promptIdentifiers, tracking: tracked)
            let text = context.tokenizer.decode(tokenIds: first.map(\.identifier))
            return (identifiers, first, second, text)
        }

        let generated = first.map(\.identifier)
        let opening = Array(closeTagIdentifiers.prefix(piecesBeforeTheTrackedOne))
        print("\(measurementPrefix) prompt identifiers = \(promptIdentifiers.count)")
        print("\(measurementPrefix) closing tag identifiers = \(closeTagIdentifiers)")
        print("\(measurementPrefix) generated identifiers = \(generated)")
        print("\(measurementPrefix) generated text = <<<\(text)>>>")
        if let start = firstIndex(of: opening, in: generated) {
            let losingStep = start + opening.count
            print("\(measurementPrefix) the closing tag opens at step \(start)")
            print(report(first[losingStep], label: "step \(losingStep) of run 1"))
            print(report(second[losingStep], label: "step \(losingStep) of run 2"))
        } else {
            print("\(measurementPrefix) the run wrote no closing tag at all")
        }
        print("\(measurementPrefix) the answer holds \(closeTag): \(text.contains(closeTag))")
        print(
            "\(measurementPrefix) the answer holds \(shortCloseTag): "
                + "\(text.contains(shortCloseTag))")

        #expect(
            generated == second.map(\.identifier),
            """
            a greedy decode is deterministic, thus the two runs must write the same \
            identifiers. Run 1 wrote \(generated.count) and run 2 wrote \(second.count).
            """)
        let calls = DSMLToolCallParser().parseEOS(text, tools: nil)
        #expect(
            calls.first?.function.name == stockToolName,
            """
            the answer must carry a closing tag the DSML parser reads for the tool round \
            to complete. The parser read \(calls). The model wrote: <<<\(text)>>>
            """)
    }
}
