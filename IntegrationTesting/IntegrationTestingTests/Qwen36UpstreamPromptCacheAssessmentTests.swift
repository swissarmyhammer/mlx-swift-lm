// Copyright © 2026 Apple Inc.
//
// Real-weights measurement of the OFFICIAL upstream prompt-cache path on a
// genuine hybrid Mamba/attention checkpoint (`mlx-community/Qwen3.6-27B-mxfp4`).
//
// Upstream keeps its multi-turn cache-reuse machinery in
// `MLXLMCommon.ChatSession` (`PromptCacheReusePolicy`, `ExtendCachedPrefixRule`,
// `RewindToCommonPrefixRule`). Nothing in this file touches
// `MLXFoundationModels.PromptCache` or `PromptCacheChunks`, so the numbers it
// prints are what upstream does on its own.
//
// A unit test cannot answer this question. The Qwen3 family chat template
// renders a turn's generation region differently live and as history, so
// round 1's fed token sequence is not always a prefix of round 2's re-render.
// When that happens the cache silently does nothing while every unit-level
// assertion still passes. The two facts below therefore have to come from a
// real two-round run:
//
//   (a) Is round 2's rendered prompt a prefix extension of round 1's?
//       A template that rewrites an already-cached region defeats caching no
//       matter how good the reuse machinery is.
//   (b) Given a valid prefix, does upstream actually skip the reprocessing?
//       `GenerateCompletionInfo.promptTokenCount` reports the tokens really
//       fed this round, and `promptTime` the seconds spent on them, so a
//       reusing round shows both far below the control.
//
// The control is a fresh `ChatSession` seeded with the identical transcript
// through the `history:` initializer, which forces a full cold prefill of the
// same round 2 prompt (see `ChatSession.streamMap`'s `.history` case). It gives
// the "no reuse" baseline that round 2 is measured against.
//
// Every measurement line carries the `QWEN36 CACHE:` prefix so a run log can be
// grepped for the numbers alone.
//
// MEASURED RESULT (kanban 2ajc82t). Both facts fail, so the assertions below
// pin the measured behavior instead of the behavior we want:
//
//   round 1 rendered / fed = 4705 tokens, prefill 13.35 s
//   round 2 rendered = 4748 tokens, fed 4748, skipped 0, prefill 11.60 s
//   control fed = 4748 tokens, prefill 11.61 s
//   round 2 extends round 1 = false, common prefix 4703 of 4705
//
// (a) fails: the template rewrites round 1's last 2 tokens -- the
//     generation-priming tail -- when it re-renders that turn as history.
//     Round 1's prompt ends with the priming block `<think>\n`, and round 2
//     writes the assistant reply text at that same offset instead. Upstream's
//     `ChatSession.Conversation.record` appends `.assistant(content:)` and
//     never fills `Chat.Message.reasoning`, so a history render cannot put the
//     priming block back.
// (b) fails: round 2 fed every rendered token and spent the same prefill time
//     as the cold control, to within 0.1 percent. Two causes, both real. The
//     broken prefix denies `ExtendCachedPrefixRule`, and the surviving
//     4703-token common prefix cannot rescue it either, because
//     `RewindToCommonPrefixRule` needs `PromptCacheState.isTrimmable` and a
//     hybrid Mamba/attention stack holds recurrent layers that cannot rewind.
//     `PromptCacheReusePolicy` therefore returns `.rebuild`.
//
// This suite is a baseline, not a wish. When a later change makes upstream
// reuse the cache on this checkpoint, these assertions fail on purpose -- that
// is the signal to invert them and to revisit kanban 2ajc82t.
//
// Run explicitly via:
// `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -only-testing:IntegrationTestingTests/Qwen36UpstreamPromptCacheAssessmentTests`

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLX
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// The hybrid Mamba/attention checkpoint under measurement. Matches
/// `TestFixtures.qwen36HybridModelID`, which the FoundationModels-side
/// `PromptCacheHybridReuseTests` uses, so both suites measure the same weights.
private let qwen36HybridCheckpointID = "mlx-community/Qwen3.6-27B-mxfp4"

/// Prefix that makes every measurement line greppable in a run log.
private let measurementPrefix = "QWEN36 CACHE:"

/// Sentences in the round 1 prompt. Sized so the rendered prompt clears
/// ``minimumPrefillTokenCount`` by a wide margin, which makes prefill -- not
/// decoding -- dominate each round's wall clock.
private let longPromptSentenceCount = 140

/// Tokens the round 1 prompt must reach for the timing comparison to mean
/// anything. Below this, prefill is too cheap to separate reuse from a rebuild.
private let minimumPrefillTokenCount = 1500

/// Tokens each round may generate. Small enough that decoding stays a minor
/// part of a round, large enough for the model to close its reply.
private let generatedTokenBudget = 24

/// How far back from the end of round 1's rendered prompt the template's
/// re-render is allowed to diverge. The measured divergence is the
/// generation-priming tail alone, so a larger one would be a different defect
/// -- a rewrite reaching into the conversation body -- and must be investigated
/// rather than absorbed.
private let primingDivergenceTokenLimit = 8

/// Divergent tokens printed from each side of the turn seam, enough to show the
/// priming block without dumping the whole prompt.
private let divergenceReportTokenCount = 12

/// The share of the cold control's prefill time that round 2 must reach for the
/// run to count as "reprocessed everything". Reuse would drop round 2 far below
/// this, so falling under it means upstream started skipping work.
private let noReuseTimeFloorFraction = 0.8

/// Greedy decoding keeps both rounds and the control reproducible.
private let greedyTemperature: Float = 0

/// What one measured round of `ChatSession` reports.
private struct RoundMeasurement {
    /// The text the round generated.
    let text: String
    /// Tokens actually fed to the model this round, after any cache reuse
    /// narrowed the input.
    let promptTokenCount: Int
    /// Seconds spent processing those tokens.
    let promptTime: TimeInterval
    /// Tokens the round generated.
    let generationTokenCount: Int
}

/// Runs one round on `session` and returns what the model reported for it.
///
/// - Parameters:
///   - session: the session to continue; its cache state carries into the round
///   - prompt: the user turn to append
/// - Returns: the round's text and its prompt/generation measurements.
/// - Throws: ``IntegrationTestFailure`` when the round reports no completion
///   info, and any error the session raises.
private func measureRound(
    _ session: ChatSession, prompt: String
) async throws -> RoundMeasurement {
    var text = ""
    var completionInfo: GenerateCompletionInfo?
    for try await generation in session.streamDetails(to: prompt) {
        if let chunk = generation.chunk {
            text += chunk
        }
        if let info = generation.info {
            completionInfo = info
        }
    }
    guard let completionInfo else {
        throw IntegrationTestFailure("a round ended without a GenerateCompletionInfo")
    }
    return RoundMeasurement(
        text: text,
        promptTokenCount: completionInfo.promptTokenCount,
        promptTime: completionInfo.promptTime,
        generationTokenCount: completionInfo.generationTokenCount)
}

/// Renders `messages` through the model's own chat template and returns the
/// resulting prompt tokens.
///
/// This is the same `UserInputProcessor.prepare(input:)` call `ChatSession`
/// makes for every turn, so the tokens here are the tokens a turn would feed
/// with a cold cache.
///
/// - Parameters:
///   - container: the loaded model container
///   - messages: the conversation to render
/// - Returns: the rendered prompt token ids.
/// - Throws: any error the processor raises.
private func renderPromptTokens(
    _ container: LLModelContainer, messages: [Chat.Message]
) async throws -> [Int] {
    // `[Chat.Message]` is not `Sendable`, so it crosses the container boundary
    // through the `nonSendable:` overload -- the same overload upstream's own
    // `MLXLanguageModel.Executor.respond` uses for its messages.
    try await container.perform(nonSendable: messages) { context, messages in
        let input = try await context.processor.prepare(input: UserInput(chat: messages))
        return input.text.tokens.asArray(Int.self)
    }
}

/// The number of leading tokens `first` and `second` share.
private func commonPrefixLength(_ first: [Int], _ second: [Int]) -> Int {
    zip(first, second).prefix { $0 == $1 }.count
}

/// Decodes `tokens` through the model's own tokenizer, so a token range reads
/// as the template markup it came from.
///
/// - Parameters:
///   - container: the loaded model container
///   - tokens: the token ids to decode
/// - Returns: the decoded text.
private func decodeTokens(_ container: LLModelContainer, tokens: [Int]) async -> String {
    await container.perform { context in
        context.tokenizer.decode(tokenIds: tokens)
    }
}

/// The tokens `render` writes from `index` onward, capped at
/// ``divergenceReportTokenCount`` so the report stays readable.
///
/// - Parameters:
///   - render: a rendered prompt
///   - index: the first divergent position
/// - Returns: the capped divergent tail, empty when `index` is past the end.
private func divergentTail(of render: [Int], from index: Int) -> [Int] {
    guard index < render.count else { return [] }
    return Array(render[index..<min(index + divergenceReportTokenCount, render.count)])
}

/// A long user turn whose content the model has to read, so a real prefill of
/// several thousand tokens happens rather than a template-only prompt.
private func makeLongPrompt() -> String {
    let sentences = (1...longPromptSentenceCount).map { index in
        "Fact \(index): the maintenance log for pump \(index) records a steady flow rate, "
            + "a nominal bearing temperature, and no alarms during the night shift."
    }
    return "Read this maintenance log. Do not summarize it yet.\n"
        + sentences.joined(separator: "\n")
}

/// Releases the checkpoint and the GPU buffer pool after the measurement, so a
/// 14 GB model does not stay resident for the rest of the run.
private func releaseCheckpoint(_ configuration: ModelConfiguration) async {
    await models.evictLLM(configuration)
    Stream.gpu.synchronize()
    Memory.clearCache()
}

/// Measures whether upstream's own `ChatSession` path reuses a prompt cache
/// across turns on a real hybrid Mamba/attention Qwen3.6 checkpoint.
@Suite(.serialized, .timeLimit(.minutes(60)))
struct Qwen36UpstreamPromptCacheAssessmentTests {

    @Test("Upstream ChatSession's second turn against Qwen3.6")
    func secondTurnPrefillCostOnHybridCheckpoint() async throws {
        let configuration = ModelConfiguration(id: qwen36HybridCheckpointID)
        let container = try await models.llmContainer(for: configuration)
        let parameters = GenerateParameters(
            maxTokens: generatedTokenBudget, temperature: greedyTemperature)

        let firstPrompt = makeLongPrompt()
        let secondPrompt = "Now name the pump in fact 7."
        let firstRendered = try await renderPromptTokens(container, messages: [.user(firstPrompt)])

        let session = ChatSession(container, generateParameters: parameters)
        let firstRound = try await measureRound(session, prompt: firstPrompt)

        // The transcript `ChatSession.Conversation.record` builds for round 2:
        // the user turn, the assistant reply as plain content, then the new
        // user turn. Rendering it here reproduces round 2's cold prompt.
        let secondRendered = try await renderPromptTokens(
            container,
            messages: [.user(firstPrompt), .assistant(firstRound.text), .user(secondPrompt)])
        let secondRound = try await measureRound(session, prompt: secondPrompt)

        let controlSession = ChatSession(
            container,
            history: [.user(firstPrompt), .assistant(firstRound.text)],
            generateParameters: parameters)
        let controlRound = try await measureRound(controlSession, prompt: secondPrompt)

        let isPrefixExtension = secondRendered.starts(with: firstRendered)
        let sharedPrefix = commonPrefixLength(firstRendered, secondRendered)
        let skippedTokenCount = secondRendered.count - secondRound.promptTokenCount
        let firstTail = await decodeTokens(
            container, tokens: divergentTail(of: firstRendered, from: sharedPrefix))
        let secondTail = await decodeTokens(
            container, tokens: divergentTail(of: secondRendered, from: sharedPrefix))
        report(
            firstRendered: firstRendered, firstRound: firstRound,
            secondRendered: secondRendered, secondRound: secondRound,
            controlRound: controlRound,
            isPrefixExtension: isPrefixExtension, sharedPrefix: sharedPrefix,
            skippedTokenCount: skippedTokenCount,
            firstTail: firstTail, secondTail: secondTail)

        #expect(!firstRound.text.isEmpty, "Round 1 should produce some response text")
        #expect(!secondRound.text.isEmpty, "Round 2 should produce some response text")
        #expect(
            firstRendered.count >= minimumPrefillTokenCount,
            """
            Round 1's rendered prompt (\(firstRendered.count) tokens) must reach \
            \(minimumPrefillTokenCount) tokens for prefill to dominate the timing
            """)

        // Fact (a): the template does NOT re-render round 1's turn verbatim,
        // and the rewrite stays inside the generation-priming tail.
        #expect(
            !isPrefixExtension,
            """
            (a) Round 2's rendered prompt now extends round 1's. The template stopped \
            rewriting the priming tail, so prefix reuse became reachable -- invert this \
            assertion and revisit kanban 2ajc82t.
            """)
        #expect(
            firstRendered.count - sharedPrefix <= primingDivergenceTokenLimit,
            """
            (a) The template rewrote the last \(firstRendered.count - sharedPrefix) of round \
            1's \(firstRendered.count) rendered tokens, past the \
            \(primingDivergenceTokenLimit)-token priming tail. A rewrite that reaches into \
            the conversation body is a different defect than the turn seam this suite \
            measures. Round 1 wrote <<<\(firstTail)>>> where round 2 writes <<<\(secondTail)>>>.
            """)

        // Fact (b): upstream reprocesses the whole prompt anyway. Note that a
        // 4703-token common prefix survives, and upstream still cannot use it:
        // `RewindToCommonPrefixRule` needs a trimmable cache, and this hybrid
        // stack holds recurrent layers that cannot rewind.
        #expect(
            skippedTokenCount == 0,
            """
            (b) Round 2 skipped \(skippedTokenCount) of its \(secondRendered.count) rendered \
            tokens; it fed every one of them when this baseline was taken. Reuse appeared -- \
            invert this assertion and revisit kanban 2ajc82t.
            """)
        #expect(
            secondRound.promptTime >= controlRound.promptTime * noReuseTimeFloorFraction,
            """
            (b) Round 2 spent \(secondRound.promptTime) s on prefill against the cold \
            control's \(controlRound.promptTime) s for the same prompt, dropping below \
            \(noReuseTimeFloorFraction) of the control. Round 2 stopped reprocessing the \
            whole prompt -- invert this assertion and revisit kanban 2ajc82t.
            """)

        await releaseCheckpoint(configuration)
    }

    /// Prints every measured number under ``measurementPrefix``.
    ///
    /// - Parameters:
    ///   - firstRendered: round 1's cold rendered prompt tokens
    ///   - firstRound: round 1's measurements
    ///   - secondRendered: round 2's cold rendered prompt tokens
    ///   - secondRound: round 2's measurements
    ///   - controlRound: the fresh-session control's measurements
    ///   - isPrefixExtension: whether round 2's render extends round 1's
    ///   - sharedPrefix: leading tokens the two renders share
    ///   - skippedTokenCount: rendered tokens round 2 did not feed
    ///   - firstTail: round 1's decoded text from the first divergent token
    ///   - secondTail: round 2's decoded text from the same position
    private func report(
        firstRendered: [Int], firstRound: RoundMeasurement,
        secondRendered: [Int], secondRound: RoundMeasurement,
        controlRound: RoundMeasurement,
        isPrefixExtension: Bool, sharedPrefix: Int, skippedTokenCount: Int,
        firstTail: String, secondTail: String
    ) {
        print("\(measurementPrefix) round 1 rendered prompt tokens = \(firstRendered.count)")
        print("\(measurementPrefix) round 1 fed prompt tokens = \(firstRound.promptTokenCount)")
        print("\(measurementPrefix) round 1 prefill seconds = \(firstRound.promptTime)")
        print("\(measurementPrefix) round 1 generated tokens = \(firstRound.generationTokenCount)")
        print("\(measurementPrefix) round 2 rendered prompt tokens = \(secondRendered.count)")
        print("\(measurementPrefix) round 2 extends round 1 = \(isPrefixExtension)")
        print(
            "\(measurementPrefix) round 2 common prefix with round 1 = \(sharedPrefix) "
                + "of \(firstRendered.count)")
        print("\(measurementPrefix) round 2 fed prompt tokens = \(secondRound.promptTokenCount)")
        print("\(measurementPrefix) round 2 prefill seconds = \(secondRound.promptTime)")
        print("\(measurementPrefix) round 2 tokens skipped by reuse = \(skippedTokenCount)")
        print("\(measurementPrefix) control fed prompt tokens = \(controlRound.promptTokenCount)")
        print("\(measurementPrefix) control prefill seconds = \(controlRound.promptTime)")
        print("\(measurementPrefix) round 1 divergent tail = <<<\(firstTail)>>>")
        print("\(measurementPrefix) round 2 divergent tail = <<<\(secondTail)>>>")
    }
}
