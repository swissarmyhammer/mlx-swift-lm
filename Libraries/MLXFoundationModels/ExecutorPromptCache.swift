// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import os

// MARK: - Planning one generation pass

/// What one generation pass does with the prompt cache of its session.
struct ExecutorPromptCachePlan {

    /// The caches to hand to generation, and, once the generation ended, the
    /// caches it left (see ``holding(generationCache:)``).
    private(set) var caches: [KVCache]

    /// The input to feed, narrowed to the tokens `caches` does not already hold.
    let input: LMInput

    /// The leading prompt tokens `caches` already holds, thus the tokens this
    /// pass does not feed to the model.
    let reusedTokenCount: Int

    /// The whole rendered prompt of this pass.
    let promptTokens: [Int]

    /// The tokens `caches` represents once this pass has fed `input`, before
    /// generation. This is `promptTokens` on the standard path. A protocol rule
    /// that splices past a committed turn names the tokens the model wrote
    /// instead, plus the tail of the render it fed.
    let representedTokens: [Int]

    /// The model state to seed the iterator with: the state the entry carried
    /// when this pass reuses its caches, and nil when the pass builds fresh
    /// caches.
    let state: LMOutput.State?

    /// The rule that decided this plan, which the log line of the pass names.
    let decision: ExecutorPromptCacheDecision

    /// The tokens that this pass fed to the model before generation started,
    /// to take a checkpoint between them and `input`: prompt tokens, and the
    /// ``uncommittedTokens`` that go ahead of them. Zero when the pass feeds
    /// the whole tail through `input`.
    ///
    /// These tokens are not reused: the pass fed them. They are not in
    /// `input` either, thus the usage of the pass adds them back.
    var prefilledTokenCount = 0

    /// The tokens the previous pass generated and the carried caches did not
    /// hold, which this pass feeds ahead of the prompt tail (see
    /// ``ExecutorPromptCacheEntry/uncommittedTokens``). They go first in the
    /// feed to the checkpoint when the pass takes one, and first in `input`
    /// otherwise. They end the previous answer, thus the usage of this pass
    /// does not count them as prompt tokens.
    var uncommittedTokens: [Int] = []

    /// The checkpoint this pass took before generation, which the entry it
    /// commits carries to the next turn, or nil.
    var checkpoint: PromptCacheCheckpoint?

    /// The MTP drafter continuation the generation of this pass resumes,
    /// aligned with what `caches` hold before `input` is fed, or nil for a
    /// drafter that starts cold.
    ///
    /// The plan hands it to the generation beside `caches`, never as model
    /// state: the target never reads it.
    var drafterState: MTPDrafterContinuation?

    /// Plans what `entry` may serve for `input`, building fresh caches when it
    /// may serve nothing.
    ///
    /// The protocol rules and the standard rules decide first. When none of
    /// them reuses the caches, the checkpoint the entry carries can still
    /// serve a render that agrees with the entry up to that checkpoint.
    ///
    /// The drafter state of the entry comes along only when the render
    /// extends the ledger whole or a protocol rule splices onto it, thus the
    /// caches stay where the ledger ends, and the drafter `drafterID` made the
    /// state.
    /// The key of a session names no drafter, thus a session that changed its
    /// drafter finds the state of the old one and starts the new one cold. A
    /// checkpoint restore takes the drafter state of the checkpoint.
    ///
    /// - Parameters:
    ///   - entry: the cache the session carries, or nil for a cold session.
    ///   - input: the prepared input of the pass about to run.
    ///   - model: the model that owns the cache shape.
    ///   - parameters: the generation parameters the new caches must match.
    ///   - protocolRules: the cache-reuse rules of the model's response
    ///     protocol, consulted before the standard prefix rules.
    ///   - drafterID: the identity of the drafter of the pass, or nil when the
    ///     pass does not speculate with a resumable drafter.
    /// - Returns: the plan, or nil when a carried cache cannot serve this input
    ///   at all. Media, a mask that hides a token and a batch of more than one
    ///   row each place content in the model's input that a token ledger cannot
    ///   describe; the caller then generates with no carried cache.
    static func make(
        reusing entry: ExecutorPromptCacheEntry?,
        input: LMInput,
        model: any LanguageModel,
        parameters: GenerateParameters,
        protocolRules: [any PromptCacheReuseRule] = [],
        drafterID: MTPDrafterID? = nil
    ) throws -> ExecutorPromptCachePlan? {
        guard let promptTokens = ledgerTokens(of: input), !promptTokens.isEmpty else {
            return nil
        }

        if let entry,
            let reuse = reconcilePromptCache(
                promptTokens: promptTokens, cachedTokens: entry.tokens,
                previousRenderTokens: entry.renderTokens, caches: entry.caches,
                protocolRules: protocolRules, carriesModelState: entry.state != nil,
                previousGenerationUncommittedTokens: entry.uncommittedTokens),
            reuse.suffixStart < promptTokens.count
        {
            let fedTokens = reuse.fedTokens(of: promptTokens)
            var plan = ExecutorPromptCachePlan(
                caches: entry.caches,
                input: fedTokens == promptTokens ? input : narrowed(input, to: fedTokens),
                reusedTokenCount: reuse.suffixStart,
                promptTokens: promptTokens,
                representedTokens: reuse.representedTokens,
                state: entry.state,
                decision: decision(of: reuse, render: promptTokens, ledger: entry.tokens))
            plan.uncommittedTokens = reuse.uncommittedTokens
            // An extension and a splice both feed the render onto the caches
            // where the ledger ends, and the drafter state stands there too, as
            // `ChatSession` resumes it after either. A rewind moves the caches
            // away from the end of the ledger, and the drafter state cannot
            // move with them.
            if reuse.kind == .extend || reuse.kind == .splice {
                plan.drafterState = entry.drafterState(resumableBy: drafterID)
            }
            return plan
        }

        if let entry,
            let restored = restoring(
                entry, input: input, promptTokens: promptTokens, drafterID: drafterID)
        {
            return restored
        }

        return ExecutorPromptCachePlan(
            caches: try model.newCache(parameters: parameters),
            input: input,
            reusedTokenCount: 0,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: nil,
            decision: entry.map { .rebuild(.init(render: promptTokens, ledger: $0.tokens)) }
                ?? .cold)
    }

    /// This plan, holding the caches the generation of the pass left in the
    /// place of the caches it handed over.
    ///
    /// A KV-cache configuration replaces each cache it compresses with a new
    /// cache object while the generation runs, thus the caches the pass handed
    /// over can stop at the end of the prompt, in full precision, while the
    /// generation goes on in their compressed replacements. The next turn
    /// must carry the replacements.
    ///
    /// - Parameter generationCache: the caches the generation left, or nil
    ///   when it handed none over.
    /// - Returns: the plan with those caches, or this plan when there are none.
    func holding(generationCache: [KVCache]?) -> ExecutorPromptCachePlan {
        guard let generationCache else { return self }
        var plan = self
        plan.caches = generationCache
        return plan
    }

    /// The decision a reuse of a carried cache stands for.
    ///
    /// - Parameters:
    ///   - reuse: what the caches hold of the render once the decision is
    ///     applied.
    ///   - render: the whole rendered prompt.
    ///   - ledger: the tokens the carried caches held before the decision.
    /// - Returns: the decision, with the seam a rewind went back to.
    private static func decision(
        of reuse: PromptCacheReuse, render: [Int], ledger: [Int]
    ) -> ExecutorPromptCacheDecision {
        switch reuse.kind {
        case .prefill:
            // The carried caches hold nothing, thus the pass is cold in all
            // but name: the whole prompt is fed.
            return .cold
        case .extend:
            return .extend
        case .splice:
            return .splice
        case .rewind:
            return .rewind(ExecutorPromptCacheDivergence(render: render, ledger: ledger))
        }
    }

    /// The plan that takes the caches of `entry` back to the checkpoint the
    /// entry carries and feeds the render past it, or nil when the
    /// checkpoint cannot serve the render.
    ///
    /// ``make(reusing:input:model:parameters:protocolRules:drafterID:)`` asks for this
    /// plan only when no rule found a reuse, thus a checkpoint never takes
    /// the place of an extension, a splice or a rewind.
    ///
    /// - Parameters:
    ///   - entry: the cache the session carries.
    ///   - input: the prepared input of the pass about to run.
    ///   - promptTokens: the whole rendered prompt of `input`.
    ///   - drafterID: the identity of the drafter of the pass, or nil.
    /// - Returns: the plan, or nil. The plan resumes a copy of the drafter
    ///   state the checkpoint keeps, when the drafter `drafterID` made it.
    private static func restoring(
        _ entry: ExecutorPromptCacheEntry, input: LMInput, promptTokens: [Int],
        drafterID: MTPDrafterID?
    ) -> ExecutorPromptCachePlan? {
        guard let checkpoint = entry.checkpoint, checkpoint.serves(promptTokens),
            let caches = checkpoint.restored(onto: entry.caches)
        else {
            return nil
        }
        let reusedTokenCount = checkpoint.tokens.count
        var plan = ExecutorPromptCachePlan(
            caches: caches,
            input: narrowed(input, to: Array(promptTokens[reusedTokenCount...])),
            reusedTokenCount: reusedTokenCount,
            promptTokens: promptTokens,
            representedTokens: promptTokens,
            state: checkpoint.state,
            decision: .restore(.init(render: promptTokens, ledger: entry.tokens)))
        plan.drafterState = drafterID.flatMap { checkpoint.drafterContinuation(resumableBy: $0) }
        return plan
    }

    /// This plan, with the prompt fed up to `boundary` before generation, a
    /// checkpoint taken there, and only the rest of the prompt left in
    /// `input`.
    ///
    /// `boundary` is the index where a next render of the same messages
    /// plus one new message parts from this prompt: the end of the last
    /// message, before the generation prompt. A next turn that drops the
    /// generated turn agrees with this prompt up to there, thus the
    /// checkpoint serves it.
    ///
    /// - Parameters:
    ///   - boundary: the index of the prompt where the checkpoint goes.
    ///   - model: the model that feeds the caches.
    ///   - parameters: the generation parameters. Their prefill parameters
    ///     cut the fed tokens into forward passes, as a prefill does.
    ///   - speculation: the MTP drafter of the pass, or nil. The fed tokens go
    ///     into the drafter too, from `drafterState` when it stands where the
    ///     caches stand and from a cold drafter state there otherwise, thus the
    ///     checkpoint keeps the drafter state at `boundary` and the generation
    ///     resumes it there.
    /// - Returns: the plan to generate with. This plan as it is when
    ///   `boundary` is not past the reused tokens or not before the end of
    ///   the prompt: the caches hold that position already, or only the
    ///   prefill of the generation reaches it.
    /// - Throws: `CancellationError` when the task is cancelled between two
    ///   forward passes.
    func prefillingToCheckpoint(
        at boundary: Int, model: any LanguageModel, parameters: GenerateParameters,
        speculation: MTPSpeculation? = nil
    ) throws -> ExecutorPromptCachePlan {
        // A splice can leave the caches at a position other than the reused
        // count, thus the drafter starts where the caches stand.
        let drafterFeed = speculation?.prefillFeed(
            target: model, parameters: parameters, cachePosition: caches.first?.offset ?? 0,
            continuation: drafterState)
        guard
            let feed = try PromptCacheCheckpoint.prefillToCheckpoint(
                at: boundary, of: promptTokens, representing: representedTokens,
                from: reusedTokenCount, after: uncommittedTokens, into: caches, state: state,
                model: model, prefill: parameters.prefill, drafterFeed: drafterFeed)
        else {
            return self
        }
        var plan = ExecutorPromptCachePlan(
            caches: caches,
            input: Self.narrowed(input, to: Array(promptTokens[boundary...])),
            reusedTokenCount: reusedTokenCount,
            promptTokens: promptTokens,
            representedTokens: representedTokens,
            state: feed.state,
            decision: decision)
        plan.prefilledTokenCount = uncommittedTokens.count + boundary - reusedTokenCount
        plan.uncommittedTokens = uncommittedTokens
        plan.checkpoint = feed.checkpoint
        // The fed tokens moved the caches past the drafter state this plan
        // carried, thus only the state the feed left can be resumed.
        plan.drafterState = feed.drafterContinuation
        return plan
    }

    /// A checkpoint of the caches at the end of the prompt, which the
    /// generation reaches when its prefill ends.
    ///
    /// - Parameters:
    ///   - state: the model state the prefill left.
    ///   - drafterState: the MTP drafter continuation at the end of the
    ///     prompt, or nil. The checkpoint keeps a compacted copy of it.
    /// - Returns: the checkpoint, or nil when every cache can rewind or a
    ///   cache does not stand at the end of the prompt.
    func checkpointAtPromptEnd(
        state: LMOutput.State?, drafterState: MTPDrafterContinuation? = nil
    ) -> PromptCacheCheckpoint? {
        PromptCacheCheckpoint(
            caches: caches, tokens: representedTokens, state: state,
            drafterContinuation: drafterState)
    }

    /// The rank of a token array a processor batched: one row for each
    /// sequence, and the tokens of that sequence along the other axis.
    private static let batchedTokenRank = 2

    /// The prompt tokens of `input`, or nil when `input` carries content a
    /// token ledger cannot describe.
    ///
    /// A VLM processor batches a text-only prompt to one row and masks every
    /// token present. That input carries nothing a ledger cannot describe, thus
    /// it passes. Media, a batch of more than one row and a mask that hides a
    /// token each carry more than the tokens say, thus they do not.
    private static func ledgerTokens(of input: LMInput) -> [Int]? {
        guard input.image == nil, input.video == nil, input.audio == nil,
            holdsOneSequence(input.text.tokens),
            marksEveryTokenPresent(input.text.mask, of: input.text.tokens)
        else {
            return nil
        }
        return input.text.tokens.asArray(Int.self)
    }

    /// Whether `tokens` holds one sequence: a plain vector, or a batch of one
    /// row.
    private static func holdsOneSequence(_ tokens: MLXArray) -> Bool {
        tokens.ndim == 1 || (tokens.ndim == batchedTokenRank && tokens.dim(0) == 1)
    }

    /// Whether `mask` marks every token of `tokens` present. No mask hides
    /// nothing; a mask of another shape, or one that holds a zero, hides
    /// something the ledger cannot name.
    private static func marksEveryTokenPresent(_ mask: MLXArray?, of tokens: MLXArray) -> Bool {
        guard let mask else { return true }
        return mask.shape == tokens.shape && mask.asArray(Int32.self).allSatisfy { $0 != 0 }
    }

    /// `input` narrowed to `tokens`, in the rank and the mask presence of
    /// `input`, thus the model sees the shape its processor makes.
    private static func narrowed(_ input: LMInput, to tokens: [Int]) -> LMInput {
        var narrowedTokens = MLXArray(tokens)
        if input.text.tokens.ndim == batchedTokenRank {
            narrowedTokens = narrowedTokens.expandedDimensions(axis: 0)
        }
        let narrowedMask = input.text.mask.map { ones(like: narrowedTokens).asType($0.dtype) }
        return LMInput(text: .init(tokens: narrowedTokens, mask: narrowedMask))
    }

    /// The cache this finished pass leaves for the next turn of its session.
    ///
    /// Generation feeds the tokens the model sampled into the caches, thus the
    /// caches hold what this pass represented before generation AND those
    /// generated tokens. The ledger names both, which is the ledger
    /// ``ChatSession`` keeps. A rewind back to the render alone is not available
    /// to every model -- a rotating cache past its sliding window drops the keys
    /// a rewind needs, and a recurrent cache never rewinds -- and this ledger
    /// asks for none.
    ///
    /// The next turn reconciles the two. When its render extends this ledger,
    /// `ExtendCachedPrefixRule` feeds the tail alone. When the render rewrites
    /// the turn the model wrote, a protocol rule splices past the commit that
    /// closed that turn, with the render recorded here as its proof. Otherwise
    /// `RewindToCommonPrefixRule` takes over and answers what the caches allow.
    ///
    /// - Parameters:
    ///   - generatedTokens: every token this pass generated, in order.
    ///     `TokenIterator` feeds a token before it answers it, thus each token
    ///     of this list stands in the caches.
    ///   - state: the model state the prefill of this pass left, which the
    ///     next turn seeds its iterator with, or nil for a model that carries
    ///     none.
    ///   - drafterState: the MTP drafter continuation the generation of this
    ///     pass handed over when it ended, or nil.
    /// - Returns: the entry to check in, or nil when the caches did not land on
    ///   a position this ledger can name and the session must start cold.
    ///   ``commitOutcome(generatedTokens:state:drafterState:)`` names the
    ///   reason.
    func committed(
        generatedTokens: [Int], state: LMOutput.State? = nil,
        drafterState: MTPDrafterContinuation? = nil
    ) -> ExecutorPromptCacheEntry? {
        commitOutcome(generatedTokens: generatedTokens, state: state, drafterState: drafterState)
            .entry
    }

    /// The cache this finished pass leaves for the next turn, or the reason it
    /// leaves none.
    ///
    /// This is ``committed(generatedTokens:state:drafterState:)`` with the
    /// reason kept, thus the log line of the pass can name it.
    ///
    /// The entry keeps `drafterState` only when it stands at the end of the
    /// ledger, the stop token included when the generation fed it: a drafter
    /// state of any other position cannot be resumed there.
    ///
    /// - Parameters:
    ///   - generatedTokens: every token this pass generated, in order.
    ///   - state: the model state the prefill of this pass left, or nil.
    ///   - drafterState: the MTP drafter continuation the generation of this
    ///     pass handed over when it ended, or nil.
    /// - Returns: the entry to check in, or the refusal.
    func commitOutcome(
        generatedTokens: [Int], state: LMOutput.State? = nil,
        drafterState: MTPDrafterContinuation? = nil
    ) -> ExecutorPromptCacheCommitOutcome {
        guard let position = caches.first?.offset else { return .refused(.noCaches) }
        let positions = caches.map(\.offset)
        guard positions.allSatisfy({ $0 == position }) else {
            return .refused(.positionsDisagree(positions))
        }
        let ledgerLength = representedTokens.count
        guard position >= ledgerLength else {
            return .refused(.behindTheLedger(position: position, ledgerLength: ledgerLength))
        }
        let committedGeneratedTokenCount = position - ledgerLength
        guard committedGeneratedTokenCount <= generatedTokens.count else {
            return .refused(
                .pastTheGeneration(
                    position: position, ledgerLength: ledgerLength,
                    generatedTokenCount: generatedTokens.count))
        }
        let ledger = representedTokens + generatedTokens.prefix(committedGeneratedTokenCount)
        return .checkedIn(
            ExecutorPromptCacheEntry(
                caches: caches,
                tokens: ledger,
                renderTokens: promptTokens,
                uncommittedTokens: Array(generatedTokens.dropFirst(committedGeneratedTokenCount)),
                state: state,
                checkpoint: checkpoint,
                drafterState: drafterState.flatMap {
                    $0.isAligned(withMainTokenCount: ledger.count, drafterID: $0.drafterID)
                        ? $0 : nil
                }))
    }
}

// MARK: - The rule that decided a pass

/// Where a render parts from the ledger it did not extend: the first index
/// where the two differ, and a short window of tokens on each side of it.
struct ExecutorPromptCacheDivergence: Equatable {

    /// How many tokens each side keeps past the seam. Enough to read the seam
    /// in a log line, and no more.
    static let reportedTokenCount = 12

    /// The index of the first token where the render and the ledger differ.
    /// This is the length of the prefix the two share, thus it is the length
    /// of the shorter side when that side is a prefix of the other.
    let index: Int

    /// The render's tokens from `index`, at most ``reportedTokenCount`` of
    /// them. Empty when the render ends at `index`.
    let renderTokens: [Int]

    /// The ledger's tokens from `index`, at most ``reportedTokenCount`` of
    /// them. Empty when the ledger ends at `index`.
    let ledgerTokens: [Int]

    /// Creates a seam at `index` with the tokens each side holds past it.
    ///
    /// - Parameters:
    ///   - index: the first index where the two sides differ.
    ///   - renderTokens: the render's tokens from `index`.
    ///   - ledgerTokens: the ledger's tokens from `index`.
    init(index: Int, renderTokens: [Int], ledgerTokens: [Int]) {
        self.index = index
        self.renderTokens = renderTokens
        self.ledgerTokens = ledgerTokens
    }

    /// Finds the seam between `render` and `ledger`.
    ///
    /// - Parameters:
    ///   - render: the whole rendered prompt.
    ///   - ledger: the tokens the carried caches held.
    init(render: [Int], ledger: [Int]) {
        let index = commonPrefixLength(of: render, and: ledger)
        self.init(
            index: index,
            renderTokens: Self.window(of: render, from: index),
            ledgerTokens: Self.window(of: ledger, from: index))
    }

    /// At most ``reportedTokenCount`` tokens of `tokens` from `index`.
    private static func window(of tokens: [Int], from index: Int) -> [Int] {
        Array(tokens.dropFirst(index).prefix(reportedTokenCount))
    }
}

/// The rule that decided what one generation pass does with the cache its
/// session carries.
///
/// The log line of the pass names it, thus a slow round of an agent run can
/// be attributed: a cold prefill, a rebuild the caches could not avoid, or a
/// long generation on a warm cache.
enum ExecutorPromptCacheDecision: Equatable {

    /// The session carried no cache, or a cache that held nothing, thus the
    /// whole prompt is fed into fresh caches.
    case cold

    /// The render extends the ledger whole, thus the tail alone is fed.
    case extend

    /// A rule of the response protocol kept the tokens the model wrote and fed
    /// the render past the turn the model committed.
    case splice

    /// The caches rewound to the seam, and the render past the seam is fed.
    case rewind(ExecutorPromptCacheDivergence)

    /// The render parts from the ledger at the seam and the recurrent caches
    /// cannot rewind to it, but the render agrees with the checkpoint the
    /// entry carries. The caches went back to that checkpoint, and the render
    /// past it is fed.
    case restore(ExecutorPromptCacheDivergence)

    /// The render parts from the ledger at the seam and the caches cannot
    /// rewind to it, thus the whole prompt is fed into fresh caches.
    case rebuild(ExecutorPromptCacheDivergence)

    /// The name the log line gives the rule.
    var name: String {
        switch self {
        case .cold: "cold"
        case .extend: "extend"
        case .splice: "splice"
        case .rewind: "rewind"
        case .restore: "restore"
        case .rebuild: "rebuild"
        }
    }

    /// The seam the decision names, or nil when the render extends the ledger
    /// or no ledger was carried.
    var divergence: ExecutorPromptCacheDivergence? {
        switch self {
        case .cold, .extend, .splice: nil
        case .rewind(let divergence), .restore(let divergence), .rebuild(let divergence):
            divergence
        }
    }
}

/// Why a finished pass checked nothing in, thus why the next turn of its
/// session starts cold.
enum ExecutorPromptCacheCommitRefusal: Equatable {

    /// The pass carried no plan: its input carries content a token ledger
    /// cannot describe, or the pass owned its cache.
    case noPlan

    /// The plan carried no cache at all.
    case noCaches

    /// The caches do not agree on one position. The positions stand in cache
    /// order, thus a recurrent layer that never advanced shows as a zero.
    case positionsDisagree([Int])

    /// The caches stand before the end of the tokens the plan represented.
    case behindTheLedger(position: Int, ledgerLength: Int)

    /// The caches stand past the represented tokens plus every token the pass
    /// generated, thus the ledger cannot name what they hold.
    case pastTheGeneration(position: Int, ledgerLength: Int, generatedTokenCount: Int)

    /// The reason as the log line states it.
    var reason: String {
        switch self {
        case .noPlan:
            "the pass carried no plan"
        case .noCaches:
            "the pass carried no cache"
        case .positionsDisagree(let positions):
            "the caches disagree on their position \(positions)"
        case .behindTheLedger(let position, let ledgerLength):
            "the caches stand at \(position), behind the \(ledgerLength)-token ledger"
        case .pastTheGeneration(let position, let ledgerLength, let generatedTokenCount):
            "the caches stand at \(position), past the \(ledgerLength)-token ledger "
                + "and the \(generatedTokenCount) generated tokens"
        }
    }
}

/// What a finished pass leaves for the next turn: an entry to check in, or
/// the reason there is none.
enum ExecutorPromptCacheCommitOutcome {

    /// The entry the next turn of the session reuses.
    case checkedIn(ExecutorPromptCacheEntry)

    /// Nothing is checked in, for this reason.
    case refused(ExecutorPromptCacheCommitRefusal)

    /// The entry to check in, or nil when the pass was refused.
    var entry: ExecutorPromptCacheEntry? {
        switch self {
        case .checkedIn(let entry): entry
        case .refused: nil
        }
    }

    /// The refusal, or nil when an entry was checked in.
    var refusal: ExecutorPromptCacheCommitRefusal? {
        switch self {
        case .checkedIn: nil
        case .refused(let refusal): refusal
        }
    }
}

// MARK: - The log line of one pass

/// Where the cache that a pass plans with came from.
enum ExecutorPromptCacheSource: Equatable {

    /// The pass carries no cache from an earlier turn.
    case none

    /// The cache stayed in memory since the earlier turn.
    case memory

    /// The cache went to disk, and the pass read it back. The value is how long the read took.
    case disk(restoreDuration: Duration)
}

/// Composes the log lines of one generation pass.
///
/// Every function here is pure, thus a unit test reads the exact text that
/// `log show` shows for an agent run.
enum ExecutorPromptCacheReport {

    /// The line that names what the pass does with the carried cache.
    ///
    /// - Parameters:
    ///   - key: the session the cache belongs to, or nil when the request
    ///     names no session.
    ///   - source: where the carried cache came from.
    ///   - plan: the plan of the pass, or nil when the pass carries no cache.
    ///   - decodeTokens: decodes a token window to the text the seam shows.
    /// - Returns: one line, with the source, the restore time for a cache from
    ///   disk, and the seam and its decoded sides when the decision names one.
    static func planLine(
        key: ExecutorPromptCacheKey?, source: ExecutorPromptCacheSource,
        plan: ExecutorPromptCachePlan?, decodeTokens: ([Int]) -> String
    ) -> String {
        let head = "prompt cache plan \(session(key)) \(field(of: source)) "
        guard let plan else {
            return head + "rule=none (the input carries media, a batch or a mask)"
        }
        let rendered = plan.promptTokens.count
        // The tokens of the previous answer a splice feeds ahead of the prompt
        // are named apart, because they are no prompt tokens.
        let uncommitted =
            plan.uncommittedTokens.isEmpty ? "" : " uncommitted=\(plan.uncommittedTokens.count)"
        let counts =
            "rendered=\(rendered) reused=\(plan.reusedTokenCount) "
            + "fed=\(rendered - plan.reusedTokenCount)\(uncommitted) rule=\(plan.decision.name)"
        guard let seam = plan.decision.divergence else { return head + counts }
        return head + counts
            + " divergence=\(seam.index) render=<<<\(decodeTokens(seam.renderTokens))>>> "
            + "ledger=<<<\(decodeTokens(seam.ledgerTokens))>>>"
    }

    /// The line of a guided pass, which owns its cache and carries none.
    ///
    /// - Parameter key: the session of the pass, or nil.
    /// - Returns: one line.
    static func guidedLine(key: ExecutorPromptCacheKey?) -> String {
        "prompt cache plan \(session(key)) "
            + "rule=guided (the guided pass owns its cache and carries none)"
    }

    /// The line that names what the finished pass checked in.
    ///
    /// - Parameters:
    ///   - key: the session of the pass, or nil.
    ///   - outcome: what the pass leaves for the next turn.
    /// - Returns: one line with the ledger length and the position of the
    ///   checkpoint when the entry carries one, or the reason nothing was
    ///   checked in.
    static func commitLine(
        key: ExecutorPromptCacheKey?, outcome: ExecutorPromptCacheCommitOutcome
    ) -> String {
        let head = "prompt cache commit \(session(key)) "
        switch outcome {
        case .checkedIn(let entry):
            let checkpoint = entry.checkpointPosition.map { " checkpoint=\($0)" } ?? ""
            return head + "ledger=\(entry.tokens.count)" + checkpoint
        case .refused(let refusal):
            return head + "checked in nothing: \(refusal.reason)"
        }
    }

    /// The line that names an entry the store evicted for its byte budget.
    ///
    /// - Parameters:
    ///   - key: the session of the evicted entry.
    ///   - byteCount: the bytes the entry held in memory.
    /// - Returns: one line with the session and the bytes.
    static func evictionLine(key: ExecutorPromptCacheKey, byteCount: Int) -> String {
        byteCountLine("evict", key: key, byteCount: byteCount)
    }

    /// The line that names a remote entry the store released for its worker
    /// budget, which frees its caches on the workers.
    ///
    /// - Parameters:
    ///   - key: the session of the released entry.
    ///   - workerByteCount: the bytes the entry held on the workers.
    /// - Returns: one line with the session and the worker bytes.
    static func workerReleaseLine(key: ExecutorPromptCacheKey, workerByteCount: Int) -> String {
        byteCountLine("release", key: key, label: "workerBytes", byteCount: workerByteCount)
    }

    /// The line that names a file the store deleted for its disk budget.
    ///
    /// - Parameters:
    ///   - key: the session of the deleted file.
    ///   - byteCount: the size of the file.
    /// - Returns: one line with the session and the bytes.
    static func diskEvictionLine(key: ExecutorPromptCacheKey, byteCount: Int) -> String {
        byteCountLine("disk evict", key: key, byteCount: byteCount)
    }

    /// The one format of the lines that name what the store removed from a
    /// session and the bytes that left with it.
    ///
    /// - Parameters:
    ///   - verb: what the store did, for example `evict`.
    ///   - key: the session of the removed entry or file.
    ///   - label: the name of the byte count in the line.
    ///   - byteCount: the bytes that left.
    /// - Returns: one line with the verb, the session and the bytes.
    private static func byteCountLine(
        _ verb: String, key: ExecutorPromptCacheKey, label: String = "bytes", byteCount: Int
    ) -> String {
        "prompt cache \(verb) \(session(key)) \(label)=\(byteCount)"
    }

    /// The line that names the end of one write of the spool.
    ///
    /// - Parameters:
    ///   - key: the session of the entry.
    ///   - byteCount: the bytes the entry held in memory.
    ///   - duration: how long the write took.
    ///   - outcome: what the write gave.
    /// - Returns: one line with the session, the bytes, the seconds of the
    ///   write and its result.
    static func spillLine(
        key: ExecutorPromptCacheKey, byteCount: Int, duration: Duration,
        outcome: ExecutorPromptCacheSpillOutcome
    ) -> String {
        byteCountLine("spill", key: key, byteCount: byteCount)
            + " seconds=\(seconds(duration)) result=\(outcome.description)"
    }

    /// The line of a turn whose spilled cache did not read back. The turn
    /// starts cold.
    ///
    /// - Parameters:
    ///   - key: the session of the turn.
    ///   - error: why the read failed.
    /// - Returns: one line with the session and the error.
    static func restoreFailureLine(key: ExecutorPromptCacheKey, error: any Error) -> String {
        "prompt cache restore \(session(key)) failed, the turn starts cold: \(error)"
    }

    /// The line of a turn whose spilled drafter state did not read back. The
    /// main caches restored, thus only the drafter starts cold.
    ///
    /// - Parameters:
    ///   - key: the session of the turn.
    ///   - error: why the read failed.
    /// - Returns: one line with the session and the error.
    static func drafterRestoreFailureLine(key: ExecutorPromptCacheKey, error: any Error) -> String {
        "prompt cache restore \(session(key)) drafter state failed, the drafter starts cold: \(error)"
    }

    /// The line of a spill whose drafter state cannot be written. The file
    /// keeps the main caches, thus the next turn starts only its drafter cold.
    ///
    /// - Parameters:
    ///   - key: the session of the entry.
    ///   - error: why the drafter state cannot be written.
    /// - Returns: one line with the session and the error.
    static func drafterSpillFailureLine(key: ExecutorPromptCacheKey, error: any Error) -> String {
        "prompt cache spill \(session(key)) drafter state failed, the file keeps the main caches: "
            + "\(error)"
    }

    /// The source field of a plan line: `source=none`, `source=memory`, or
    /// `source=disk` and the seconds of the restore.
    private static func field(of source: ExecutorPromptCacheSource) -> String {
        switch source {
        case .none: "source=none"
        case .memory: "source=memory"
        case .disk(let restoreDuration):
            "source=disk restoreSeconds=\(seconds(restoreDuration))"
        }
    }

    /// `duration` in seconds, with three decimals.
    private static func seconds(_ duration: Duration) -> String {
        String(format: "%.3f", duration / .seconds(1))
    }

    /// The model and the session of `key`, or `none` for each when the
    /// request names no session.
    private static func session(_ key: ExecutorPromptCacheKey?) -> String {
        "model=\(key?.modelID.rawValue ?? "none") session=\(key?.sessionID.rawValue ?? "none")"
    }
}

/// The unified log of the executor prompt cache: `info` level, in the
/// `ExecutorPromptCache` category of the `com.apple.FoundationModels-MLX`
/// subsystem.
enum ExecutorPromptCacheLog {

    /// The log every line of the prompt cache goes to.
    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "ExecutorPromptCache")

    /// Writes `line` to the unified log at `info` level. Every field is
    /// public: the line carries token counts, byte counts, a session
    /// identifier and a short decoded seam, and `log show` must show them.
    ///
    /// - Parameter line: the line to write.
    static func info(_ line: String) {
        logger.info("\(line, privacy: .public)")
    }
}

/// Times the work of the prompt cache whose duration a log line reports: a
/// write of the spool and a restore of a slot.
enum ExecutorPromptCacheTiming {

    /// Runs `work` and measures it on a `ContinuousClock`.
    ///
    /// An error of `work` leaves the function with no duration, thus a caller
    /// that reports a failure handles the error outside.
    ///
    /// - Parameter work: the work to time.
    /// - Returns: the value of `work`, and the time it took.
    /// - Throws: the error of `work`.
    static func measure<Value>(
        _ work: () throws -> Value
    ) rethrows -> (value: Value, duration: Duration) {
        let clock = ContinuousClock()
        let start = clock.now
        let value = try work()
        return (value, start.duration(to: clock.now))
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
