// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// One draft of an MTP round under a grammar: the token the drafter proposed from the logits the
/// grammar shaped, the position it was decided at, and what the matcher did when it took it.
struct GuidedDraft {

    /// The drafted token.
    let token: Int

    /// The position the draft was decided at: the position before it.
    let position: GuidedPosition

    /// What the matcher returned when it took the draft: the tokens it fast-forwarded after it,
    /// and whether the grammar terminated.
    let commit: CommitResult

    /// The whitespace run after the draft.
    let whitespaceAfter: WhitespaceRunTracker

    /// The tokens the matcher accepted for this draft: the draft and its fast-forwarded tokens.
    /// A rollback of the draft rolls back this many.
    var acceptanceCount: Int { 1 + commit.tokens.count }

    /// The tokens the target is fed for this draft: see
    /// ``GuidedGenerationLoop/modelFeed(sampled:fastForward:)``.
    var feed: [Int32] {
        GuidedGenerationLoop.modelFeed(sampled: Int32(token), fastForward: commit.tokens)
    }
}

/// The grammar side of the drafts of one MTP round.
///
/// The drafter runs each draft step through ``processor``: the chain adds the mask and the
/// biases of the step's position to the drafter's logits, then commits the drafted token to the
/// matcher, records how many tokens the matcher accepted for it, and computes the next position.
/// A rejected draft and every draft after it roll back through ``rollBack(draftsFrom:)``.
///
/// The chain ends at a draft that stops the generation, terminates the grammar, forces tokens,
/// or reaches the token budget: a draft after forced tokens needs their target hidden rows,
/// which only a verify gives. The drafter's steps after the end draw from uniform logits, which
/// a drafter confidence floor stops, and the chain records none of them.
///
/// A class, because the drafter holds the processor by value and copies it, while every copy
/// must advance the same matcher records.
final class GuidedDraftChain {

    /// The rules of the generation.
    private let rules: GuidedDecisionRules

    /// The drafts the chain recorded, in order.
    private(set) var drafts: [GuidedDraft] = []

    /// The drafts the matcher still holds: a rollback takes the drafts after this count away.
    private(set) var committedDraftCount = 0

    /// The position of the next draft, or `nil` once the chain ended.
    private(set) var nextPosition: GuidedPosition?

    /// The error the matcher met, which ended the chain.
    private(set) var failure: (any Error)?

    /// Whether the chain records no further draft.
    var hasEnded: Bool { nextPosition == nil }

    /// Starts a chain at `start`, the position of its first draft.
    ///
    /// - Parameters:
    ///   - rules: The rules of the generation.
    ///   - start: The position of the first draft, which the matcher stands at.
    init(rules: GuidedDecisionRules, start: GuidedPosition) {
        self.rules = rules
        self.nextPosition = start
    }

    /// The processor a drafter runs each draft step through.
    var processor: GuidedDraftProcessor { GuidedDraftProcessor(chain: self) }

    /// The logits of a draft step: constrained at the next position, or uniform once the chain
    /// ended.
    ///
    /// - Parameter logits: The drafter's logits of the step, shape `[B, vocabulary]`.
    /// - Returns: The logits the drafter draws from.
    func shape(_ logits: MLXArray) -> MLXArray {
        guard let nextPosition else { return MLX.zeros(like: logits) }
        return rules.constrainedLogits(logits, at: nextPosition)
    }

    /// Records the token a draft step drew: commits it to the matcher and computes the position
    /// after it, or ends the chain.
    ///
    /// - Parameter token: The drafted token, shape `[B]` or `[B, 1]`.
    func record(_ token: MLXArray) {
        guard let position = nextPosition else { return }
        let draftedToken = token.item(Int.self)
        guard !rules.endsGeneration(draftedToken, at: position) else {
            nextPosition = nil
            return
        }
        do {
            let commit = try rules.constraint.commitToken(Int32(draftedToken))
            let whitespace = rules.whitespace(after: draftedToken, from: position.whitespace)
            drafts.append(
                GuidedDraft(
                    token: draftedToken, position: position, commit: commit,
                    whitespaceAfter: whitespace))
            committedDraftCount = drafts.count
            let tokenCount = position.tokenCount + 1 + commit.tokens.count
            guard !commit.isTerminated, commit.tokens.isEmpty, tokenCount < rules.maxTokens else {
                nextPosition = nil
                return
            }
            nextPosition = try rules.position(tokenCount: tokenCount, whitespace: whitespace)
        } catch {
            failure = error
            nextPosition = nil
        }
    }

    /// Rolls the matcher back over the drafts from `index` on, with the tokens each forced.
    ///
    /// - Parameter index: The first draft to roll back. The drafts after it go too.
    /// - Throws: The error of the matcher.
    func rollBack(draftsFrom index: Int) throws {
        precondition(index <= committedDraftCount, "Rolling back a draft the matcher no longer holds")
        let acceptances = drafts[index ..< committedDraftCount].reduce(0) {
            $0 + $1.acceptanceCount
        }
        nextPosition = nil
        committedDraftCount = index
        guard acceptances > 0 else { return }
        try rules.constraint.rollback(Int32(acceptances))
    }
}

/// The logit processor of a guided MTP draft: each step's logits take the grammar mask and the
/// biases of the step's position, and each drafted token goes to the matcher.
///
/// Every copy shares one ``GuidedDraftChain``, so the copies a drafter makes all record into the
/// same round.
struct GuidedDraftProcessor: LogitProcessor {

    /// The chain the drafts go into.
    let chain: GuidedDraftChain

    /// A draft has no prompt of its own.
    mutating func prompt(_ prompt: MLXArray) {}

    /// The step's logits, constrained by the chain.
    func process(logits: MLXArray) -> MLXArray {
        chain.shape(logits)
    }

    /// Records the drafted token in the chain.
    mutating func didSample(token: MLXArray) {
        chain.record(token)
    }
}
