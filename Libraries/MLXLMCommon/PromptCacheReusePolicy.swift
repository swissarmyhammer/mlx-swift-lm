// Copyright © 2025 Apple Inc.

import Foundation

/// How a newly rendered prompt should be reconciled with the tokens a live
/// KV cache already represents.
///
/// A decision describes *what* to do; applying it (trimming, rebuilding,
/// prefilling) belongs to whoever owns the caches.
package enum PromptCacheReuseDecision: Equatable {

    /// Feed the whole prompt. The cache holds nothing worth reusing.
    case prefillAll

    /// The cache already represents a usable prefix. Feed only
    /// `promptTokens[suffixStart...]`; afterwards the cache represents
    /// `representedTokens`.
    ///
    /// `representedTokens` is not necessarily the rendered prompt. A response
    /// protocol may keep generated tokens that a cold template render cannot
    /// reproduce, in which case the ledger diverges from the render by design.
    case appendSuffix(suffixStart: Int, representedTokens: [Int])

    /// Feed a suffix into the main cache while discarding an unusable draft
    /// cache. The current generation must use the main model only; a later
    /// turn may rebuild a draft cache from a reproducible prompt.
    case appendSuffixToMain(suffixStart: Int, representedTokens: [Int])

    /// Rewind every cache by `trimCount` (down to `commonPrefixLength`), then
    /// feed `promptTokens[commonPrefixLength...]`.
    case trimToCommonPrefix(commonPrefixLength: Int, trimCount: Int)

    /// The cache cannot be reconciled with this prompt. Discard it, drop any
    /// carried model state, and feed the whole prompt.
    case rebuild

    /// `true` when a non-empty cached prefix is carried into this turn.
    var reusesCachedPrefix: Bool {
        switch self {
        case .appendSuffix, .appendSuffixToMain, .trimToCommonPrefix:
            return true
        case .prefillAll, .rebuild:
            return false
        }
    }
}

/// The prompt-side facts of one turn that affect cache reuse.
package struct PromptCacheTurn: Sendable {
    /// The full rendered prompt for this turn.
    package var promptTokens: [Int]

    /// This turn's messages introduce new media, so the cached text prefix is
    /// no longer a valid prefix of the model's actual input.
    package var carriesNewMedia: Bool = false

    /// The prepared input contains image/video/audio tensors, which the rewind
    /// path cannot account for.
    package var carriesPreparedMedia: Bool = false

    /// The prepared input carries an explicit attention mask; a partial prefill
    /// would apply it against the wrong positions.
    package var carriesAttentionMask: Bool = false

    /// Per-call model state (e.g. M-RoPE deltas) is carried across turns. Such
    /// state is anchored to a prefill and cannot be rewound.
    package var carriesModelState: Bool = false

    /// This turn appends tool results to a transcript whose last assistant
    /// message issued tool calls, i.e. the generation loop is resuming rather
    /// than starting a new exchange.
    ///
    /// Response protocols that keep private per-turn state in the cache use
    /// this to recognize a restart they can splice onto.
    package var isToolResultContinuation: Bool = false

    /// Generated tokens returned to the previous caller but not yet represented
    /// by the main cache. Speculative iterators can leave their final verifier
    /// sample in this state.
    package var previousGenerationUncommittedTokens: [Int] = []

    /// Number of structured assistant tool-call messages represented by the
    /// rendered prompt. A protocol can compare this with raw control-token
    /// occurrences to reject ambiguous boundaries introduced by message text.
    package var structuredToolCallCount: Int?

    /// The session is configured to use a draft model when possible. Protocol
    /// rules use this to avoid resuming from a private main-cache path with a
    /// missing or divergent draft cache.
    package var usesSpeculativeDecoding: Bool = false
}

/// What the caches currently hold.
package struct PromptCacheState: Sendable {
    /// Exact tokens the main cache represents, per the session's ledger. Empty
    /// means the ledger was invalidated and nothing may be spliced onto.
    package var cachedTokens: [Int]

    /// The whole prompt that the last prefill rendered, which is not the same as
    /// ``cachedTokens``: the ledger also holds the tokens the model generated
    /// after that render.
    ///
    /// A protocol rule compares this with the new render to prove that the
    /// template rewrote no already-cached rendered region. Once that holds, the
    /// only region the two can differ in is the one the model generated, and the
    /// cache holds the true version of it. Empty means no render is on record,
    /// thus no rule may splice.
    package var previousRenderTokens: [Int] = []

    /// Authoritative logical position of the main cache — the model-wide
    /// timeline maintained by ``KVCacheStorage``, not a per-entry offset.
    package var processedTokenCount: Int

    /// The main cache's timeline agrees with the ledger length.
    package var mainCacheIsAligned: Bool = false

    /// A live draft cache is available for the next generation.
    package var hasDraftCache: Bool = false

    /// The draft cache's timeline agrees with the ledger length. This remains
    /// `true` when no draft exists so generic main-only cache decisions keep
    /// their existing behavior; protocol rules can inspect `hasDraftCache`
    /// when absence matters.
    package var draftCacheIsAligned: Bool = true

    /// Every cache supports rewinding.
    package var isTrimmable: Bool = false
}

/// One reusability rule.
///
/// Returning `nil` means "this rule does not apply"; the policy then consults
/// the next rule. This is the extension point for response protocols whose
/// on-device token stream is not reproducible by a chat-template render.
package protocol PromptCacheReuseRule: Sendable {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision?
}

/// A decision, and whether a rule of the response protocol made it.
struct PromptCacheReuseVerdict: Equatable {

    /// What to do with the caches.
    let decision: PromptCacheReuseDecision

    /// Whether a rule of the model's response protocol made `decision`.
    /// `false` for a standard rule, and for the terminal rebuild.
    let isProtocolDecision: Bool
}

/// Decides how to reuse a KV cache across turns by consulting an ordered list
/// of rules.
///
/// The policy is free of MLX and session state, so the whole decision table can
/// be unit tested directly and the caller's only job is to *apply* the result.
struct PromptCacheReusePolicy: Sendable {

    /// Rules that apply to every model, in priority order.
    static let standardRules: [any PromptCacheReuseRule] = [
        ExtendCachedPrefixRule(),
        RewindToCommonPrefixRule(),
    ]

    private let protocolRules: [any PromptCacheReuseRule]

    /// - Parameter protocolRules: rules contributed by the model's response
    ///   protocol. They are consulted before the standard rules, because a
    ///   protocol that keeps unrenderable state in the cache must claim the
    ///   turn before generic prefix comparison is attempted on token streams
    ///   that are not comparable.
    init(protocolRules: [any PromptCacheReuseRule] = []) {
        self.protocolRules = protocolRules
    }

    /// Decides what to do with the caches, and names the kind of rule that
    /// decided.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of the turn.
    ///   - cache: what the caches hold.
    /// - Returns: the verdict. The protocol rules answer first; the standard
    ///   rules answer next; a rebuild is the answer when no rule applies.
    func resolve(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseVerdict {
        if let decision = Self.firstDecision(of: protocolRules, turn: turn, cache: cache) {
            return PromptCacheReuseVerdict(decision: decision, isProtocolDecision: true)
        }
        let decision =
            Self.firstDecision(of: Self.standardRules, turn: turn, cache: cache) ?? .rebuild
        return PromptCacheReuseVerdict(decision: decision, isProtocolDecision: false)
    }

    /// Decides what to do with the caches.
    ///
    /// - Parameters:
    ///   - turn: the prompt-side facts of the turn.
    ///   - cache: what the caches hold.
    /// - Returns: the decision of ``resolve(turn:cache:)``.
    func decide(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision {
        resolve(turn: turn, cache: cache).decision
    }

    /// The decision of the first rule of `rules` that applies, or nil when
    /// none applies.
    private static func firstDecision(
        of rules: [any PromptCacheReuseRule], turn: PromptCacheTurn, cache: PromptCacheState
    ) -> PromptCacheReuseDecision? {
        for rule in rules {
            if let decision = rule.reuse(turn: turn, cache: cache) {
                return decision
            }
        }
        return nil
    }
}

/// How many leading tokens `first` and `second` share.
///
/// - Parameters:
///   - first: one token list.
///   - second: the other token list.
/// - Returns: the length of the common prefix.
package func commonPrefixLength(of first: [Int], and second: [Int]) -> Int {
    zip(first, second).prefix { $0 == $1 }.count
}

// MARK: - Standard rules

/// Appends the trailing tokens when the prompt strictly extends the cache.
struct ExtendCachedPrefixRule: PromptCacheReuseRule {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        guard !cache.cachedTokens.isEmpty,
            cache.mainCacheIsAligned,
            cache.draftCacheIsAligned,
            !turn.carriesNewMedia,
            !turn.carriesAttentionMask,
            turn.promptTokens.count > cache.cachedTokens.count,
            turn.promptTokens.starts(with: cache.cachedTokens)
        else {
            return nil
        }

        return .appendSuffix(
            suffixStart: cache.cachedTokens.count, representedTokens: turn.promptTokens)
    }
}

/// Rewinds to the longest common prefix, or rebuilds when that is unsafe.
///
/// This rule always decides, so it is the natural terminal rule.
struct RewindToCommonPrefixRule: PromptCacheReuseRule {
    func reuse(turn: PromptCacheTurn, cache: PromptCacheState) -> PromptCacheReuseDecision? {
        // A cache at position zero has nothing to rewind and nothing to
        // invalidate, so the prompt is simply prefilled into it.
        guard cache.processedTokenCount != 0 else {
            return .prefillAll
        }

        let sharedPrefixLength = commonPrefixLength(of: turn.promptTokens, and: cache.cachedTokens)
        let trimCount = cache.cachedTokens.count - sharedPrefixLength

        let canRewind =
            sharedPrefixLength > 0
            && sharedPrefixLength < turn.promptTokens.count
            && trimCount > 0
            && cache.mainCacheIsAligned
            && cache.draftCacheIsAligned
            && cache.isTrimmable
            && !turn.carriesNewMedia
            && !turn.carriesPreparedMedia
            && !turn.carriesAttentionMask
            && !turn.carriesModelState

        guard canRewind else {
            // The template changed an already-cached portion of the transcript,
            // or this input carries state/media that cannot be rewound safely.
            // Rebuild rather than combining a mismatched prompt with stale
            // model state.
            return .rebuild
        }

        return .trimToCommonPrefix(commonPrefixLength: sharedPrefixLength, trimCount: trimCount)
    }
}

// MARK: - Applying a decision to live caches

/// Rewinds `caches` to `position`, and confirms that every cache landed there.
///
/// A cache answers a trim with the number of positions it really dropped, and a
/// rotating cache past its window drops none. A caller must never splice a
/// prompt onto a cache that did not land, thus this function reports the
/// outcome instead of assuming it.
///
/// - Parameters:
///   - caches: the live caches, one for each layer of the model.
///   - position: the logical position every cache must hold on return.
/// - Returns: `true` when every cache holds exactly `position` positions. A
///   `false` answer leaves the caches unusable, and the caller must build new
///   ones.
package func rewindPromptCache(_ caches: [KVCache], to position: Int) -> Bool {
    guard let first = caches.first, position >= 0 else { return false }
    let trimCount = first.offset - position
    guard trimCount >= 0 else { return false }
    guard trimCount == 0 || trimPromptCache(caches, numTokens: trimCount) == trimCount else {
        return false
    }
    return caches.allSatisfy { $0.offset == position }
}

/// The rule that decided a reuse, as a log names it.
package enum PromptCacheReuseKind: Equatable, Sendable {
    /// The caches hold nothing, and the whole prompt is fed into them.
    case prefill

    /// The prompt extends the ledger whole, and the tail alone is fed.
    case extend

    /// A rule of the response protocol kept the tokens the model wrote and
    /// fed the render past the turn the model committed.
    case splice

    /// The caches rewound to the common prefix of the prompt and the ledger,
    /// and the rest of the prompt is fed.
    case rewind
}

/// What live caches hold of a newly rendered prompt once a decision is applied.
package struct PromptCacheReuse: Equatable, Sendable {
    /// How many leading tokens of the prompt the caches hold, thus the caller
    /// feeds the prompt from this index onward.
    package let suffixStart: Int

    /// The tokens the caches represent once the caller has fed the suffix. This
    /// is the prompt itself on the standard path, and the ledger the model wrote
    /// plus the new tail when a protocol rule spliced past a committed turn.
    package let representedTokens: [Int]

    /// The rule that decided this reuse.
    package let kind: PromptCacheReuseKind

    /// Creates a reuse of `suffixStart` leading tokens that `kind` decided.
    package init(suffixStart: Int, representedTokens: [Int], kind: PromptCacheReuseKind) {
        self.suffixStart = suffixStart
        self.representedTokens = representedTokens
        self.kind = kind
    }
}

/// Reconciles live caches with a newly rendered prompt, and reports how much of
/// that prompt the caches already hold and what they represent afterwards.
///
/// The caller owns `caches` and the ledger `cachedTokens`, which names the exact
/// tokens `caches` represents. This function asks ``PromptCacheReusePolicy`` for
/// a decision, applies a rewind when the decision asks for one, and confirms
/// that the rewind landed.
///
/// The protocol rules come first, the way ``ChatSession`` orders them. A rule
/// serves a ledger that holds generated tokens a template render cannot write
/// again; it needs `previousRenderTokens` to prove that the new render rewrote
/// no cached region. A caller with no rule, or no render on record, gets the
/// plain prefix comparison. ``ChatSession`` keeps the richer application,
/// because it also owns a draft cache, carried model state and prepared media.
///
/// - Parameters:
///   - promptTokens: the whole newly rendered prompt.
///   - cachedTokens: the tokens `caches` represents.
///   - previousRenderTokens: the whole prompt the last prefill rendered, or
///     empty when no render is on record.
///   - caches: the live caches, one for each layer of the model.
///   - protocolRules: the rules of the model's response protocol, consulted
///     before the standard rules.
///   - carriesModelState: whether the caller carries per-call model state
///     with `caches`, such as the M-RoPE anchor of a Qwen VL model. That
///     state is tied to the prefill that made it, thus the caches must not
///     rewind under it: a prompt that rewrites a cached token gets no reuse.
/// - Returns: what the caches hold and represent when this function returns.
///   `nil` when the caches cannot serve this prompt at all, and the caller must
///   build new ones.
package func reconcilePromptCache(
    promptTokens: [Int],
    cachedTokens: [Int],
    previousRenderTokens: [Int] = [],
    caches: [KVCache],
    protocolRules: [any PromptCacheReuseRule] = [],
    carriesModelState: Bool = false
) -> PromptCacheReuse? {
    guard !caches.isEmpty else { return nil }

    let turn = PromptCacheTurn(promptTokens: promptTokens, carriesModelState: carriesModelState)
    let cacheState = PromptCacheState(
        cachedTokens: cachedTokens,
        previousRenderTokens: previousRenderTokens,
        processedTokenCount: caches.first?.offset ?? 0,
        mainCacheIsAligned: caches.allSatisfy { $0.offset == cachedTokens.count },
        isTrimmable: canTrimPromptCache(caches))

    let verdict = PromptCacheReusePolicy(protocolRules: protocolRules).resolve(
        turn: turn, cache: cacheState)
    switch verdict.decision {
    case .prefillAll:
        // The rule reaches this case only at position zero, thus the caches
        // hold nothing and the whole prompt is fed into them.
        return PromptCacheReuse(suffixStart: 0, representedTokens: promptTokens, kind: .prefill)

    case .appendSuffix(let suffixStart, let representedTokens),
        .appendSuffixToMain(let suffixStart, let representedTokens):
        // A protocol rule splices past the turn the model committed; the
        // standard rule extends the ledger. The two can agree on the numbers,
        // thus the verdict names the rule and the numbers do not.
        return PromptCacheReuse(
            suffixStart: suffixStart, representedTokens: representedTokens,
            kind: verdict.isProtocolDecision ? .splice : .extend)

    case .trimToCommonPrefix(let commonPrefixLength, _):
        guard rewindPromptCache(caches, to: commonPrefixLength) else { return nil }
        return PromptCacheReuse(
            suffixStart: commonPrefixLength, representedTokens: promptTokens, kind: .rewind)

    case .rebuild:
        return nil
    }
}

/// Reconciles live caches with a newly rendered prompt through the standard
/// rules alone, and reports how much of that prompt the caches already hold.
///
/// This is ``reconcilePromptCache(promptTokens:cachedTokens:previousRenderTokens:caches:protocolRules:)``
/// for a caller whose ledger holds a render alone, thus the answer is the
/// prefix and nothing more.
///
/// - Parameters:
///   - promptTokens: the whole newly rendered prompt.
///   - cachedTokens: the tokens `caches` represents.
///   - caches: the live caches, one for each layer of the model.
/// - Returns: how many leading tokens of `promptTokens` the caches hold when
///   this function returns, or `nil` when the caches cannot serve this prompt.
package func reusablePromptPrefix(
    promptTokens: [Int], cachedTokens: [Int], caches: [KVCache]
) -> Int? {
    reconcilePromptCache(promptTokens: promptTokens, cachedTokens: cachedTokens, caches: caches)?
        .suffixStart
}
