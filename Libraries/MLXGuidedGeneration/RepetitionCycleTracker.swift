// Copyright © 2026 Apple Inc.

/// Detects degenerate short-period repetition cycles in the sampled-token
/// stream and accumulates the cycling token ids for logit suppression.
///
/// Grammar-constrained decode samples greedily, so once a model's context
/// is dominated by a short repeating pattern the argmax is deterministic
/// and the cycle never ends on its own — the live failure signatures this
/// guards against are `}7}7}7…` (period 2), `10101010…` (period 1-2 in
/// token space), and `…1234567890123456789…` (period ~10 when a BPE
/// tokenizer groups digits). Position-based closing bias cannot break such
/// a cycle when every cycling token stays grammar-legal (e.g. inside a
/// JSON string), so this tracker detects the cycle itself and reports the
/// participating token ids for suppression: with the cycle's ids penalized,
/// argmax must pick a different grammar-legal token, and generation can
/// make structural progress again.
///
/// Suppression is cumulative and latched for the run, mirroring
/// ``WhitespaceRunTracker``: a model that has demonstrated a degenerate
/// cycle should never be allowed to re-enter it, and a suppression penalty
/// (finite, unlike the grammar mask's `-inf`) still lets a suppressed id
/// win when it is the *only* grammar-legal choice.
public struct RepetitionCycleTracker {

    // MARK: - Private State

    private let minRunLength: Int
    private let minCycles: Int
    private let maxPeriod: Int
    /// Maximum number of recent sampled ids retained in `window`.
    private let capacity: Int
    /// The most recent sampled token ids, oldest first.
    private var window: [Int] = []

    // MARK: - Public API

    /// Creates a tracker.
    ///
    /// A cycle of period `p` is detected when the last
    /// `max(minRunLength, minCycles * p)` sampled tokens are exactly
    /// periodic with period `p` (for some `p` in `1...maxPeriod`).
    ///
    /// - Parameters:
    ///   - minRunLength: Minimum number of consecutive periodic tokens
    ///     before any cycle is detected, whatever its period. Keeps short
    ///     legitimate repeats from triggering: at the default 16, a
    ///     constant JSON array survives up to 8 identical elements
    ///     (period-2 `7,` runs), and separators/indentation never come
    ///     close. Genuinely degenerate cycles run for hundreds-to-
    ///     thousands of tokens (the live `}7}7…`/`1010…` failures), so
    ///     the extra evidence costs nothing there. Longer legitimate
    ///     constant runs CAN still trip detection — an accepted
    ///     rescue-vs-corruption tradeoff, since the suppression penalty
    ///     is finite and the grammar stays satisfiable either way.
    ///   - minCycles: Minimum number of full cycle repetitions required,
    ///     so longer periods need proportionally longer evidence.
    ///   - maxPeriod: Longest cycle period (in tokens) to look for.
    public init(minRunLength: Int = 16, minCycles: Int = 3, maxPeriod: Int = 12) {
        self.minRunLength = minRunLength
        self.minCycles = minCycles
        self.maxPeriod = maxPeriod
        self.capacity = Swift.max(minRunLength, minCycles * maxPeriod)
    }

    /// Token ids that have participated in a detected cycle. Cumulative
    /// across detections and latched for the remainder of the run.
    public private(set) var suppressedTokenIDs: Set<Int> = []

    /// Whether any cycle has been detected so far (latch behavior).
    public var isActive: Bool { !suppressedTokenIDs.isEmpty }

    /// Records a sampled token and checks for a newly completed cycle.
    ///
    /// On detection, the cycle's distinct token ids are added to
    /// ``suppressedTokenIDs`` and the run window is cleared (the caller's
    /// suppression makes an immediate identical re-run impossible, and a
    /// *different* subsequent cycle should be detected on its own fresh
    /// evidence).
    ///
    /// - Parameter tokenID: The sampled token id (grammar-forced
    ///   fast-forward tokens should not be recorded — see ``interrupt()``).
    /// - Returns: `true` when this record completed a new cycle detection.
    @discardableResult
    public mutating func record(tokenID: Int) -> Bool {
        window.append(tokenID)
        if window.count > capacity {
            window.removeFirst(window.count - capacity)
        }

        for period in 1 ... maxPeriod {
            let runLength = Swift.max(minRunLength, minCycles * period)
            guard window.count >= runLength else { break }
            let start = window.count - runLength
            var periodic = true
            for i in (start + period) ..< window.count where window[i] != window[i - period] {
                periodic = false
                break
            }
            if periodic {
                suppressedTokenIDs.formUnion(window.suffix(period))
                window.removeAll(keepingCapacity: true)
                return true
            }
        }
        return false
    }

    /// Clears the run window without touching the latched suppression set.
    ///
    /// Call when sampling is interrupted by grammar-forced fast-forward
    /// tokens: the forced splice already broke any in-flight run, and
    /// joining the sampled tokens on either side of it would fabricate a
    /// contiguous run that never happened.
    public mutating func interrupt() {
        window.removeAll(keepingCapacity: true)
    }
}
