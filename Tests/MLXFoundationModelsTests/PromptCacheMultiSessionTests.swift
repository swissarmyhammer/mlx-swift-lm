// Copyright © 2026 Apple Inc.
//
// Multi-session and fork suite rewrite for kanban `ap4mg4y`: full behavioral
// replacement for the deleted `PromptCacheMultiSessionTests.swift` (old
// slot-pool semantics, removed at the `cthbfmw` cutover), re-established over
// chunk storage. This is a distinct follow-up to `t71kdmj`'s already-restored
// resolve/eviction-scope/concurrency suites: this file is specifically about
// the MULTI-SESSION and FORK-shaped scenarios the old suite covered.
//
// The old suite's every guarantee was bounded by `PromptCache
// .defaultMaxSlotsPerModel` (4): only that many conversations could hold a
// checked-out slot at once, an interleaved sibling's turn could evict (and
// therefore silently steal capacity from) another session's slot, and a
// forked continuation sharing a parent's prefix had no dedup mechanism at
// all -- it either found the parent's slot already checked out by someone
// else (a miss forcing a full rebuild) or, if the parent's slot happened to
// be free, checked it out for itself, leaving every OTHER sibling fork
// without it. Chunks are dedup'd, read-only, and never checked out (see
// `PromptCache`'s own doc comment: "no checkout, no steal, and no
// double-checkout hazard by construction"), so every one of these guarantees
// is now STRICTLY STRONGER -- proven below:
//
// 1. Interleaved round-robin sessions --
//    `allInterleavedSessionsReuseChunkAlignedPrefixEveryRoundSimultaneously`:
//    every session (far more than the old 4-slot cap) reuses its own
//    chunk-aligned prefix on EVERY round, simultaneously, with an exact
//    `tokensToFeed` count -- never stolen or evicted by an interleaved
//    sibling's turn in between.
// 2. Fork fan-out (the FoundationModelsRouter fork scenario, see
//    `MLXLanguageModel.prewarm`'s doc comment for that production
//    scenario) --
//    `forkFanOutSharesParentPrefixFromChunksForEveryForkFoundationModelsRouterScenario`:
//    one parent transcript prefix, forked into 7 sibling continuations (plus
//    the parent's own continuation) with distinct tails, ALL resolved
//    interleaved in parallel `Task`s -- every single one (not just
//    whichever happened to run first) is served the shared prefix from
//    chunks, feeding exactly its own tail.
// 3. Concurrent isolation --
//    `concurrentSessionsGetIsolatedAssembledCachesRawBufferMutationNeverLeaks`:
//    two sessions resolving the identical SINGLE-CHUNK shared prefix
//    concurrently get distinct assembled instances with equal content AND
//    genuinely independent underlying buffers (verified past the Swift
//    wrapper, via raw buffer address), and writing directly into one
//    session's buffer in place never appears in the other's already-resolved
//    cache OR in a later fresh resolve against the same stored prefix --
//    proving immutability of the underlying stored chunk, not just distinct
//    Swift object identity. A single-chunk prefix and a raw in-place write
//    are both deliberate: MLX's `concatenate` special-cases exactly one
//    input by returning it unchanged, and `KVCacheSimple.update()` always
//    functionally REPLACES its arrays rather than mutating in place, so
//    neither a multi-chunk prefix nor an `update()`-based mutation could
//    ever exercise (or catch a regression in) `assemble()`'s owned-copy
//    guarantee for the one shape where it matters. This is deliberately at
//    the MULTI-SESSION `resolve()`/`store()` level, not a re-test of
//    `PromptCacheConcurrencyTests`
//    .concurrentResolvesForSamePrefixReturnDistinctInstancesWithEqualContent`
//    (identity/content-equality, no mutation) or `PromptCacheAssembleTests`
//    .concurrentAssembleCallsReturnDistinctInstancesWithEqualContent`/
//    `singleChunkAssemblyAllocatesIndependentBuffer` (call `assemble()`
//    directly, not through a simulated multi-session `resolve()`/`store()`
//    usage pattern); this one adds the still-missing "two concurrent
//    SESSIONS, not just two concurrent assemble() calls" angle.
// 4. Oversubscription under a small byte budget --
//    `oversubscribedSessionsUnderSmallByteBudgetDegradeNeverAssembleWrongPrefix`:
//    many round-robin sessions collectively store far more chunk bytes than
//    a deliberately tiny budget allows, forcing eviction pressure across
//    sessions -- degradation only ever means a LARGER `tokensToFeed` (up to
//    a full rebuild), and the fed tokens always reconstruct exactly the
//    session's own trailing suffix, never a wrong-prefix assembly. This
//    generalizes `PromptCacheByteBudgetTests`'s single/two-lineage LRU tests
//    to genuine multi-session round-robin churn.
//
// Real tensor CONTENT is populated (`PromptCacheTestSupport.makeChunkableCache`)
// because `store()` routes through `sliceChunks`, which requires a genuine
// `KVCacheSimple.state` (not just `offset`) to slice -- this forces a real
// GPU-device eval under plain `swift test`; see `TestBootstrap.swift`'s
// `MetalLibraryTestBootstrap` (kanban 23ff1zx, memory note
// `swiftpm-test-gpu-metallib-limit`) for why the `init()` bootstrap call
// below is required.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// Per-session conversation state a simulated multi-turn round-robin
/// exchange tracks against `PromptCache`, mirroring how `Executor` re-sends
/// each FoundationModels session's growing transcript on every round (there
/// is no session identity at the `LanguageModelExecutor` protocol level --
/// see `PromptCache`'s own doc comment -- so reuse is keyed purely on this
/// tracked token history matching what was previously stored).
private struct SimulatedSession {
    /// Distinguishes this session's own token lineage from every other
    /// session's in the same round-robin run.
    let id: Int
    /// The full prompt tokens as of the last completed turn.
    var tokens: [Int]

    init(id: Int) {
        self.id = id
        // A distinct, large first token per session (cross-session longest-
        // common-prefix is always 0, exactly like distinct system prompts /
        // agent personas that never happen to share a literal token prefix),
        // followed by one more token so that after the FIRST turn's user +
        // generated token are appended (see `runMultiSessionTurn`), the
        // stored round lands on exactly one 4-token chunk boundary --
        // letting the oversubscription scenario learn a genuinely non-zero
        // per-lineage byte cost from real seeded chunks, not an empty store.
        self.tokens = [(id + 1) * 100_000, (id + 1) * 100_000 + 1]
    }
}

/// Runs one turn for `session` against `cache` the way `Executor` does:
/// resolve the growing prompt, "generate" a one-token response by advancing
/// the assembled cache's offset past the fed-plus-generated tokens via
/// `update()`, then store the grown round back. Mutates `session`'s tracked
/// token history for the next call.
///
/// - Parameters:
///   - session: The session whose turn this is; mutated in place with the
///     grown token history for the next call.
///   - turn: This round's index, folded into both the user-turn and
///     generated-token IDs so no two turns (of any session) ever collide.
///   - cache: The `PromptCache` actor every session in the run shares.
///   - modelID: The model identity every session in the run shares.
///   - model: The language model whose `newCache(parameters:)` builds a
///     fresh `[KVCache]` when nothing stored matches.
///   - headDim: Per-token feature width for the simulated generated content.
/// - Returns: This turn's `resolve()` result (for reuse/feed assertions),
///   paired with `newTokens`, the full prompt this turn asked to resolve --
///   `session.tokens` (before this call) plus this turn's new user token.
private func runMultiSessionTurn(
    session: inout SimulatedSession, turn: Int, cache: PromptCache, modelID: String,
    model: any MLXLMCommon.LanguageModel, headDim: Int = 4
) async -> (resolved: (cache: [KVCache], tokensToFeed: [Int]), newTokens: [Int]) {
    let userTurnToken = (session.id + 1) * 10_000_000 + turn * 2
    let newTokens = session.tokens + [userTurnToken]
    let resolved = await resolveOnce(
        cache: cache, modelID: modelID, newTokens: newTokens, model: model)

    // "Generate" a one-token response: advance the assembled cache exactly
    // as a real generation turn would, with distinct, position-derived
    // content (not zeros) so no two sessions'/turns' stored bytes coincide.
    let assembledCache = resolved.cache[0] as! KVCacheSimple
    let newTokenCount = resolved.tokensToFeed.count + 1
    let contentBase = Int32((session.id + 1) * 1_000_000 + turn * 1_000)
    let newKeys = MLXArray(
        contentBase ..< (contentBase + Int32(newTokenCount * headDim)),
        [1, 1, newTokenCount, headDim])
    let newValues = MLXArray(
        (contentBase + 500_000) ..< (contentBase + 500_000 + Int32(newTokenCount * headDim)),
        [1, 1, newTokenCount, headDim])
    _ = assembledCache.update(keys: newKeys, values: newValues)

    let generatedToken = (session.id + 1) * 10_000_000 + turn * 2 + 1
    let grownTokens = newTokens + [generatedToken]
    await cache.store(modelID: modelID, tokens: grownTokens, cache: SendableBox([assembledCache]))

    session.tokens = grownTokens
    return (resolved, newTokens)
}

/// Tests proving the multi-session and fork behavioral guarantees hold under
/// chunk semantics, strictly stronger than the deleted slot-pool suite's
/// (see this file's header comment for the full scenario-to-test mapping).
@Suite("PromptCache multi-session and fork conversations (chunk semantics)")
struct PromptCacheMultiSessionTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    // MARK: - 1. Interleaved round-robin sessions

    @Test(
        """
        every session in an interleaved round-robin reuses ITS OWN chunk-aligned prefix on \
        EVERY continuation turn, simultaneously, feeding an EXACT count of tokens past its \
        last chunk boundary -- strictly stronger than the deleted slot-pool suite, whose \
        guarantee held for at most `PromptCache.defaultMaxSlotsPerModel` (4) sessions at once \
        and let an interleaved sibling's turn evict another session's slot; chunks are read-only \
        and dedup'd, so there is no slot to run out of and nothing for one session's round to \
        steal from another's
        """
    )
    func allInterleavedSessionsReuseChunkAlignedPrefixEveryRoundSimultaneously() async {
        let cache = PromptCache()
        let modelID = "multi-session-interleaved-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        let headDim = 4
        await cache.setChunkSize(chunkSize)

        // Deliberately MORE sessions than the deleted slot pool's default
        // capacity (`defaultMaxSlotsPerModel == 4`) ever let hold state
        // simultaneously.
        let sessionCount = 10
        let turnCount = 5

        var sessions = (0 ..< sessionCount).map { SimulatedSession(id: $0) }

        for turn in 0 ..< turnCount {
            for index in 0 ..< sessionCount {
                let previousStoredTokenCount = sessions[index].tokens.count
                let (resolved, newTokens) = await runMultiSessionTurn(
                    session: &sessions[index], turn: turn, cache: cache, modelID: modelID,
                    model: model, headDim: headDim)

                if turn == 0 {
                    // Nothing stored yet for a brand-new session: a full
                    // rebuild is the only correct behavior.
                    #expect(
                        resolved.tokensToFeed.count == newTokens.count,
                        "session \(index)'s first turn has nothing stored yet and must feed its whole prompt"
                    )
                    continue
                }

                // `store()` only ever persists complete, chunk-aligned
                // chunks (see `sliceChunks`), so the matched prefix on this
                // resolve is exactly the largest chunk-aligned multiple of
                // what was stored last turn -- regardless of how many OTHER
                // sessions' rounds interleaved in between.
                let expectedMatchedTokenCount = (previousStoredTokenCount / chunkSize) * chunkSize
                let expectedFeedCount = newTokens.count - expectedMatchedTokenCount
                #expect(
                    resolved.tokensToFeed.count == expectedFeedCount,
                    """
                    session \(index) turn \(turn): expected exactly \(expectedFeedCount) tokens fed \
                    past its last chunk boundary, got \(resolved.tokensToFeed.count) -- an interleaved \
                    sibling session must never steal or evict this session's chunks
                    """
                )
                #expect(resolved.tokensToFeed == Array(newTokens.suffix(expectedFeedCount)))
            }
        }

        // Every session must still be independently resolvable at the end,
        // each reusing its own (never another's) chunk-aligned prefix.
        for index in 0 ..< sessionCount {
            let extended = sessions[index].tokens + [999_000_000 + index]
            let resolved = await resolveOnce(
                cache: cache, modelID: modelID, newTokens: extended, model: model)
            #expect(
                resolved.tokensToFeed.count < extended.count,
                "session \(index) must still reuse its own chunk-aligned prefix at the end of the interleaved run"
            )
        }
    }

    // MARK: - 2. Fork fan-out (FoundationModelsRouter scenario)

    @Test(
        """
        FORK FAN-OUT -- the FoundationModelsRouter fork scenario at the unit level (see \
        `MLXLanguageModel.prewarm`'s doc comment for the production scenario this mirrors): \
        one parent conversation prefix forked into 7 sibling continuations with distinct tails, \
        PLUS the parent's own continuation, all resolved interleaved in parallel Tasks -- every \
        single one is served the shared prefix entirely from stored chunks (tokensToFeed == \
        exactly its own tail), not just whichever fork happened to resolve first
        """
    )
    func forkFanOutSharesParentPrefixFromChunksForEveryForkFoundationModelsRouterScenario() async {
        let cache = PromptCache()
        let modelID = "multi-session-fork-\(UUID().uuidString)"
        let model: any LanguageModel = PromptCacheProbeModel()
        let chunkSize = 4
        await cache.setChunkSize(chunkSize)

        // A parent transcript prefix, exactly chunk-aligned (a multiple of
        // chunkSize) so the WHOLE prefix is stored and matchable -- mirrors
        // a FoundationModelsRouter parent session whose shared system-
        // prompt/history has already been committed before any fork
        // branches off it.
        let parentTokens = Array(0 ..< (chunkSize * 4))
        await cache.store(
            modelID: modelID, tokens: parentTokens,
            cache: SendableBox([makeChunkableCache(tokenCount: parentTokens.count)]))

        // N >= 6 forked continuations, each sharing `parentTokens` verbatim
        // but diverging with its own distinct tail -- plus the parent's OWN
        // continuation (index 0), all in the same interleaved batch, since
        // the scenario requires "parent and EVERY fork" to resolve from
        // chunks.
        let forkCount = 7
        let tails: [[Int]] = (0 ... forkCount).map { i in
            Array((300_000 + i * 1_000) ..< (300_000 + i * 1_000 + 5))
        }

        let modelBoxes = tails.map { _ in SendableBox(model) }
        let resultBoxes = await withTaskGroup(
            of: (Int, SendableBox<(cache: [KVCache], tokensToFeed: [Int])>).self
        ) { group in
            for (i, tail) in tails.enumerated() {
                let modelBox = modelBoxes[i]
                group.addTask {
                    let forkTokens = parentTokens + tail
                    let resolved = await cache.resolve(
                        modelID: modelID, newTokens: forkTokens, model: modelBox, parameters: nil)
                    return (i, resolved)
                }
            }
            var results: [(Int, SendableBox<(cache: [KVCache], tokensToFeed: [Int])>)] = []
            for await result in group { results.append(result) }
            return results
        }

        #expect(resultBoxes.count == tails.count)

        var identities: Set<ObjectIdentifier> = []
        for (index, box) in resultBoxes {
            let resolved = box.consume()
            #expect(
                resolved.tokensToFeed == tails[index],
                """
                fork \(index) must feed exactly its own tail -- the shared parent prefix must be \
                served from chunks for EVERY fork, not just whichever resolved first
                """
            )
            #expect(resolved.cache[0].offset == parentTokens.count)
            identities.insert(cacheIdentity(resolved: resolved))
        }
        #expect(
            identities.count == tails.count,
            "every fork (and the parent's own continuation) must receive its own distinct assembled cache instance"
        )
    }

    // MARK: - 3. Concurrent isolation

    /// Reaches PAST the Swift `MLXArray` wrapper to the actual underlying
    /// C++ buffer address, exactly mirroring `PromptCacheAssembleTests`'s
    /// own helper of the same shape. `ObjectIdentifier`/`.==` comparison
    /// alone cannot distinguish "genuinely fresh buffer" from "same C++
    /// buffer, new Swift wrapper" -- and neither can `KVCacheSimple
    /// .update()`: its reset path always CONSTRUCTS a brand-new
    /// concatenated array and reassigns `self.keys`/`self.values` to it (a
    /// functional replace, never an in-place mutation), so calling
    /// `update()` on one session's cache could never, by itself, prove the
    /// underlying buffer wasn't aliased with another session's -- it would
    /// "pass" this test even if `assemble()` handed back a stored chunk's
    /// own buffer by reference. Reading through `asData(access: .noCopy)`
    /// and writing through the raw pointer closes that gap.
    ///
    /// - Parameter array: The array whose underlying buffer address to read.
    /// - Returns: The raw base address of `array`'s backing C++ buffer, or
    ///   `nil` if the buffer is empty.
    private func rawBufferAddress(of array: MLXArray) -> UInt? {
        array.asData(access: .noCopy).data.withUnsafeBytes { UInt(bitPattern: $0.baseAddress) }
    }

    /// Writes directly into `array`'s underlying buffer, in place, at its
    /// first raw byte -- bypasses `KVCacheSimple.update()`'s functional
    /// replace path entirely, so any OTHER array that genuinely shares this
    /// same underlying buffer would observe the change.
    ///
    /// - Parameter array: The array to mutate in place.
    private func mutateFirstElementInPlace(of array: MLXArray) {
        array.asData(access: .noCopy).data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            UnsafeMutableRawPointer(mutating: raw.baseAddress!)
                .storeBytes(of: Int32(-1), as: Int32.self)
        }
    }

    @Test(
        """
        CONCURRENT ISOLATION at the multi-session level: two sessions resolving the SAME shared \
        SINGLE-CHUNK prefix concurrently receive distinct assembled cache instances with equal \
        content backed by genuinely INDEPENDENT underlying buffers -- writing directly into one \
        session's buffer in place never appears in the OTHER session's already-resolved cache, \
        nor in a later fresh resolve against the same stored prefix. A single-chunk shared \
        prefix is deliberate: MLX's `concatenate` special-cases exactly one input by returning \
        it unchanged (see `PromptCacheAssembleTests.singleChunkAssemblyAllocatesIndependentBuffer`'s \
        doc comment), so this is the one shape where a missing owned-copy in assemble() could \
        hand back the chunk store's own buffer by reference -- a hazard `update()`-based \
        mutation alone could never catch, since `KVCacheSimple.update()` always functionally \
        replaces its arrays rather than mutating in place
        """
    )
    func concurrentSessionsGetIsolatedAssembledCachesRawBufferMutationNeverLeaks() async {
        let cache = PromptCache()
        let modelID = "multi-session-isolation-\(UUID().uuidString)"
        let model: any LanguageModel = PromptCacheProbeModel()
        let chunkSize = 4
        let headDim = 4
        await cache.setChunkSize(chunkSize)

        // Exactly ONE chunk (see this test's doc comment for why the
        // single-input concatenation special case matters here).
        let sharedPrefix = Array(0 ..< chunkSize)
        await cache.store(
            modelID: modelID, tokens: sharedPrefix,
            cache: SendableBox(
                [makeChunkableCache(tokenCount: sharedPrefix.count, headDim: headDim)]))

        // Two independent sessions continuing the SAME shared prefix with
        // distinct tails, resolved concurrently -- mirrors two live
        // conversations that happen to share a system-prompt prefix (e.g.
        // two forks, or two independent users with an identical system
        // prompt).
        let sessionATokens = sharedPrefix + Array(400_000 ..< 400_005)
        let sessionBTokens = sharedPrefix + Array(500_000 ..< 500_005)
        let modelBoxA = SendableBox(model)
        let modelBoxB = SendableBox(model)

        let resultBoxes = await withTaskGroup(
            of: (String, SendableBox<(cache: [KVCache], tokensToFeed: [Int])>).self
        ) { group in
            group.addTask {
                let resolved = await cache.resolve(
                    modelID: modelID, newTokens: sessionATokens, model: modelBoxA, parameters: nil)
                return ("A", resolved)
            }
            group.addTask {
                let resolved = await cache.resolve(
                    modelID: modelID, newTokens: sessionBTokens, model: modelBoxB, parameters: nil)
                return ("B", resolved)
            }
            var results: [String: SendableBox<(cache: [KVCache], tokensToFeed: [Int])>] = [:]
            for await (label, box) in group { results[label] = box }
            return results
        }

        let resolvedA = resultBoxes["A"]!.consume()
        let resolvedB = resultBoxes["B"]!.consume()

        #expect(cacheIdentity(resolved: resolvedA) != cacheIdentity(resolved: resolvedB))
        let simpleA = resolvedA.cache[0] as! KVCacheSimple
        let simpleB = resolvedB.cache[0] as! KVCacheSimple
        #expect((simpleA.state[0] .== simpleB.state[0]).all().item())
        #expect((simpleA.state[1] .== simpleB.state[1]).all().item())

        // The buffers themselves must be genuinely independent, not just
        // distinct Swift wrappers around the SAME underlying C++ memory --
        // `.==`/`ObjectIdentifier` equality can't distinguish that (see
        // this test's doc comment).
        #expect(rawBufferAddress(of: simpleA.state[0]) != rawBufferAddress(of: simpleB.state[0]))
        #expect(rawBufferAddress(of: simpleA.state[1]) != rawBufferAddress(of: simpleB.state[1]))

        // Snapshot session B's ORIGINAL content before session A mutates its own buffer.
        let originalBKeys = simpleB.state[0]
        let originalBValues = simpleB.state[1]

        // Session A mutates its OWN buffer directly, in place -- bypassing
        // update()'s functional replace entirely (see
        // mutateFirstElementInPlace's doc comment for why that matters).
        mutateFirstElementInPlace(of: simpleA.state[0])
        mutateFirstElementInPlace(of: simpleA.state[1])

        // Session B's ALREADY-RESOLVED cache must be untouched by A's
        // mutation -- proves the two assembled instances, though built from
        // the SAME underlying stored chunk, share no storage.
        #expect((simpleB.state[0] .== originalBKeys).all().item())
        #expect((simpleB.state[1] .== originalBValues).all().item())

        // And the underlying STORE itself is untouched: a fresh resolve for
        // a third session against the same shared prefix still sees the
        // original, un-mutated content -- proving immutability of the
        // stored chunk itself, not merely instance-level isolation between
        // A and B.
        let resolvedFresh = await resolveOnce(
            cache: cache, modelID: modelID, newTokens: sharedPrefix + [600_000], model: model)
        let simpleFresh = resolvedFresh.cache[0] as! KVCacheSimple
        #expect(simpleFresh.offset == sharedPrefix.count)
        #expect((simpleFresh.state[0] .== originalBKeys).all().item())
        #expect((simpleFresh.state[1] .== originalBValues).all().item())
    }

    // MARK: - 4. Oversubscription under a small byte budget

    @Test(
        """
        OVERSUBSCRIPTION under a small byte budget: many round-robin sessions collectively grow \
        to store far more chunk bytes than a deliberately tight budget allows, forcing real \
        eviction pressure across sessions -- correctness only ever DEGRADES to a larger \
        tokensToFeed (up to a full rebuild), and the fed tokens always reconstruct EXACTLY the \
        session's own trailing suffix -- never a wrong-prefix assembly. Verifies degradation is \
        genuinely budget-driven (some turn is evicted into a full rebuild) and that the budget \
        doesn't starve reuse entirely (some turn still gets genuine partial reuse) -- a \
        tautological suffix check alone can't distinguish real degradation from a bug that \
        always misses or a budget so generous nothing is ever evicted
        """
    )
    func oversubscribedSessionsUnderSmallByteBudgetDegradeNeverAssembleWrongPrefix() async {
        let cache = PromptCache()
        let modelID = "multi-session-oversubscribed-\(UUID().uuidString)"
        let model = PromptCacheProbeModel()
        let chunkSize = 4
        let headDim = 4
        await cache.setChunkSize(chunkSize)

        let sessionCount = 8
        let turnCount = 4
        var sessions = (0 ..< sessionCount).map { SimulatedSession(id: $0) }

        // Seed one round for every session first, to learn a real per-
        // lineage byte cost before clamping the budget -- sizing the clamp
        // relative to what these sessions actually cost, rather than a
        // magic constant that could drift with fixture changes.
        for index in 0 ..< sessionCount {
            _ = await runMultiSessionTurn(
                session: &sessions[index], turn: 0, cache: cache, modelID: modelID, model: model,
                headDim: headDim)
        }
        let totalAfterSeeding = await cache.totalStoredByteCount()
        let perLineageApproxBytes = totalAfterSeeding / sessionCount

        // Budget sized to exactly fit every session's CURRENT (single-chunk)
        // lineage -- no headroom for growth. As sessions' turns accumulate
        // past `chunkSize` tokens each, their lineages grow to a SECOND
        // chunk, pushing the store's true demand up to roughly double this
        // budget -- genuine oversubscription that emerges from real growth,
        // not an artificially starved budget that would evict everything
        // immediately and evict every following turn identically.
        let budget = perLineageApproxBytes * sessionCount
        await cache.setByteBudget(budget)

        var observedGenuinePartialReuse = false
        var observedFullRebuildUnderPressure = false

        for turn in 1 ..< turnCount {
            for index in 0 ..< sessionCount {
                let (resolved, newTokens) = await runMultiSessionTurn(
                    session: &sessions[index], turn: turn, cache: cache, modelID: modelID,
                    model: model, headDim: headDim)

                if resolved.tokensToFeed.count < newTokens.count {
                    observedGenuinePartialReuse = true
                } else {
                    observedFullRebuildUnderPressure = true
                }

                // Suffix-reconstruction invariant: regardless of HOW MUCH
                // was reused (fully evicted -> full rebuild, partially
                // evicted -> a partial reuse), the fed tokens must always be
                // EXACTLY the trailing suffix of this session's own true
                // growing prompt -- never a mismatched/wrong prefix, and
                // never more than the whole prompt or fewer than 1 token.
                #expect(
                    resolved.tokensToFeed.count >= 1 && resolved.tokensToFeed.count <= newTokens.count,
                    "session \(index) turn \(turn): tokensToFeed must be between 1 and the full prompt length"
                )
                #expect(
                    resolved.tokensToFeed == Array(newTokens.suffix(resolved.tokensToFeed.count)),
                    """
                    session \(index) turn \(turn): fed tokens must reconstruct exactly the session's \
                    own trailing suffix, never a wrong prefix, even under byte-budget eviction pressure
                    """
                )

                // The byte budget invariant itself must keep holding
                // throughout, confirming the degradation above is genuinely
                // budget-driven eviction, not a coincidence.
                #expect(await cache.totalStoredByteCount() <= budget)
            }
        }

        #expect(
            observedGenuinePartialReuse,
            """
            the budget must still allow at least one genuine partial reuse somewhere in this run \
            -- otherwise this test would pass just as well against a total-miss regression, never \
            actually proving reuse survives under pressure
            """
        )
        #expect(
            observedFullRebuildUnderPressure,
            """
            the budget must force at least one genuine full rebuild somewhere in this run -- \
            otherwise this test would pass just as well against a budget so generous nothing was \
            ever actually evicted, never proving real oversubscription pressure occurred
            """
        )
    }
}

#endif
