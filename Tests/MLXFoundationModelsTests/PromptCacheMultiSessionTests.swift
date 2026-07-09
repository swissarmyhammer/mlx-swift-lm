// Copyright © 2026 Apple Inc.
//
// Multi-session tests for `PromptCache`: several independent conversations
// (distinct token prefixes -- the shape of a multi-agent environment
// sharing one model) interleave and overlap their turns against one cache.
// Each turn mirrors what `Executor.respond` does per round: `resolve()` the
// full prompt-so-far, "generate" (advance the cache's offset), then
// `store()` the grown token sequence back. The invariants under test:
//
// 1. An interleaved session keeps reusing its OWN KV state turn after turn
//    (same instance identity, only the new turn's tokens fed) -- other
//    sessions' rounds in between never steal or corrupt it.
// 2. Truly concurrent sessions never receive each other's KVCache
//    instances, and each still gets multi-turn reuse.
// 3. Oversubscription (more sessions than `maxSlotsPerModel`) degrades to
//    rebuilds -- never to a wrong-prefix reuse.

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

/// One simulated conversation: a unique prefix no other session shares,
/// plus the growing token history a real transcript would re-tokenize to.
private struct SimulatedSession {
    let id: Int
    /// The full prompt tokens as of the last completed turn.
    var tokens: [Int]
    /// The KVCache instance stored for this session's last turn, or nil
    /// before the first turn completes. Held as the live instance (not just
    /// an `ObjectIdentifier`) so eviction can never deallocate it mid-test
    /// and recycle its address into a false identity match.
    var storedCache: (any KVCache)?

    init(id: Int) {
        self.id = id
        // Distinct first token per session: cross-session LCP is always 0,
        // exactly like different system prompts / agent personas.
        self.tokens = [1000 + id, 1, 2]
    }
}

/// Runs one turn for `session` against `cache` the way the executor does:
/// resolve, "generate" by advancing the offset past the full new prompt,
/// store back the grown sequence. Returns the resolved result (for
/// assertions) and mutates the session's history and stored identity.
private func runTurn(
    _ session: inout SimulatedSession, turn: Int, cache: PromptCache, modelID: String,
    model: any MLXLMCommon.LanguageModel
) async throws -> (cache: [KVCache], tokensToFeed: [Int]) {
    let newTokens = session.tokens + [10 + turn]
    let resolved = await resolveOnce(cache: cache, modelID: modelID, newTokens: newTokens, model: model)

    // "Generate" a one-token response: position the cache as if the fed
    // tokens plus the response had gone through the model, mirroring the
    // state `commitPromptCache` verifies before storing.
    let grownTokens = newTokens + [50 + turn]
    try #require(resolved.cache[0] as? KVCacheSimple).offset = grownTokens.count
    await cache.store(modelID: modelID, tokens: grownTokens, cache: SendableBox(resolved.cache))

    session.tokens = grownTokens
    session.storedCache = resolved.cache[0]
    return resolved
}

/// Tests proving several interleaved or concurrent sessions each keep reusing their own KV state.
@Suite("PromptCache multi-session conversations")
struct PromptCacheMultiSessionTests {

    private let model = PromptCacheProbeModel()

    @Test(
        """
        round-robin interleaved sessions each reuse their OWN slot on every turn: \
        another session's rounds in between never steal or invalidate a \
        conversation's KV state
        """
    )
    func interleavedSessionsEachReuseTheirOwnSlot() async throws {
        let cache = PromptCache()
        let modelID = "multi-session-interleaved-\(UUID().uuidString)"
        let sessionCount = 3  // below maxSlotsPerModel: no eviction in play
        let turnCount = 4

        var sessions = (0 ..< sessionCount).map { SimulatedSession(id: $0) }

        for turn in 0 ..< turnCount {
            for index in 0 ..< sessionCount {
                let previousIdentity = sessions[index].storedCache.map {
                    ObjectIdentifier($0 as AnyObject)
                }
                let previousLength = sessions[index].tokens.count
                let resolved = try await runTurn(
                    &sessions[index], turn: turn, cache: cache, modelID: modelID, model: model)

                if let previousIdentity {
                    #expect(
                        cacheIdentity(resolved: resolved) == previousIdentity,
                        "session \(index) turn \(turn) must reuse its own KV state despite interleaved sessions"
                    )
                    #expect(
                        resolved.tokensToFeed.count == 1,
                        "a continued session feeds only its new turn's tokens, not \(resolved.tokensToFeed.count) of \(previousLength + 1)"
                    )
                }
            }
        }
    }

    @Test(
        """
        concurrent sessions running their multi-turn conversations in parallel \
        never receive another session's KVCache instance, and each still gets \
        multi-turn reuse
        """
    )
    func concurrentSessionsStayIsolatedAndReuse() async throws {
        let cache = PromptCache()
        let modelID = "multi-session-concurrent-\(UUID().uuidString)"
        let sessionCount = PromptCache.defaultMaxSlotsPerModel  // at cap: everyone can keep a slot
        let turnCount = 5
        let models = (0 ..< sessionCount).map { _ in PromptCacheProbeModel() }

        // Each concurrent task runs one full conversation and reports the
        // identities it was handed plus how many turns reused its own state.
        let reports = await withTaskGroup(
            of: (session: Int, identities: [ObjectIdentifier], selfReuses: Int).self
        ) { group in
            for index in 0 ..< sessionCount {
                let sessionModel = models[index]
                group.addTask {
                    var session = SimulatedSession(id: index)
                    var identities: [ObjectIdentifier] = []
                    var selfReuses = 0
                    for turn in 0 ..< turnCount {
                        let previousIdentity = session.storedCache.map {
                            ObjectIdentifier($0 as AnyObject)
                        }
                        guard
                            let resolved = try? await runTurn(
                                &session, turn: turn, cache: cache, modelID: modelID,
                                model: sessionModel)
                        else { continue }
                        let identity = cacheIdentity(resolved: resolved)
                        identities.append(identity)
                        if identity == previousIdentity { selfReuses += 1 }
                        await Task.yield()
                    }
                    return (index, identities, selfReuses)
                }
            }
            var results: [(session: Int, identities: [ObjectIdentifier], selfReuses: Int)] = []
            for await report in group {
                results.append(report)
            }
            return results
        }

        #expect(reports.count == sessionCount)

        // Isolation: no KVCache instance ever appears in two different
        // sessions' conversations. (Within one session repeats are the
        // point -- that's reuse.)
        for a in reports {
            for b in reports where b.session != a.session {
                let shared = Set(a.identities).intersection(Set(b.identities))
                #expect(
                    shared.isEmpty,
                    "sessions \(a.session) and \(b.session) were handed the same KVCache instance(s): \(shared)"
                )
            }
        }

        // Reuse: sessions' token prefixes are disjoint and the pool holds
        // one slot per session, so after its first turn every session must
        // have reused its own state on every subsequent turn.
        for report in reports {
            #expect(
                report.selfReuses == turnCount - 1,
                "session \(report.session) reused its own KV state on \(report.selfReuses) of \(turnCount - 1) continuation turns"
            )
        }
    }

    @Test(
        """
        more interleaved sessions than slots degrades safely: an evicted session \
        rebuilds (feeding its full prompt), and no session is ever handed KV state \
        that mismatches its prompt
        """
    )
    func oversubscribedSessionsDegradeToRebuildNeverWrongReuse() async throws {
        let cache = PromptCache()
        let modelID = "multi-session-oversubscribed-\(UUID().uuidString)"
        // Round-robin over more sessions than slots: by the time a session's
        // next turn arrives, its slot has been LRU-evicted by the sessions
        // that ran in between.
        let sessionCount = PromptCache.defaultMaxSlotsPerModel + 2
        let turnCount = 3

        var sessions = (0 ..< sessionCount).map { SimulatedSession(id: $0) }

        for turn in 0 ..< turnCount {
            for index in 0 ..< sessionCount {
                let previousIdentity = sessions[index].storedCache.map {
                    ObjectIdentifier($0 as AnyObject)
                }
                let expectedPrompt = sessions[index].tokens + [10 + turn]
                let resolved = try await runTurn(
                    &sessions[index], turn: turn, cache: cache, modelID: modelID, model: model)

                // Safety invariant, reuse or not: what the caller feeds on
                // top of the checked-out state must reconstruct exactly the
                // prompt it asked for -- a wrong-prefix reuse would violate
                // this by feeding a suffix that doesn't extend its own prefix.
                if cacheIdentity(resolved: resolved) == previousIdentity {
                    #expect(
                        expectedPrompt.suffix(resolved.tokensToFeed.count)
                            == resolved.tokensToFeed[...],
                        "session \(index) turn \(turn): reused state must only leave a genuine suffix to feed"
                    )
                } else {
                    #expect(
                        resolved.tokensToFeed == expectedPrompt,
                        "session \(index) turn \(turn): without its own slot the session must rebuild from its full prompt"
                    )
                }
            }
        }
    }
}

#endif
