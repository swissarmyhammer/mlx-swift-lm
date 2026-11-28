---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8g2p1cnedt4emk597er8ver
  text: |-
    Research done (no code changed). The coordinator said to keep this small, leave the plain cluster MTP path alone, add no round trips to plain decoding, and stop and report if the change gets large. It gets large, so I stopped.

    Why plain conformance is not enough. The local guided MTP path sends target hidden rows through the host at every step:
    - The prefill feeds the drafter through `MTPDrafterPrefillFeed.chunkHandler` with each chunk's `mtpLastHiddenStatesKey` rows.
    - `GuidedSpeculation.init` needs `prefill.segmentHidden`, which `primedState` uses.
    - `commitDrafter` needs `output.state?[mtpLastHiddenStatesKey]` from each verify, and stands down without it.
    - `lastKeptHidden` is passed into each `draftBlockWithLogits`.

    `ClusterLanguageModel` returns no hidden rows: rank 0 keeps them inside `RemoteDrafterTrack`. Two more gaps:
    - `GuidedTarget.verifiesInPlace` is false for a cluster target. `ClusterLanguageModel` is not a `SpeculativeCacheRewindModel`, and a `RemoteKVCache` has no non-attention leaf.
    - `keepVerified` calls `rewindSpeculativePromptCache`, which calls `RemoteKVCache.trim`. That costs one extra round trip on every rejected round.

    `draftBlockWithLogits` also gets no cache handle, so a remote drafter cannot tell which `RemoteKVCache` to draft on.

    What a one-round-trip-per-round design needs (estimated 800–1500 lines including tests):
    1. Wire, version 6:
       - Add a flag to `RemoteDraftRound`: return the draft logits. Rank 0 drafts unmasked with `draftBlockWithLogits`.
       - `.drafted` carries `draftLogits` for each draft step.
       - Add the flag word to the speculate announce.
       - `RemoteDrafting` must hold a `StatefulMTPDistributionDrafterModel`.
    2. `RemoteMTPDrafter` conforms to both protocols:
       - The state methods are no-ops that report success, because the rank-0 track follows the cache.
       - `draftBlockWithLogits` runs one speculate with draft logits. It puts each step through the host grammar processor and reports drafts only while the masked argmax equals the rank-0 draft. Every reported draft is then inside the verify rank 0 already ran.
       - It keeps the verify (tokens and logits) on the `RemoteKVCache`.
       - It needs a way to reach the cache, for example through `MTPDrafterState.cache` or the session.
    3. `RemoteKVCache` / `ClusterLanguageModel`:
       - Answer a verify whose tokens are a prefix of the kept verify from the kept logits, with no round trip. Otherwise trim the extra positions and forward the rest.
       - Defer rewinds into the next speculate's `rejectedCount`, and flush them as a trim before any other op.
       - Report placeholder hidden rows under the MTP emit flag, so the generic guided code does not stand down. The rows are never read on the host.
       - Make cluster `verifiesInPlace` / `mostVerifiedTokens` true.
    4. Guided code: thread the cluster cases through `GuidedTarget` (prefill segment hidden, verify, keepVerified) without changing the local path.
    5. Tests, and a FoundationModelsCluster card for the v6 wire change.

    This touches the guided core, the wire and announce encoding, the executor, `RemoteKVCache` and the drafter: effectively a new subsystem, which the scope message rules out.

    A smaller option (needs your approval, because it does not make `RemoteMTPDrafter` conform as the card orders): on a cluster target, the guided loop runs the existing `speculate` op unchanged, with no new wire fields and no version bump. It accepts on the host by walking the grammar along `[token] + drafts`: each forced token must equal the verified token, and each decided token must equal the constrained argmax of its row. It folds rejects into the next round's `rejectedCount`, as `ClusterMTPTokenIterator` does, and ends with a trim. That is one round trip per round, no plain-path change, and about 300–400 lines plus tests. The cost is that drafts are unmasked: a step the grammar masks ends acceptance at that row, and the target still decides that token, so the output stays exact.
  timestamp: 2026-11-28T03:20:20.117436+00:00
- actor: claude-code
  id: 01m8g2p47ygh38476t1tjvbsef
  text: |-
    ### implement — stuck
    - evidence: no files changed. The card's design (`RemoteMTPDrafter` conforms to `ResumableMTPDrafterModel` and `StatefulMTPDistributionDrafterModel`) needs host-side hidden rows, a cluster in-place verify, deferred rewinds, and wire v6. That is a large change, which the coordinator's scope rules out. The previous comment gives the full analysis and a smaller option.
    - next: a person decides between (a) the full conformance design (wire v6, about 800–1500 lines) and (b) a cluster guided round over the existing `speculate` op (no wire change, about 300–400 lines, drafts unmasked, output exact). Either one changes the card.
  timestamp: 2026-11-28T03:20:23.038514+00:00
- actor: claude-code
  id: 01m8g2pqmznty2wvdcqv3hxn9n
  text: |-
    ### finish iteration 1 — stuck
    - implement: stuck — no files changed. Making RemoteMTPDrafter conform, as the card says, needs host-side hidden rows, a cluster in-place verify, deferred rewinds and wire v6, about 800–1500 lines (Option A). That is against the scope set on 2026-11-27: keep it small and don't slow plain decoding. Option B uses the existing `speculate` op with no protocol change, about 300–400 lines. Rank 0 drafts unmasked, and the host accepts drafts by walking the grammar along the verified tokens. Output is exact; acceptance may be lower.
    - test/commit/review: not run
    - Blocker: A or B is a spec decision for the requester or the user. Asked foundationmodelsacpagent-a1 and foundationmodelsrouter-22; recommended B.
  timestamp: 2026-11-28T03:20:42.911252+00:00
depends_on:
- 01M8FZZ0C8R7PYY6JH0ARXS2A5
position_column: doing
position_ordinal: '80'
title: 'Clustered models: guided speculation with the remote MTP drafter (RemoteMTPDrafter conforms to ResumableMTPDrafterModel and StatefulMTPDistributionDrafterModel)'
---
## What
A guided call (`respond(to:generating:)` with an @Generable type) on a clustered model never speculates. GuidedSpeculation.drafter(of:for:) (Libraries/MLXGuidedGeneration/GuidedSpeculation.swift:95) needs `ResumableMTPDrafterModel & StatefulMTPDistributionDrafterModel`, and RemoteMTPDrafter (Libraries/MLXLMCommon/Remote/RemoteMTPDrafter.swift, from ^rgzk56j) conforms to neither. On bench cluster4i (Inkling MTP on 2 Macs), 66 of 452 calls were guided SelectionTier calls from FoundationModelsRanker, and all of them decoded plainly. A plain call on the same model speculates: 61% acceptance, 94.9 tok/s. Reported by foundationmodelsrouter-22.

## Fix
- Make RemoteMTPDrafter conform to ResumableMTPDrafterModel and StatefulMTPDistributionDrafterModel over the remote protocol, so the guided loop can resume the drafter and mask the draft distribution with the grammar, as it does locally.
- This likely needs new protocol ops, or new fields on `speculate`: resume the drafter state at a position, and return the draft distribution (or the masked draft) so the grammar on the host can mask each draft step. Keep each round to one round trip where possible. Rank 0 holds the drafter, but the grammar is on the host.
- Bump RemoteModelProtocol.version (now 5) if the wire format changes, and record what FoundationModelsCluster must adopt, as a card for foundationmodelscluster-4e like ^6pq8cee.
- Every rank's cache offset must equal the ledger after a guided speculative turn, so prompt cache reuse keeps working.
- No runtime guard.

## Tests
- A guided (schema) turn on a clustered model with a drafter speculates: draft count > 0, and the greedy output equals the local guided MTP output.
- Acceptance counts are reported.
- A second turn reuses the cache.
- A lost member on a guided speculative turn still throws `memberLost` promptly (^axrb0cw).
Depends on ^arxs2a5 (the fallback reason for the interim). #cluster