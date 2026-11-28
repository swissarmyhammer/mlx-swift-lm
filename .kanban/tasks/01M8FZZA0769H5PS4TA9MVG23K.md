---
assignees:
- claude-code
depends_on:
- 01M8FZZ0C8R7PYY6JH0ARXS2A5
position_column: todo
position_ordinal: '9380'
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