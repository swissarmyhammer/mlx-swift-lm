---
assignees:
- claude-code
comments: []
position_column: todo
position_ordinal: '80'
title: 'Cluster MTP: test (and fix) that rank 0''s drafter starts warm on the turn after a guided turn, a template-seeded copy, and a checkpoint restore'
---
## What
foundationmodelsrouter-22 asked: on a clustered model, after a guided turn under the ^9mvg23k design (rank 0 drafts unmasked and the host keeps what the grammar accepts), does the next turn's RemoteMTPDrafter start warm? ^33h7w11 fixed only the local path. Benches run Inkling and Qwen with MTP on the cluster, and SelectionTier makes many guided turns, each a new Router session seeded from a template copy (^jxyd0ja, ^3wjdpa4).

## What the code does today
Read in Libraries/MLXLMCommon/Remote/RemoteDrafterTrack.swift; none of this is tested for these flows:
- Rank 0 keeps a RemoteDrafterTrack for each cache handle.
- A forward on an aligned track extends it, so the drafter stays warm.
- A copy copies the track.
- A trim right after a verify settles the track and keeps it aligned.
- Any other trim makes it cold: `didTrim` outside `.verifying` sets `.cold`, and so does a trim longer than the verify. The next forward then starts from `coldState`.

So a turn that extends the ledger after a guided cluster turn should be warm. The drafter likely goes cold when:
- the next turn rewinds to a divergence point or to an end-of-message re-render (Qwen/DSML/Inkling splices);
- the end-of-generation trim of the last round's rejects isn't right after a verify;
- a restored checkpoint (a worker-side copy taken mid-prefill) carries a track that isn't aligned at its position.

## Do
1. Test it with the in-process cluster worker and the MTP fixtures (Qwen 3.5 MTP, Inkling DSpark). For each of these flows, check whether rank 0's track is aligned (not cold) at the start of the next speculative turn:
   - a speculative guided turn followed by a speculative turn, plain or guided;
   - a template-seeded copy whose prompt diverges after the instructions (checkpoint restore);
   - a plain speculative turn followed by a guided one.

   Expose the track state to tests if needed, without a protocol change if possible.
2. Fix each flow that goes cold. For example:
   - take worker checkpoints where the track is aligned;
   - let a trim into an aligned region rewind the track while the hidden rows are known.

   Output must not change, and no runtime guard.
3. If a fix needs a protocol change, bump the version and file a FoundationModelsCluster card for foundationmodelscluster-4e.

## Tests
- Each flow above reports a warm rank-0 drafter on the next turn.
- Output equals a cold run.
- Acceptance counts are reported.
#cluster