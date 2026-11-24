---
assignees:
- claude-code
position_column: todo
position_ordinal: '8180'
title: 'Model cache key: discriminate clustered vs local loads'
---
The MLX model cache keys only on model identity, so a clustered load and a local load of the same model ID could collide. Add a load-identity discriminator to the cache key (e.g. `.local` vs `.clustered(planID)`), so they never share an entry. `evict()` must reach the clustered container, whose teardown kills its workers (via a teardown hook the transport provides).

- [ ] Find the cache key and every lookup/insert/evict path.
- [ ] Add the discriminator, defaulting to local, so existing callers are unchanged.
- [ ] Tests: the same model ID loaded locally and clustered (with a fake transport) gives two entries; evicting one leaves the other; evicting the clustered one calls its teardown. #cluster