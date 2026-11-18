---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m7nhvmfp6ycsrmzp4bsg24zq
  text: |-
    Research: freeDiskSpaceBytes read only volumeAvailableCapacityForImportantUsage and had its own walk-up loop (parent == probe), a copy of the walk-up in ExecutorPromptCacheStore.availableCapacity. ^0msp3ny (uncommitted) added ExecutorPromptCacheVolumeCapacity + usableFreeSpace + reader typealias + ExecutorPromptCacheStore.readVolumeCapacity in ExecutorPromptCache.swift.

    Implementation: moved the struct, its reader typealias and the default reader into new Libraries/MLXFoundationModels/ExecutorPromptCacheVolumeCapacity.swift (type name kept, since ^0msp3ny's tests use it). The default reader is now ExecutorPromptCacheVolumeCapacity.read(of:) (was ExecutorPromptCacheStore.readVolumeCapacity, used only as the store's default arg). The walk-up loop is now ExecutorPromptCacheVolumeCapacity.nearestExistingFolder(to:) (store's pathComponents.count > 1 form, which keeps the non-file-URL test behavior). Store and MLXLanguageModel both call it. freeDiskSpaceBytes now delegates to internal freeDiskSpaceBytes(reading:) which returns Int64(usableFreeSpace.bytes), nil when the reader throws or reports neither key.

    TDD: Tests/MLXFoundationModelsTests/FreeDiskSpaceTests.swift (importantUsage 0, available 238e9 -> 238e9). RED observed as compile failure (no freeDiskSpaceBytes(reading:)); behavioral run deferred — GPU reserved by another session, xctest barred. CPU-only scratch script with the moved struct: injected reading -> 238000000000; live reading of ~/Library/Caches on this Mac: importantUsage 0, available 238190305280 -> 238190305280.

    Follow-up worth considering: the shared type still carries the ExecutorPromptCache prefix although MLXLanguageModel now uses it; renaming would touch ^0msp3ny's tests, so not done here.
  timestamp: 2026-11-17T20:06:02.486290+00:00
- actor: claude-code
  id: 01m7nhvr95c16v309enrnmnpkg
  text: |-
    ### implement — changed
    - evidence: new Libraries/MLXFoundationModels/ExecutorPromptCacheVolumeCapacity.swift, Libraries/MLXFoundationModels/MLXLanguageModel+Availability.swift, Libraries/MLXFoundationModels/ExecutorPromptCache.swift (moved-out struct/reader/walk-up), new Tests/MLXFoundationModelsTests/FreeDiskSpaceTests.swift; swift-format clean; `swift build --build-tests` Build complete, 0 errors, 0 repo warnings; xctest deferred — GPU reserved
    - next: /test runs FreeDiskSpaceTests + ExecutorPromptCacheSpoolTests in MLXFoundationModelsTests (acceptance box left unchecked until then), then /review
  timestamp: 2026-11-17T20:06:06.373233+00:00
- actor: claude-code
  id: 01m7nw5g1bctxjnnrs9fxab2z6
  text: |-
    ### review — clean
    - evidence: review file on ExecutorPromptCacheVolumeCapacity.swift (0 findings, 1 refuted), MLXLanguageModel+Availability.swift (0), FreeDiskSpaceTests.swift (0). `swift build --build-tests` exit 0. No tests run (GPU reserved); work is uncommitted.
    - next: commit together with ^0msp3ny; the Acceptance unit-test box is still unchecked by the implementer.
  timestamp: 2026-11-17T23:06:11.371682+00:00
position_column: review
position_ordinal: '8180'
title: MLXLanguageModel.freeDiskSpaceBytes reads only important-usage capacity (0 on this Mac)
---
## Why
Found while fixing ^0msp3ny. `MLXLanguageModel.freeDiskSpaceBytes` (Libraries/MLXFoundationModels/MLXLanguageModel+Availability.swift) queries only `.volumeAvailableCapacityForImportantUsageKey`. Since the 2026-11-17 crash this Mac reports 0 for that key while `volumeAvailableCapacity` reports ~240 GB, so the disk-space pre-flight reports 0 free bytes.

## What
- Fall back to `.volumeAvailableCapacityKey` when important-usage is 0 or nil, as `PreparedCheckpoint.write` and (after ^0msp3ny) `ExecutorPromptCacheStore` do. Consider reusing `ExecutorPromptCacheVolumeCapacity.usableFreeSpace`.
- Unit test first with an injected reading.

## Acceptance
- [x] Unit test: important-usage 0, plain capacity > 0 → positive free bytes.
#prompt-cache