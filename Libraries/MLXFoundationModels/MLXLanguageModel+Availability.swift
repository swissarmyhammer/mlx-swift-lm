// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import Metal
import MLXLMCommon

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension MLXLanguageModel {

    /// The availability of an `MLXLanguageModel` for inference.
    ///
    /// MLX models depend on three things to serve a request: a Metal-capable
    /// device, the model weights present in the on-disk location supplied at
    /// construction, and no in-flight download already running. ``availability``
    /// rolls all three into a single value you can use to drive UI affordances
    /// ("Tap to download", "Downloading…", "Ready").
    ///
    /// Use ``MLXLanguageModel/preload()`` to trigger a download when the
    /// availability is ``unavailable(_:)`` with reason
    /// ``UnavailableReason/modelNotDownloaded``. To check whether a download
    /// will fit on disk before kicking it off, compare ``freeDiskSpaceBytes``
    /// against a pre-flight size estimate from your model source (e.g. summing
    /// the remote file sizes reported by your `Downloader` / hub client).
    public enum Availability: Sendable, Equatable {
        /// Weights are downloaded; the model can serve a request.
        ///
        /// Inference may still be slow on the first request after process
        /// launch while Metal shaders are JIT-compiled. Use
        /// ``MLXLanguageModel/Executor/prewarm(model:transcript:)`` (via
        /// `session.prewarm()`) to amortize that cost ahead of time.
        case available

        /// Weights are actively being fetched.
        ///
        /// This corresponds to a genuine in-flight download (an
        /// ``MLXLanguageModel/preload()`` task, or the fetch a `respond()` or
        /// `session.prewarm()` triggers for a not-yet-downloaded model).
        /// A background warmup of an *already-present* model does not report
        /// `.downloading` — the model stays ``available``. Re-check
        /// ``MLXLanguageModel/availability`` after the task completes to
        /// determine the resulting state.
        case downloading

        /// The model cannot serve a request right now.
        case unavailable(UnavailableReason)

        /// The reason an `MLXLanguageModel` cannot currently serve requests.
        public enum UnavailableReason: Sendable, Equatable {
            /// The current device cannot run MLX models because no Metal GPU
            /// is available.
            ///
            /// In practice this only occurs on the iOS Simulator running on
            /// Intel Macs and on a small number of legacy devices. All
            /// supported iOS 27 hardware satisfies this check.
            case deviceNotCapable

            /// Model weights are not present at the configured on-disk
            /// location.
            ///
            /// Call ``MLXLanguageModel/preload()`` to download them.
            case modelNotDownloaded

            /// A previous attempt to download the model failed.
            ///
            /// Calling ``MLXLanguageModel/preload()`` again will retry. This
            /// case clears as soon as a subsequent download succeeds.
            case downloadFailed
        }
    }

    /// A snapshot of the model's current availability.
    ///
    /// This call is fast -- it inspects local on-disk state and the in-process
    /// model cache without contacting any remote service. Network reachability
    /// and remote download size are intentionally not part of the result;
    /// query them explicitly via the relevant helper for your weights source.
    ///
    /// The returned value is a snapshot. Between you reading it and acting on
    /// it, another caller can change the underlying state -- for example, by
    /// starting or completing a download. Treat the value as advisory.
    public var availability: Availability {
        get async {
            // Device capability is a hard precondition. Without Metal,
            // nothing else MLX needs is going to work.
            guard Self.isDeviceCapable else {
                return .unavailable(.deviceNotCapable)
            }

            // A genuine in-flight download takes precedence over disk state --
            // the bytes may not be there yet, or only partially. A background
            // warmup of an already-present model is deliberately excluded here
            // (it is not a user-facing download), so it does not flip an
            // already-`.available` model to `.downloading`.
            if await Self.isDownloadingInCache(modelID: modelID) {
                return .downloading
            }

            // Model weights present on disk -> we can serve a request.
            // (In-memory cached models also satisfy this because the cache
            // never deletes their on-disk source.)
            if modelExistsOnDisk() {
                return .available
            }

            // Nothing on disk and nothing in flight. Distinguish "tried and
            // failed" from "never tried" so callers can show a retry vs. a
            // first-time download affordance.
            if await Self.lastLoadErrorInCache(modelID: modelID) != nil {
                return .unavailable(.downloadFailed)
            }

            return .unavailable(.modelNotDownloaded)
        }
    }

    /// Convenience that returns `true` iff ``availability`` is
    /// ``Availability/available``. Mirrors ``isAvailable`` on
    /// `SystemLanguageModel`.
    public var isAvailable: Bool {
        get async {
            if case .available = await availability { return true }
            return false
        }
    }

    // MARK: - Disk-space pre-flight

    /// Free bytes on the volume hosting this model's configured weights
    /// location, or `nil` if the volume can't be resolved.
    ///
    /// Reads the first extant ancestor of `weightsLocation(configuration.name)`
    /// and reports `URLResourceKey.volumeAvailableCapacityForImportantUsageKey`
    /// when it is positive, else `URLResourceKey.volumeAvailableCapacityKey`:
    /// macOS can report zero for the important-usage key while the volume has
    /// room. Returns `nil` rather than `0` on lookup failure so callers can
    /// distinguish "low" from "unknown". Synchronous because it's just an
    /// `URLResourceValues` lookup -- no I/O.
    public var freeDiskSpaceBytes: Int64? {
        freeDiskSpaceBytes(reading: ExecutorPromptCacheVolumeCapacity.read(of:))
    }

    /// Free bytes on the volume hosting this model's configured weights
    /// location, as `reader` reports them: the
    /// ``ExecutorPromptCacheVolumeCapacity/usableFreeSpace`` of its reading.
    ///
    /// The per-model location won't exist until after a download, thus the
    /// reader reads the first extant ancestor (usually the caches directory,
    /// which the app sandbox always provides).
    ///
    /// - Parameter reader: reads the free space of the volume of a folder that
    ///   exists. Tests give a reader that reports a fixed volume.
    /// - Returns: the free bytes, or `nil` when the reader throws or reports no
    ///   value under either key.
    func freeDiskSpaceBytes(reading reader: ExecutorPromptCacheVolumeCapacityReader) -> Int64? {
        let folder = ExecutorPromptCacheVolumeCapacity.nearestExistingFolder(
            to: weightsLocation(configuration.name))
        guard let freeSpace = try? reader(folder).usableFreeSpace else {
            return nil
        }
        return Int64(freeSpace.bytes)
    }

    // MARK: - Internals

    /// Whether the host has a Metal device available.
    ///
    /// Exposed at module scope because the check is cheap and synchronous,
    /// and consumers occasionally want it independent of the per-model
    /// availability snapshot (e.g. to gate UI that lists candidate models).
    static var isDeviceCapable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }

    /// Whether `config.json` is present at this model's configured on-disk
    /// location.
    ///
    /// `config.json` is the canonical entry point for an MLX-converted
    /// model -- its presence is a strong signal that the snapshot completed.
    /// A partial download that finished `config.json` but not the weight
    /// shards will report `.available` here and fail at load time; that's an
    /// acceptable trade-off versus walking the full file list on every check.
    ///
    /// Resolves through `configuration.name`, not ``modelID``: the cache key
    /// carries the revision (`org/repo@rev`), and a `weightsLocation` closure
    /// expects the path-shaped repository id.
    func modelExistsOnDisk() -> Bool {
        let configPath = weightsLocation(configuration.name).appending(path: "config.json")
        return FileManager.default.fileExists(atPath: configPath.path)
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
