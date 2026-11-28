// Copyright © 2026 Apple Inc.

import Foundation

/// The policy of a ``RemoteKVCacheQuantization`` when only some layers of a rank can quantize:
/// the ``KVCacheConfiguration/CompatibilityPolicy`` of the request.
public enum RemoteKVCacheCompatibility: Int32, Codable, Sendable, CaseIterable {

    /// Quantize the layers that can, and keep the others: ``KVCacheConfiguration/CompatibilityPolicy/allowPartial``.
    case allowPartial = 0

    /// Fail unless one attention layer can quantize:
    /// ``KVCacheConfiguration/CompatibilityPolicy/requireAtLeastOneLayer``.
    case requireAtLeastOneLayer = 1

    /// Fail unless every attention layer can quantize:
    /// ``KVCacheConfiguration/CompatibilityPolicy/requireAllLayers``.
    case requireAllLayers = 2

    /// The policy of `policy`.
    ///
    /// - Parameter policy: The policy of a KV-cache configuration.
    init(_ policy: KVCacheConfiguration.CompatibilityPolicy) {
        switch policy {
        case .allowPartial: self = .allowPartial
        case .requireAtLeastOneLayer: self = .requireAtLeastOneLayer
        case .requireAllLayers: self = .requireAllLayers
        }
    }

    /// The policy of a KV-cache configuration.
    var policy: KVCacheConfiguration.CompatibilityPolicy {
        switch self {
        case .allowPartial: .allowPartial
        case .requireAtLeastOneLayer: .requireAtLeastOneLayer
        case .requireAllLayers: .requireAllLayers
        }
    }
}

/// The affine quantization of the KV caches of one handle, which ``RemoteModelOp/newCache(_:quantization:)``
/// carries to every rank: each rank quantizes the attention caches of its own heads.
///
/// It is the ``AffineKVCacheConfiguration`` and the compatibility policy of the request, as
/// ``GenerateParameters/kvBits``, ``GenerateParameters/kvGroupSize`` and
/// ``GenerateParameters/quantizedKVStart`` or an affine ``GenerateParameters/kvCache`` name
/// them. A rank makes its caches with that configuration, and converts them when the app sends
/// ``RemoteModelOp/quantize(_:)``, where a local generation applies its ``KVCachePlan``.
public struct RemoteKVCacheQuantization: Codable, Sendable, Hashable {

    /// The bits of each quantized element: ``AffineKVCacheConfiguration/bits``.
    public let bits: Int

    /// The elements that share one scale: ``AffineKVCacheConfiguration/groupSize``.
    public let groupSize: Int

    /// The positions a cache holds before it quantizes:
    /// ``AffineKVCacheConfiguration/compressionStart``.
    public let compressionStart: Int

    /// What a rank does when only some layers can quantize.
    public let compatibility: RemoteKVCacheCompatibility

    /// Makes a quantization.
    ///
    /// - Parameters:
    ///   - bits: The bits of each quantized element.
    ///   - groupSize: The elements that share one scale.
    ///   - compressionStart: The positions a cache holds before it quantizes.
    ///   - compatibility: What a rank does when only some layers can quantize.
    public init(
        bits: Int, groupSize: Int, compressionStart: Int, compatibility: RemoteKVCacheCompatibility
    ) {
        self.bits = bits
        self.groupSize = groupSize
        self.compressionStart = compressionStart
        self.compatibility = compatibility
    }

    /// The quantization of `configuration`, a KV-cache configuration a worker can make.
    ///
    /// A worker makes the caches of the model itself, thus it takes an affine strategy and
    /// nothing else: no capacity, no rewind reserve, and no other strategy.
    ///
    /// - Parameter configuration: The resolved KV-cache configuration of a request.
    /// - Throws: ``ClusterModelError/unsupported(_:)`` of
    ///   ``ClusterModelFeature/cacheConfiguration`` for any other configuration.
    init(_ configuration: KVCacheConfiguration) throws {
        guard configuration.capacity == nil, configuration.rewind == nil,
            case .affine(let affine) = configuration.strategy.storage
        else {
            throw ClusterModelError.unsupported(.cacheConfiguration)
        }
        self.init(
            bits: affine.bits, groupSize: affine.groupSize,
            compressionStart: affine.compressionStart,
            compatibility: RemoteKVCacheCompatibility(configuration.compatibility))
    }

    /// The KV-cache configuration a rank makes and converts its caches with.
    ///
    /// - Returns: The affine configuration.
    /// - Throws: ``RemoteModelExecutorError/cacheConfiguration(_:)`` for bits, a group size or a
    ///   compression start no affine cache takes.
    func configuration() throws -> KVCacheConfiguration {
        do {
            return KVCacheConfiguration(
                strategy: .affine(
                    try AffineKVCacheConfiguration(
                        bits: bits, groupSize: groupSize, compressionStart: compressionStart)),
                compatibility: compatibility.policy)
        } catch let error as KVCacheConfigurationError {
            throw RemoteModelExecutorError.cacheConfiguration(error.localizedDescription)
        }
    }

    /// Whether a ``KVCachePlan`` of `configuration` quantizes as this quantization does.
    ///
    /// - Parameter configuration: The KV-cache configuration of a generation.
    /// - Returns: `true` when the configuration names this quantization.
    func matches(_ configuration: KVCacheConfiguration) -> Bool {
        (try? Self(configuration)) == self
    }
}
