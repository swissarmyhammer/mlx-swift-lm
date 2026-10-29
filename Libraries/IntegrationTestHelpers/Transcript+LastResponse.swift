// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
extension Transcript {

    /// The last response entry of the transcript, or `nil` when it holds none.
    ///
    /// A session turn leaves its response entry last among the responses of its transcript,
    /// thus a test reads the metadata of a turn from this entry.
    public var lastResponse: Transcript.Response? {
        compactMap { entry -> Transcript.Response? in
            guard case .response(let response) = entry else { return nil }
            return response
        }.last
    }
}

#endif
