import Foundation

/// A read-only answer to "where would this run, and roughly how long".
///
/// It reserves nothing and queues nothing -- which makes it both a useful
/// thing to show before someone commits, and the way to check a request is
/// well-formed without spending GPU time on it.
public struct PlacementPreview: Codable, Hashable, Sendable {
    public let outcome: String
    public let reason: String?
    public let candidate: PlacementCandidate?
    public let pendingDownloads: [String]?
    public let missingComponents: [String]?
}

public struct PlacementCandidate: Codable, Hashable, Sendable {
    public let deviceId: String?
    public let predictedStartAfterMs: Int?
    public let predictedCompletionAfterMs: Int?
    public let setupMs: Int?
    /// `cold` means the weights still have to be loaded.
    public let setupKind: String?
    /// `low`, `medium` or `high`. A low-confidence estimate is shown as
    /// approximate rather than as a countdown, because presenting a guess as a
    /// measurement is how a progress bar starts lying.
    public let estimateConfidence: String?

    public var predictedDuration: Duration? {
        guard let ms = predictedCompletionAfterMs else { return nil }
        return .milliseconds(ms)
    }
}

public struct PlacementRequest: Codable, Sendable {
    public let request: GenerateRequest
    public let copies: Int

    public init(request: GenerateRequest, copies: Int = 1) {
        self.request = request
        self.copies = copies
    }
}
