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
    /// What admission would fetch before this could run, each with the
    /// terms that still block it. OBJECTS on the wire (`PendingModelDownload`,
    /// `types.rs`); typed `[String]` here once, which made every preview for
    /// an uninstalled model fail to decode.
    public let pendingDownloads: [PendingModelDownload]?
    public let missingComponents: [ModelComponentStatus]?

    /// Every licence the pending downloads still need, each once, in the
    /// order the machine listed them -- what must be accepted BEFORE a
    /// Generate that would fetch them (the web's `licenseRequirements`).
    public var outstandingLicenses: [LicenseRefusal] { outstandingLicenses(excluding: []) }

    /// The same, less the ids accepted since this answer was read.
    public func outstandingLicenses(excluding accepted: Set<String>) -> [LicenseRefusal] {
        var seen = accepted
        return (pendingDownloads ?? []).flatMap { $0.licenses ?? [] }
            .filter { seen.insert($0.id).inserted }
    }
}

/// One dependency admission will materialize before a render starts
/// (`PendingModelDownload`, `types.rs`). A preview never starts it.
public struct PendingModelDownload: Codable, Hashable, Sendable {
    public let kind: String
    public let name: String
    public let repo: String?
    public let bytes: UInt64?
    /// The manifest bundle to install -- the retry target once the licences
    /// below are accepted. Additive.
    public let installModel: String?
    /// Exact, server-pinned terms that still block this download. Absent or
    /// empty means admission may fetch it on first use.
    public let licenses: [LicenseRefusal]?
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
