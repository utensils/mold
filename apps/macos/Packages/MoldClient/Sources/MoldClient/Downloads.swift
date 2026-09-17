import Foundation

/// Asking a host to fetch a model.
public struct DownloadRequest: Codable, Sendable {
    public let model: String
    public let acceptLicenses: [LicenseAcceptance]

    public init(model: String, acceptLicenses: [LicenseAcceptance] = []) {
        self.model = model
        self.acceptLicenses = acceptLicenses
    }
}

/// What the host says when a download is queued.
///
/// A 409 here is NOT a failure: it means this model is already queued or
/// running, and the body names the job that owns it. Treating it as an error
/// would tell someone their click did nothing when it did the right thing.
public struct DownloadTicket: Codable, Hashable, Sendable {
    public let id: String
    public let position: Int?
}

/// A frame from `GET /api/downloads/stream`.
public struct DownloadEvent: Codable, Sendable {
    public let type: String
    public let id: String?
    public let model: String?
    public let position: Int?
    public let filesDone: Int?
    public let filesTotal: Int?
    public let bytesDone: Int64?
    public let bytesTotal: Int64?
    public let currentFile: String?
    public let error: String?
    /// The FIRST frame every subscriber receives, `type == "snapshot"`
    /// (`types.rs:13065-13070`). It is how a client learns about jobs it did
    /// not start -- a `mold pull` at a terminal, or the web app on the same
    /// machine -- so `DownloadStore` can adopt them on connect rather than
    /// only ever knowing what THIS app queued.
    public let listing: DownloadsListing?

    public var fraction: Double? {
        guard let done = bytesDone, let total = bytesTotal, total > 0 else { return nil }
        return Double(done) / Double(total)
    }

    public var isTerminal: Bool {
        ["job_done", "job_failed", "job_cancelled"].contains(type)
    }
}
