import Foundation

/// `GET /api/downloads`. Every job this machine is running, queued, or has
/// finished, whoever asked for it -- from this app, the CLI, or another
/// client on the same machine (`types.rs:13274-13285`).
public struct DownloadsListing: Codable, Hashable, Sendable {
    /// Every download currently transferring. New clients read THIS, not
    /// `active` -- which remains only as a compatibility view of the first
    /// job.
    public let activeJobs: [DownloadJob]
    public let active: DownloadJob?
    public let queued: [DownloadJob]
    public let history: [DownloadJob]

    public init(
        activeJobs: [DownloadJob] = [], active: DownloadJob? = nil,
        queued: [DownloadJob] = [], history: [DownloadJob] = []
    ) {
        self.activeJobs = activeJobs
        self.active = active
        self.queued = queued
        self.history = history
    }
}

/// One entry in a `DownloadsListing` (`types.rs:13045-13062`).
public struct DownloadJob: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let model: String
    public let catalogId: String?
    public let status: JobStatus
    public let filesDone: Int
    public let filesTotal: Int
    public let bytesDone: Int64
    public let bytesTotal: Int64
    public let currentFile: String?
    public let startedAt: Int64?
    public let completedAt: Int64?
    public let error: String?

    /// A client-side record of what a live stream frame already said, for a
    /// job that just went terminal -- `DownloadStore` builds one of these to
    /// keep in its bounded `finished` list, since the server itself retains
    /// no history for the popover to re-read (design fact/decision 15, M5).
    public init(
        id: String, model: String, catalogId: String? = nil, status: JobStatus,
        filesDone: Int = 0, filesTotal: Int = 0, bytesDone: Int64 = 0, bytesTotal: Int64 = 0,
        currentFile: String? = nil, startedAt: Int64? = nil, completedAt: Int64? = nil,
        error: String? = nil
    ) {
        self.id = id
        self.model = model
        self.catalogId = catalogId
        self.status = status
        self.filesDone = filesDone
        self.filesTotal = filesTotal
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
        self.currentFile = currentFile
        self.startedAt = startedAt
        self.completedAt = completedAt
        self.error = error
    }
}

/// A download's lifecycle state (`types.rs:13031-13039`), decoded leniently:
/// a status this build has never heard of becomes `.unknown` rather than
/// failing the whole listing -- the same rule every other open wire enum
/// here follows.
public enum JobStatus: String, OpenWireEnum {
    case queued, active, completed, failed, cancelled
    case unknown
}
