import Foundation

/// One download a machine is running, as a client draws it.
public struct DownloadProgress: Hashable, Sendable {
    public var model: String
    public var fraction: Double?
    public var bytesDone: Int64?
    public var bytesTotal: Int64?
    public var currentFile: String?
    public var failed: String?

    public init(model: String, fraction: Double? = nil, bytesDone: Int64? = nil, bytesTotal: Int64? = nil,
                currentFile: String? = nil, failed: String? = nil) {
        self.model = model
        self.fraction = fraction
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
        self.currentFile = currentFile
        self.failed = failed
    }

    /// "2.1 / 11.8 GB · 42 MB/s": bytes in decimal units, the unit named
    /// once, by the total. "Starting…" until the machine knows a size.
    public func sentence(bytesPerSecond: Double?) -> String {
        guard let total = bytesTotal, total > 0 else { return "Starting…" }
        let (scale, unit): (Double, String) =
            total >= 1_000_000_000 ? (1e9, "GB") : total >= 1_000_000 ? (1e6, "MB") : (1e3, "KB")
        let done = Double(bytesDone ?? 0) / scale
        var words = "\(Self.figure(done, below: 100)) / \(Self.figure(Double(total) / scale, below: 100)) \(unit)"
        if let rate = bytesPerSecond, rate > 0 {
            words += rate >= 1e6 ? " · \(Self.figure(rate / 1e6)) MB/s" : " · \(Self.figure(rate / 1e3)) KB/s"
        }
        return words
    }

    /// One decimal under `below` (sizes: 100, rates: 10), whole numbers from there.
    private static func figure(_ value: Double, below: Double = 10) -> String {
        value < below ? String(format: "%.1f", value) : String(Int(value.rounded()))
    }
}

/// The reducer over `GET /api/downloads/stream` that the Mac and the phone
/// both draw from: `DownloadEvent.effect` says what a frame may do; this does
/// it. Pure, so every arm is tested without a machine.
public enum DownloadBoard {
    /// A listing (`GET /api/downloads` or the stream's snapshot frame) as the
    /// whole board: running and queued jobs, by job id.
    public static func adopt(_ listing: DownloadsListing) -> [String: DownloadProgress] {
        var board: [String: DownloadProgress] = [:]
        for job in listing.activeJobs + listing.queued {
            board[job.id] = DownloadProgress(
                model: job.model,
                fraction: job.bytesTotal > 0 ? Double(job.bytesDone) / Double(job.bytesTotal) : nil,
                bytesDone: job.bytesDone, bytesTotal: job.bytesTotal,
                currentFile: job.currentFile, failed: job.error)
        }
        return board
    }

    /// Applies one delta frame. Answers the finished job when the frame
    /// settled one -- remembered with the last figures the row had, since the
    /// server keeps no history to re-read. Snapshot frames are the caller's
    /// (`adopt` their listing).
    public static func apply(_ event: DownloadEvent, to board: inout [String: DownloadProgress]) -> DownloadJob? {
        guard let id = event.id else { return nil }
        switch event.effect {
        case .settle:
            let last = board.removeValue(forKey: id)
            let status: JobStatus =
                switch event.type {
                case "job_done": .completed
                case "job_cancelled": .cancelled
                default: .failed
                }
            return DownloadJob(
                id: id, model: event.model ?? last?.model ?? "", status: status,
                bytesDone: event.bytesDone ?? last?.bytesDone ?? 0,
                bytesTotal: event.bytesTotal ?? last?.bytesTotal ?? 0,
                currentFile: event.currentFile ?? last?.currentFile, error: event.error ?? last?.failed)
        case .forget:
            board.removeValue(forKey: id)
        case .introduce:
            board[id] = moved(board[id] ?? DownloadProgress(model: event.model ?? ""), by: event)
        case .update:
            if let known = board[id] { board[id] = moved(known, by: event) }
        case .snapshot, .ignore:
            break
        }
        return nil
    }

    private static func moved(_ row: DownloadProgress, by event: DownloadEvent) -> DownloadProgress {
        var progress = row
        if let model = event.model { progress.model = model }
        progress.fraction = event.fraction ?? progress.fraction
        progress.bytesDone = event.bytesDone ?? progress.bytesDone
        progress.bytesTotal = event.bytesTotal ?? progress.bytesTotal
        progress.currentFile = event.currentFile ?? progress.currentFile
        return progress
    }
}
