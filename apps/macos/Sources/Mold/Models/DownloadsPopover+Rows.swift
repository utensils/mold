import MoldClient
import Foundation

// The popover's rows as pure data. Split from `DownloadsPopover.swift`
// past the file-size advisory.
extension DownloadsPopover {
    /// The popover's row list as pure data: active jobs first, then this
    /// launch's finished ones, exactly the two dictionaries `DownloadStore`
    /// already keeps per machine.
    enum Rows {
        struct Row: Identifiable {
            let id: String
            let model: String
            let bytesDone: Int64?
            let bytesTotal: Int64?
            let currentFile: String?
            let error: String?
            let isActive: Bool

            /// `nil` before a total is known -- a queued job, or an active
            /// one still resolving its manifest -- which is exactly the
            /// case `detailText` reads as "Waiting".
            var fraction: Double? {
                guard let bytesDone, let bytesTotal, bytesTotal > 0 else { return nil }
                return Double(bytesDone) / Double(bytesTotal)
            }

            var isFailed: Bool { !isActive && error != nil }

            var detailText: String {
                guard isActive else { return error ?? "Done" }
                guard let bytesDone, let bytesTotal, bytesTotal > 0 else { return "Waiting" }
                return FileBytes.progress(done: bytesDone, total: bytesTotal)
            }
        }

        /// Active rows are not ordered by the server -- a dictionary keyed
        /// by job id -- so they sort by model name for a stable list;
        /// finished rows keep `DownloadStore.finished`'s own newest-first
        /// order, which is already how it is built.
        static func resolve(active: [String: DownloadStore.Progress], finished: [DownloadJob]) -> [Row] {
            let activeRows = active.map { id, progress in
                Row(
                    id: id, model: progress.model, bytesDone: progress.bytesDone,
                    bytesTotal: progress.bytesTotal, currentFile: progress.currentFile,
                    error: nil, isActive: true)
            }.sorted { $0.model < $1.model }
            let finishedRows = finished.map { job in
                Row(
                    id: job.id, model: job.model, bytesDone: nil, bytesTotal: nil,
                    currentFile: nil, error: job.error, isActive: false)
            }
            return activeRows + finishedRows
        }
    }
}
