import MoldClient
import SwiftUI

/// What this MACHINE is doing, not what this app started -- `DownloadStore`
/// adopts a `mold pull` at a terminal and the web app's own clicks the same
/// way it adopts its own (decision 15, M5), so this popover is one list for
/// whatever is true of the host right now.
struct DownloadsPopover: View {
    let host: MoldHost
    @Environment(DownloadStore.self) private var downloads
    @Environment(ModelStore.self) private var models

    private var active: [String: DownloadStore.Progress] { downloads.active[host.id] ?? [:] }
    private var finished: [DownloadJob] { downloads.finished[host.id] ?? [] }
    private var rows: [Rows.Row] { Rows.resolve(active: active, finished: finished) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(rows) { row in
                rowView(row)
                if row.id != rows.last?.id {
                    Divider()
                }
            }
            if !finished.isEmpty {
                Divider()
                HStack {
                    Spacer()
                    Button("Clear") { downloads.clearFinished(on: host.id) }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
            }
        }
        .padding(.vertical, 4)
        .frame(width: 300)
    }

    private func rowView(_ row: Rows.Row) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(row.model).font(.callout.weight(.medium)).lineLimit(1)
                    if let tradeOff = models.model(named: row.model, on: host.id)?.tradeOff {
                        Text(tradeOff).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer()
                if row.isActive {
                    Button {
                        Task { await downloads.cancel(jobID: row.id, on: host) }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .help("Cancel this download")
                }
            }
            if row.isActive, let fraction = row.fraction {
                ProgressView(value: fraction).progressViewStyle(.linear)
            }
            Text(row.detailText)
                .font(.caption2)
                .foregroundStyle(row.isFailed ? .red : .secondary)
                .lineLimit(1)
            if row.isActive, let currentFile = row.currentFile {
                Text(currentFile)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }
}

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
                return "\(bytes(bytesDone)) of \(bytes(bytesTotal))"
            }

            private func bytes(_ count: Int64) -> String {
                count.formatted(.byteCount(style: .file))
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
