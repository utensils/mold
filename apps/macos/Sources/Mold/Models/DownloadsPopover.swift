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
        .contentShape(Rectangle())
        .rowActionMenu(menu(for: row)) { _ in
            Task { await downloads.cancel(jobID: row.id, on: host) }
        }
    }

    /// The row's own ✕ a second way, in the Model menu's words
    /// ("Cancel Download", `ModelActions+Menu.swift`) rather than a third
    /// spelling. A finished row has nothing to act on -- Clear is about the
    /// whole list, not this row -- so it gets no menu at all rather than one
    /// holding a disabled item.
    private func menu(for row: Rows.Row) -> [RowAction<String>] {
        guard row.isActive else { return [] }
        return [RowAction(kind: "cancel", title: "Cancel Download", isDestructive: true)]
    }
}
