import MoldClient
import SwiftUI

/// One group: a plain row, or a batch as a parent row whose children open
/// beneath it.
struct QueueGroupRows: View {
    let group: QueueGroup
    let host: MoldHost

    var body: some View {
        if group.isExpandable {
            DisclosureGroup {
                ForEach(group.rows) { entry in QueueEntryRow(entry: entry, host: host, inBatch: true) }
            } label: {
                QueueBatchLabel(rows: group.rows)
            }
        } else {
            QueueEntryRow(entry: group.rows[0], host: host, inBatch: false)
        }
    }
}

/// "Batch of 4 · flux-dev:q4" over "1 rendering · 3 waiting".
private struct QueueBatchLabel: View {
    let rows: [QueueEntry]

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Batch of \(rows.count)")
            if let model = rows.first?.model {
                Text(verbatim: model).font(.caption.monospaced()).foregroundStyle(.secondaryText)
            }
            Text(summary).font(.caption).foregroundStyle(.secondaryText)
        }
        .accessibilityElement(children: .combine)
    }

    private var summary: String {
        let running = rows.filter { $0.state == .running }.count
        let held = rows.filter { $0.state == .held }.count
        let waiting = rows.count - running - held
        var parts: [String] = []
        if running > 0 { parts.append(String(localized: "\(running) rendering")) }
        if waiting > 0 { parts.append(String(localized: "\(waiting) waiting")) }
        if held > 0 { parts.append(String(localized: "\(held) held")) }
        return parts.joined(separator: " · ")
    }
}

/// A job: its model, where it stands in words, the preview while it runs,
/// and -- when held -- the paragraph and the named buttons that answer it.
struct QueueEntryRow: View {
    @Environment(QueueStore.self) private var queue
    @Environment(\.dynamicTypeSize) private var size
    @ScaledMetric(relativeTo: .body) private var thumb = 52
    let entry: QueueEntry
    let host: MoldHost
    let inBatch: Bool

    var body: some View {
        let layout = RowAxis.for(size) == .horizontal
            ? AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
        layout {
            if entry.state == .running { preview }
            VStack(alignment: .leading, spacing: 4) {
                title
                if let hold = queue.hold(for: entry, on: host.id) {
                    QueueHeldActions(entry: entry, hold: hold, host: host)
                } else if entry.state == .running {
                    running
                } else {
                    Text(caption).font(.callout).foregroundStyle(.secondaryText)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 2)
        .swipeActions(edge: .trailing) {
            if queue.canCancel(entry, on: host.id) {
                Button(role: .destructive) { Task { await queue.cancel(entry, on: host.id) } } label: {
                    Label("Cancel", systemImage: "xmark")
                }
            }
        }
        .swipeActions(edge: .leading) {
            if queue.canPause(entry, on: host.id) {
                let paused = entry.state == .paused
                Button { Task { await queue.setPaused(!paused, entry, on: host.id) } } label: {
                    Label(paused ? "Resume" : "Pause", systemImage: paused ? "play" : "pause")
                }
                .tint(.orange)
            }
        }
        .contextMenu { menu }
    }

    private var title: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            if entry.state == .held {
                // a11y: the paragraph below says it is held, in words.
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange).accessibilityHidden(true)
            }
            Text(verbatim: entry.model ?? String(localized: "Unknown model"))
                .font(.body.monospaced())
        }
    }

    private var caption: String {
        if inBatch, let index = entry.batchIndex {
            return String(localized: "Picture \(index) · \(entry.waitDescription)")
        }
        return entry.waitDescription
    }

    @ViewBuilder private var running: some View {
        let progress = queue.progress[entry.id]
        Text(ProgressWords.sentence(progress)).font(.callout)
        if let step = progress?.step, let total = progress?.total, total > 0 {
            ProgressView(value: Double(step), total: Double(total))
                .accessibilityValue(ProgressWords.spoken(progress))
        } else {
            ProgressView().frame(maxWidth: .infinity, alignment: .leading)
        }
        if let figure = ProgressWords.figure(progress) {
            Text(verbatim: "\(figure) · \(host.name)").font(.caption.monospacedDigit()).foregroundStyle(.secondaryText)
        }
    }

    @ViewBuilder private var preview: some View {
        let image = queue.progress[entry.id]?.previewData.flatMap(UIImage.init(data:))
        Group {
            if let image {
                Image(uiImage: image).resizable().scaledToFill()
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .frame(width: thumb, height: thumb)
        .clipShape(.rect(cornerRadius: 5))
        .accessibilityHidden(true) // a11y: the sentence beside it says what it shows.
    }

    /// The Mac row menu's items, in its order; Cancel last, behind a divider.
    @ViewBuilder private var menu: some View {
        if entry.state.isReorderable, queue.canReorder(on: host.id) {
            Button("Move Up", systemImage: "arrow.up") { Task { await queue.move(entry, up: true, on: host.id) } }
            Button("Move Down", systemImage: "arrow.down") { Task { await queue.move(entry, up: false, on: host.id) } }
        }
        if entry.state == .held {
            MoveToMenu(entry: entry, host: host)
        }
        if queue.canPause(entry, on: host.id) {
            let paused = entry.state == .paused
            Button(paused ? "Resume" : "Pause", systemImage: paused ? "play" : "pause") {
                Task { await queue.setPaused(!paused, entry, on: host.id) }
            }
        }
        if queue.canCancel(entry, on: host.id) {
            Divider()
            Button("Cancel Job", systemImage: "xmark", role: .destructive) {
                Task { await queue.cancel(entry, on: host.id) }
            }
        }
    }
}

/// A held row's paragraph and its named buttons: Pull (then Retry) for a
/// missing model, Retry where the machine says it would help, Move to…
/// another machine, and Cancel -- side by side, stacked full width at AX
/// sizes.
private struct QueueHeldActions: View {
    @Environment(QueueStore.self) private var queue
    @Environment(ModelStore.self) private var models
    @Environment(\.dynamicTypeSize) private var size
    let entry: QueueEntry
    let hold: QueueHold
    let host: MoldHost

    var body: some View {
        Text(sentence).font(.callout).foregroundStyle(.secondaryText)
        let stacked = RowAxis.for(size) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
                             : AnyLayout(HStackLayout(spacing: 8))
        layout {
            if case let .missingModel(model, _) = hold {
                Button("Pull and Retry") { models.pullThenRetry(model, entry: entry, on: host.id) }
                    .frame(maxWidth: stacked ? .infinity : nil)
            } else if case .prose(_, retryable: true) = hold {
                Button("Retry") { Task { await queue.retry(entry, on: host.id) } }
                    .frame(maxWidth: stacked ? .infinity : nil)
            }
            MoveToMenu(entry: entry, host: host)
                .frame(maxWidth: stacked ? .infinity : nil)
        }
        .buttonStyle(.bordered)
        .padding(.top, 4)
    }

    private var sentence: String {
        switch hold {
        case let .missingModel(_, sentence), let .prose(sentence, _):
            sentence.isEmpty ? String(localized: "The machine put this job aside.") : sentence
        }
    }
}

/// "Move to…": the other machines that are up and generate. Absent when
/// there are none.
struct MoveToMenu: View {
    @Environment(TransferStore.self) private var transfers
    let entry: QueueEntry
    let host: MoldHost

    var body: some View {
        let destinations = transfers.destinations(from: host.id)
        if !destinations.isEmpty {
            Menu {
                ForEach(destinations) { destination in
                    Button(destination.name) {
                        Task { await transfers.transfer(entry, from: host.id, to: destination.id) }
                    }
                }
            } label: {
                Label("Move to…", systemImage: "arrow.right.circle")
            }
            .disabled(transfers.transferring != nil)
        }
    }
}
