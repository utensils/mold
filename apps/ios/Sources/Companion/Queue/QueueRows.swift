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
                QueueBatchLabel(rows: group.rows, host: host)
            }
        } else {
            QueueEntryRow(entry: group.rows[0], host: host, inBatch: false)
        }
    }
}

/// "Batch of 4 · flux-dev:q4" over "1 rendering · 3 waiting".
private struct QueueBatchLabel: View {
    @Environment(QueueStore.self) private var queue
    let rows: [QueueEntry]
    let host: MoldHost

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Batch of \(rows.count)")
            if let row = rows.first {
                Text(verbatim: queue.headline(for: row, on: host.id)).font(.caption).foregroundStyle(.secondaryText)
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
/// and, when held, its explanation, transfer menu and diagnostics.
struct QueueEntryRow: View {
    @Environment(HostStore.self) private var hosts
    @Environment(QueueStore.self) private var queue
    @Environment(AppRouter.self) private var router
    @Environment(ModelStore.self) private var models
    @Environment(\.dynamicTypeSize) private var size
    @ScaledMetric(relativeTo: .body) private var thumb = 52
    let entry: QueueEntry
    let host: MoldHost
    let inBatch: Bool
    @State private var inspecting: QueueEntry?
    @State private var showingFailure = false

    var body: some View {
        let layout = RowAxis.for(size) == .horizontal
            ? AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
        VStack(alignment: .leading, spacing: 8) {
            Button { inspecting = entry } label: {
                layout {
                    inputPreview
                    VStack(alignment: .leading, spacing: 6) {
                        Text(queue.headline(for: entry, on: host.id)).font(.headline)
                            .foregroundStyle(.primary)
                        if let prompt = queue.prompt(for: entry, on: host.id), !prompt.isEmpty {
                            Text(prompt).font(.callout).foregroundStyle(.primary)
                                .lineLimit(size.isAccessibilitySize ? nil : 2)
                        }
                        HStack {
                            Text(entry.state == .held ? String(localized: "Held") : entry.state == .running ? String(localized: "Rendering") : caption)
                            Spacer()
                            Image(systemName: "chevron.right").accessibilityHidden(true)
                        }
                        .font(.callout).foregroundStyle(.secondaryText)
                        if entry.state == .running, let step = queue.progress[entry.id]?.step,
                           let total = queue.progress[entry.id]?.total, total > 0 {
                            ProgressView(value: Double(step), total: Double(total))
                                .accessibilityValue(ProgressWords.spoken(queue.progress[entry.id]))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("queue-open-" + entry.id)
            .accessibilityHint("Show job details and controls")
            QueueItemActions(entry: entry, host: host)
            if entry.state != .held { MoveToMenu(entry: entry, host: host) }
            if QueueFailureDetails.diagnostic(entry, child: queue.child(for: entry, on: host.id)) != nil {
                Button { showingFailure = true } label: {
                    Text("Failure Details").fixedSize(horizontal: false, vertical: true)
                }
                .buttonStyle(.borderless)
                .accessibilityIdentifier("queue-failure-details-" + entry.id)
            }
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("queue-entry-" + entry.id)
        .sheet(isPresented: $showingFailure) { QueueFailureDetailsSheet(entry: entry, host: host) }
        .sheet(item: $inspecting) { row in QueueDetailSheet(entry: row, host: host) }
        .task(id: "\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(hosts.isUp(host))|\(entry.id)") {
            await queue.loadSourceThumbnail(for: entry, on: host.id)
        }
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            if queue.canCancel(entry, on: host.id) {
                Button(role: .destructive) { Task { await queue.cancel(entry, on: host.id) } } label: {
                    Label("Cancel", systemImage: "xmark")
                }
                .accessibilityIdentifier("queue-swipe-cancel-" + entry.id)
            }
        }
        .swipeActions(edge: .leading, allowsFullSwipe: false) {
            retryAction
            if !queue.canPause(entry, on: host.id), !retryAvailable {
                Button("Details", systemImage: "info.circle") { inspecting = entry }
                    .accessibilityIdentifier("queue-swipe-details-" + entry.id)
            }
            if queue.canPause(entry, on: host.id) {
                let paused = entry.state == .paused
                Button { Task { await queue.setPaused(!paused, entry, on: host.id) } } label: {
                    Label(paused ? "Resume" : "Pause", systemImage: paused ? "play" : "pause")
                }
                .tint(.orange)
                .accessibilityIdentifier("queue-swipe-pause-" + entry.id)
            }
        }
        .contextMenu { menu }
    }

    private var retryAvailable: Bool {
        queue.canRetry(entry, on: host.id) && models.queueDownloads.state(host: host.id, job: entry.id)?.isBusy != true
    }

    @ViewBuilder private var retryAction: some View {
        if retryAvailable {
            if let hold = queue.hold(for: entry, on: host.id), case .missingModel = hold {
                Button("Download and Retry", systemImage: "arrow.down.circle") {
                    guard retryAvailable else { return }
                    if let model = entry.model {
                        models.pullThenRetry(model, entry: entry, on: host.id, presenter: router.presentationID)
                    }
                }
                .accessibilityIdentifier("queue-swipe-download-" + entry.id)
            } else {
                Button("Retry", systemImage: "arrow.clockwise") {
                    guard retryAvailable else { return }
                    Task { await queue.retry(entry, on: host.id) }
                }
                .accessibilityIdentifier("queue-swipe-retry-" + entry.id)
            }
        }
    }

    private var caption: String {
        if inBatch, let index = entry.batchIndex {
            return String(localized: "Picture \(index) · \(entry.state == .held ? "Held" : entry.waitDescription)")
        }
        return entry.state == .held ? String(localized: "Held") : entry.waitDescription
    }

    @ViewBuilder private var inputPreview: some View {
        if let data = queue.sourceThumbnail(for: entry, on: host.id), let image = UIImage(data: data) {
            VStack(alignment: .leading, spacing: 4) {
                Image(uiImage: image).resizable().scaledToFill()
                    .frame(width: thumb, height: thumb)
                    .clipShape(.rect(cornerRadius: 7))
                    .accessibilityLabel(queue.inputPreviews(for: entry, on: host.id).first(where: { $0.bytes != nil })?.input.label ?? "Source")
                    .accessibilityIdentifier("queue-source-" + entry.id)
                Text(inputCaption).font(.caption).foregroundStyle(.secondaryText)
            }
        }
    }

    private var inputCaption: String {
        let inputs = queue.inputPreviews(for: entry, on: host.id)
        let label = inputs.first(where: { $0.bytes != nil })?.input.label ?? "Source"
        return inputs.count > 1 ? "\(label) +\(inputs.count - 1)" : label
    }

    /// The Mac row menu's items, in its order; Cancel last, behind a divider.
    @ViewBuilder private var menu: some View {
        retryAction
        if queue.canMove(entry, on: host.id) {
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

/// A held row keeps its explanation, download progress and Move to menu.
/// Job Details also offers explicit Retry and Download and Retry controls.
/// Card recovery actions are native swipes; details keep explicit recovery controls.
struct QueueHeldActions: View {
    @Environment(AppRouter.self) private var router
    @Environment(QueueStore.self) private var queue
    @Environment(ModelStore.self) private var models
    @Environment(\.dynamicTypeSize) private var size
    let entry: QueueEntry
    let hold: QueueHold
    let host: MoldHost
    var detail = false

    private var recovery: QueueDownloadRecovery.State? { models.queueDownloads.state(host: host.id, job: entry.id) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(hold.summary(modelName: queue.headline(for: entry, on: host.id), hostName: host.name))
                .font(.callout).foregroundStyle(.secondaryText)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("queue-held-reason-" + entry.id)
            if let recovery {
                Text(recovery.message).font(.callout).foregroundStyle(.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier((detail ? "queue-detail-download-status-" : "queue-download-status-") + entry.id)
                if let fraction = recovery.fraction {
                    ProgressView(value: fraction).accessibilityLabel("Model download")
                } else if recovery.isBusy { ProgressView().accessibilityLabel(recovery.message) }
            }
            if RowAxis.for(size) == .vertical { buttons(stacked: true) } else {
                ViewThatFits(in: .horizontal) { buttons(stacked: false); buttons(stacked: true) }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .buttonStyle(.bordered)
        .padding(.top, 4)
    }

    private func buttons(stacked: Bool) -> some View {
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8)) : AnyLayout(HStackLayout(spacing: 8))
        return layout {
            if detail, case .missingModel = hold, queue.canRetry(entry, on: host.id) {
                Button(recovery?.isBusy == true ? "Downloading…" : "Download and Retry") {
                    if let model = entry.model { models.pullThenRetry(model, entry: entry, on: host.id, presenter: router.presentationID) }
                }
                .disabled(recovery?.isBusy == true)
                .frame(maxWidth: stacked ? .infinity : nil)
                .fixedSize(horizontal: !stacked, vertical: false)
                .accessibilityIdentifier((detail ? "queue-detail-download-" : "queue-download-") + entry.id)
            } else if detail, queue.canRetry(entry, on: host.id) {
                Button("Retry") { Task { await queue.retry(entry, on: host.id) } }
                    .frame(maxWidth: stacked ? .infinity : nil)
            }
            if queue.canTransfer(entry, on: host.id) {
                MoveToMenu(entry: entry, host: host)
            }
        }
    }
}

/// "Move to…": the other machines that are up and generate. Absent when
/// there are none.
struct MoveToMenu: View {
    @Environment(TransferStore.self) private var transfers
    @Environment(QueueStore.self) private var queue
    let entry: QueueEntry
    let host: MoldHost

    var body: some View {
        let destinations = transfers.destinations(from: host.id)
        if !destinations.isEmpty, queue.canTransfer(entry, on: host.id) {
            Menu {
                ForEach(destinations) { destination in
                    Button(destination.name) {
                        Task { await transfers.transfer(entry, from: host.id, to: destination.id) }
                    }
                }
            } label: {
                Label("Move to…", systemImage: "arrow.right.circle")
            }
            .accessibilityIdentifier("queue-move-to-" + entry.id)
            .disabled(transfers.transferring != nil)
        }
    }
}
