import AppKit
import MoldClient
import SwiftUI

/// Job settings from the owning machine, with status kept live by the queue store.
struct QueueDetailSheet: View {
    let entry: QueueEntry
    let host: MoldHost
    @Environment(QueueStore.self) private var queue
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @State private var detail: QueueEntry?
    @State private var progress: JobProgress?
    @State private var unavailable = false
    @State private var loading = true

    private var current: QueueEntry? { Self.current(entry, in: queue.entries(on: host.id)) }
    private struct DetailIdentity: Equatable {
        let host: MoldHost?
        let instance: String?
        let up: Bool
        let job: String
        var state: QueueState?
    }
    private var activeHost: MoldHost { hosts.host(host.id) ?? host }
    private var identity: DetailIdentity {
        DetailIdentity(host: hosts.host(host.id), instance: hosts.instanceID(of: host.id),
                       up: hosts.isUp(activeHost), job: entry.id)
    }
    private var previewIdentity: DetailIdentity {
        var value = identity
        value.state = current?.state
        return value
    }

    static func current(_ entry: QueueEntry, in entries: [QueueEntry]) -> QueueEntry? {
        entries.first { $0.id == entry.id }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Job Details").font(.title2.weight(.semibold))
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text((current ?? entry).modelHeadline).font(.headline)
                    Text(activeHost.name).foregroundStyle(.secondary)
                    Text(current?.waitDescription ?? "This job is no longer in the queue.")
                    if current?.state == .running {
                        Text(progress?.stage ?? "Getting ready…")
                        if let step = progress?.step, let total = progress?.total, total > 0 {
                            Text("Step \(step) of \(total)").monospacedDigit()
                        }
                        if let bytes = progress?.previewData, let image = NSImage(data: bytes) {
                            Image(nsImage: image).resizable().scaledToFit().frame(maxHeight: 280)
                                .accessibilityLabel("Live render preview")
                        }
                    }
                    QueueSourceThumbnail(entry: entry, host: activeHost, size: 240)
                    Text(entry.model ?? "Model").textSelection(.enabled)
                    Text(entry.id).font(.caption.monospaced()).textSelection(.enabled)
                    if let metadata = detail?.metadata ?? current?.metadata ?? entry.metadata {
                        ForEach(detailGroups(metadata)) { group in
                            VStack(alignment: .leading, spacing: 8) {
                                Text(group.title).font(.headline)
                                ForEach(group.rows, id: \.label) { row in
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(row.label).foregroundStyle(.secondary)
                                        Text(row.value).textSelection(.enabled)
                                    }
                                }
                            }
                        }
                    }
                    if loading { ProgressView("Reading job settings…") }
                    if unavailable {
                        Text("Full settings are unavailable. The current queue state remains above.")
                            .foregroundStyle(.secondary)
                        Button("Try Again") { Task { await load() } }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                if let current {
                    let actions = QueueRowActions.resolve(current, on: hosts.capabilities[host.id])
                    if actions.pause { Button("Pause") { act(.pause, current) } }
                    if actions.resume { Button("Resume") { act(.resume, current) } }
                    if actions.cancel { Button("Cancel Job", role: .destructive) { act(.cancel, current) } }
                }
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 560, height: 620)
        .accessibilityIdentifier("queue-detail")
        .task(id: identity) { await load() }
        .task(id: previewIdentity) {
            let requestIdentity = previewIdentity
            guard !queue.isSeeded else { return }
            while !Task.isCancelled, current?.state == .running, hosts.isUp(activeHost) {
                if let value = try? await hosts.backend(for: activeHost).jobPreview(jobId: entry.id), !Task.isCancelled, previewIdentity == requestIdentity { progress = value }
                try? await Task.sleep(for: .seconds(1))
            }
            if previewIdentity == requestIdentity { progress = nil }
        }
    }

    private func act(_ action: QueueRow.Action, _ entry: QueueEntry) {
        Task {
            switch action {
            case .pause: await queue.pause(entry, on: host.id)
            case .resume: await queue.resume(entry, on: host.id)
            case .cancel: await queue.cancel(entry, on: host.id)
            case .retry: await queue.retry(entry, on: host.id)
            }
            await queue.refresh(on: host.id)
        }
    }

    private func detailGroups(_ metadata: OutputMetadata) -> [PrintDetailGroup] {
        var displaying = current ?? entry
        displaying.metadata = metadata
        if let pinned = detail?.seedPinned { displaying.seedPinned = pinned }
        return PrintDetails.groups(for: displaying)
    }

    private func load() async {
        loading = true
        detail = nil
        let requestIdentity = identity
        defer { if !Task.isCancelled, identity == requestIdentity { loading = false } }
        guard !queue.isSeeded, hosts.host(host.id) != nil, hosts.isUp(activeHost) else { unavailable = true; return }
        do {
            let result = try await hosts.backend(for: activeHost).queueJob(id: entry.id)
            guard !Task.isCancelled, identity == requestIdentity else { return }
            detail = result.job
            unavailable = false
        } catch {
            guard !Task.isCancelled, identity == requestIdentity else { return }
            unavailable = true
        }
    }
}

struct QueueDetailTarget: Identifiable {
    let entry: QueueEntry
    let host: MoldHost
    var id: String { "\(host.id)|\(entry.id)" }
}
