import AppKit
import MoldClient
import SwiftUI

/// Job settings from the owning machine, with status kept live by the queue store.
struct QueueDetailSheet: View {
    let entry: QueueEntry
    let host: MoldHost
    @Environment(DownloadStore.self) var downloads
    @Environment(QueueStore.self) var queue
    @Environment(HostStore.self) var hosts
    @Environment(\.dismiss) var dismiss
    @State private var detail: QueueEntry?
    @State var progress: JobProgress?
    @State private var unavailable = false
    @State private var loading = true
    @State var showsTechnicalDetails = false

    var current: QueueEntry? { Self.current(entry, in: queue.entries(on: host.id)) }
    private struct DetailIdentity: Equatable {
        let host: MoldHost?
        let instance: String?
        let up: Bool
        let job: String
        var state: QueueState?
    }
    var activeHost: MoldHost { hosts.host(host.id) ?? host }
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
        VStack(spacing: 0) {
            header
                .padding(20)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    overview
                    if current?.state == .running,
                       let bytes = progress?.previewData, let image = NSImage(data: bytes) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Live preview").font(.headline)
                            Image(nsImage: image).resizable().scaledToFit()
                                .frame(maxWidth: .infinity, maxHeight: 260)
                                .accessibilityLabel("Live render preview")
                        }
                    }
                    if let metadata = detail?.metadata ?? current?.metadata ?? entry.metadata {
                        ForEach(detailGroups(metadata)) { group in
                            QueueDetailFacts(group: group)
                        }
                    }
                    if loading { ProgressView("Reading job settings…").controlSize(.small) }
                    if unavailable {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Full settings are unavailable. The current queue status is shown above.")
                                .font(.callout).foregroundStyle(.secondary)
                            Button("Try Again") { Task { await load() } }
                                .help("Try loading the full settings for this job again")
                        }
                    }
                    technicalDetails
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
            }
            Divider()
            actions
                .padding(16)
        }
        .frame(minWidth: 440, idealWidth: 580, maxWidth: 760,
               minHeight: 440, idealHeight: 620, maxHeight: 820)
        .onAppear { downloads.licenseDetailContext = (host.id, entry.id) }
        .onDisappear {
            if downloads.licenseDetailContext?.host == host.id, downloads.licenseDetailContext?.job == entry.id { downloads.licenseDetailContext = nil }
        }
        .sheet(item: Binding(get: {
            downloads.pendingLicense?.host == host.id && downloads.pendingLicense?.recoveryJob == entry.id ? downloads.pendingLicense : nil
        }, set: { if $0 == nil { downloads.cancelLicense() } })) { pending in LicenseSheet(pending: pending) }
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

    func act(_ action: QueueRow.Action, _ entry: QueueEntry) {
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
