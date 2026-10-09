import MoldClient
import SwiftUI

struct QueueDetailSheet: View {
    @Environment(AppRouter.self) private var router
    @Environment(ModelStore.self) private var models
    @Environment(QueueStore.self) private var queue
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    let entry: QueueEntry
    let host: MoldHost
    @State private var detail: QueueEntry?
    @State private var unavailable = false
    @State private var showingFailure = false

    private var current: QueueEntry? { queue.current(entry, on: host.id) }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text(queue.headline(for: current ?? entry, on: host.id)).font(.title3.weight(.semibold))
                    Text(host.name).foregroundStyle(.secondaryText)
                    if let current {
                        Text(current.state == .held ? "Held" : current.waitDescription).foregroundStyle(.secondaryText)
                        if current.state == .running {
                            Text(ProgressWords.sentence(queue.progress[entry.id]))
                            if let figure = ProgressWords.figure(queue.progress[entry.id]) {
                                Text(figure).font(.callout.monospacedDigit()).foregroundStyle(.secondaryText)
                            }
                        }
                    } else { Text("This job is no longer in the queue.").foregroundStyle(.secondaryText) }
                }
                if current?.state == .running, let bytes = queue.progress[entry.id]?.previewData,
                   let image = UIImage(data: bytes) {
                    Section {
                        Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 320)
                            .accessibilityLabel("Live render preview")
                    } header: { SectionHeader("Rendering preview") }
                }
                let inputs = queue.inputPreviews(for: entry, on: host.id)
                if !inputs.isEmpty || queue.inputLoadFailed(for: entry, on: host.id) {
                    Section {
                        if queue.inputLoadFailed(for: entry, on: host.id) {
                            Text("Input previews unavailable").foregroundStyle(.secondaryText)
                        }
                        ForEach(inputs) { preview in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(preview.input.label).font(.headline)
                                if let bytes = preview.bytes, let image = UIImage(data: bytes) {
                                    Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 320)
                                        .accessibilityLabel(preview.input.label)
                                } else {
                                    Text(preview.input.preview ? "Preview unavailable" : "No still preview").foregroundStyle(.secondaryText)
                                }
                            }
                        }
                        if queue.inputLoadFailed(for: entry, on: host.id) || inputs.contains(where: { $0.input.preview && $0.bytes == nil }) {
                            Button("Retry input previews") { Task { await queue.loadSourceThumbnail(for: entry, on: host.id, detailed: true, retry: true) } }
                        }
                    } header: { SectionHeader("Input images and references") }
                }
                if let current {
                    Section {
                        QueueItemActions(entry: current, host: host, detail: true)
                        if queue.canCancel(current, on: host.id) {
                            Button("Cancel Job", role: .destructive) { Task { await queue.cancel(current, on: host.id) } }
                        }
                    } header: { SectionHeader("Job controls") }
                }
                Section {
                    Text(entry.model ?? "Model").textSelection(.enabled)
                    Text(entry.id).font(.caption).textSelection(.enabled)
                    if QueueFailureDetails.diagnostic(current ?? entry, child: queue.child(for: current ?? entry, on: host.id)) != nil {
                        Button { showingFailure = true } label: {
                            Text("Failure Details").fixedSize(horizontal: false, vertical: true)
                        }
                        .buttonStyle(.borderless)
                    }
                } header: { SectionHeader("Model and job identity") }
                if let metadata = detail?.metadata ?? current?.metadata ?? entry.metadata {
                    ForEach(detailGroups(metadata)) { group in
                        Section {
                            ForEach(group.rows, id: \.label) { row in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(row.label).foregroundStyle(.secondaryText)
                                    Text(row.value).textSelection(.enabled)
                                }
                            }
                        } header: { SectionHeader(group.title) }
                    }
                }
                if unavailable {
                    Text("Full settings are unavailable. The current queue state remains above.").foregroundStyle(.secondaryText)
                    Button("Try Again") { Task { await load() } }
                }
            }
            .accessibilityIdentifier("queue-detail")
            .navigationTitle("Job Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .task(id: "\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(hosts.isUp(host))|\(entry.id)") { await load() }
            .task(id: "inputs|\(host.id)|\(hosts.instanceID(of: host.id) ?? "unknown")|\(hosts.isUp(host))|\(entry.id)") {
                await queue.loadSourceThumbnail(for: entry, on: host.id, detailed: true)
            }
        }
        .onAppear { router.licenseDetailContext = (host.id, entry.id) }
        .onDisappear {
            if router.licenseDetailContext?.host == host.id, router.licenseDetailContext?.job == entry.id { router.licenseDetailContext = nil }
        }
        .sheet(item: Binding(get: {
            models.pendingLicense?.host == host.id && models.pendingLicense?.presentationOwner == router.presentationID && models.pendingLicense?.recoveryJob == entry.id ? models.pendingLicense : nil
        }, set: { if $0 == nil { models.cancelLicense() } })) { pending in LicenceSheet(pending: pending) }
        .sheet(isPresented: $showingFailure) { QueueFailureDetailsSheet(entry: current ?? entry, host: host) }
        .presentationDetents([.large])
    }

    private func detailGroups(_ metadata: OutputMetadata) -> [PrintDetailGroup] {
        var displaying = current ?? entry
        displaying.metadata = metadata
        if let pinned = detail?.seedPinned { displaying.seedPinned = pinned }
        return PrintDetails.groups(for: displaying)
    }

    private func load() async {
        guard let host = hosts.host(host.id), hosts.isUp(host) else { unavailable = true; return }
        do {
            let result = try await hosts.backend(for: host).queueJob(id: entry.id)
            guard !Task.isCancelled else { return }
            detail = result.job; unavailable = false
        } catch is CancellationError { return } catch {
            guard !Task.isCancelled else { return }; unavailable = true
        }
    }
}
