import MoldClient
import SwiftUI

struct QueueDetailSheet: View {
    @Environment(QueueStore.self) private var queue
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    let entry: QueueEntry
    let host: MoldHost
    @State private var detail: QueueEntry?
    @State private var unavailable = false

    private var current: QueueEntry? { queue.current(entry, on: host.id) }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text(queue.headline(for: current ?? entry, on: host.id)).font(.title3.weight(.semibold))
                    Text(host.name).foregroundStyle(.secondaryText)
                    if let current {
                        Text(current.waitDescription).foregroundStyle(.secondaryText)
                        if current.state == .running {
                            Text(ProgressWords.sentence(queue.progress[entry.id]))
                            if let figure = ProgressWords.figure(queue.progress[entry.id]) {
                                Text(figure).font(.callout.monospacedDigit()).foregroundStyle(.secondaryText)
                            }
                        }
                    } else { Text("This job is no longer in the queue.").foregroundStyle(.secondaryText) }
                }
                if let bytes = queue.sourceThumbnail(for: entry, on: host.id), let image = UIImage(data: bytes) {
                    Section {
                        Image(uiImage: image).resizable().scaledToFit().frame(maxHeight: 240)
                            .accessibilityLabel("Source image for this render")
                    } header: { SectionHeader("Source image") }
                }
                if let current {
                    Section {
                        QueueItemActions(entry: current, host: host)
                        if queue.canCancel(current, on: host.id) {
                            Button("Cancel Job", role: .destructive) { Task { await queue.cancel(current, on: host.id) } }
                        }
                    } header: { SectionHeader("Job controls") }
                }
                Section {
                    Text(entry.model ?? "Model").textSelection(.enabled)
                    Text(entry.id).font(.caption).textSelection(.enabled)
                } header: { SectionHeader("Model and job identity") }
                if let metadata = detail?.metadata ?? current?.metadata ?? entry.metadata {
                    ForEach(PrintDetails.groups(for: metadata)) { group in
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
        }
        .presentationDetents([.large])
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
