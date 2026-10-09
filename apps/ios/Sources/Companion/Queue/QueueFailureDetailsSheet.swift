import MoldClient
import SwiftUI

struct QueueFailureDetailsSheet: View {
    @Environment(QueueStore.self) private var queue
    @Environment(\.dismiss) private var dismiss
    let entry: QueueEntry
    let host: MoldHost

    private var current: QueueEntry { queue.current(entry, on: host.id) ?? entry }
    private var diagnostic: String { QueueFailureDetails.diagnostic(current, child: queue.child(for: current, on: host.id)) ?? "The machine did not provide failure details." }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text(host.name)
                    Text(current.id).font(.caption.monospaced()).textSelection(.enabled)
                } header: { SectionHeader("Machine and job") }
                Section {
                    Text(diagnostic).font(.callout.monospaced()).textSelection(.enabled)
                        .accessibilityIdentifier("queue-failure-diagnostic")
                    Button {
                        UIPasteboard.general.string = QueueFailureDetails.copyText(current, child: queue.child(for: current, on: host.id), machine: host.name)
                    } label: {
                        Text("Copy Details").fixedSize(horizontal: false, vertical: true)
                    }
                    .buttonStyle(.borderless)
                } header: { SectionHeader("Machine diagnostic") }
                Text("These are the details saved with this job. Older machines may provide only a short reason.")
                    .font(.caption).foregroundStyle(.secondaryText)
            }
            .accessibilityIdentifier("queue-failure-list")
            .navigationTitle("Failure Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }
}
