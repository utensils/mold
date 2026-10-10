import MoldClient
import SwiftUI

struct PromptHistorySheet: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    private let recall: ((String) -> Void)?
    @State private var history: PromptHistoryStore
    @State private var query = ""
    @State private var confirmsClear = false

    init(host: MoldHost, hosts: HostStore, recall: ((String) -> Void)? = nil) {
        self.recall = recall
        _history = State(initialValue: PromptHistoryStore(host: host, hosts: hosts))
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Picker("Machine", selection: Binding(get: { history.host.id }, set: { id in
                        guard id != history.host.id else { return }
                        if let host = hosts.host(id) { history = PromptHistoryStore(host: host, hosts: hosts) }
                    })) {
                        ForEach(hosts.hosts) { host in Text(host.name).tag(host.id) }
                    }
                    .disabled(history.clearing)
                    .accessibilityIdentifier("history-machine")
                    Text("Use a prompt without changing your model, settings, or source images.")
                        .font(.callout).foregroundStyle(.secondaryText)
                }
                notice
                ForEach(history.entries) { entry in
                    Button {
                        if let recall { recall(entry.prompt) } else {
                            PromptHistoryStore.recall(entry.prompt, into: &generate.draft)
                        }
                        dismiss()
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            Text(entry.prompt).foregroundStyle(.primary).lineLimit(4)
                            Text(modelTitle(entry)).font(.caption).foregroundStyle(.secondaryText)
                            Text(entry.usedAtDate, format: .dateTime.month().day().hour().minute())
                                .font(.caption).foregroundStyle(.secondaryText)
                        }
                        .padding(.vertical, 4)
                    }
                    .disabled(history.clearing)
                    .accessibilityHint("Use this prompt")
                }
                if history.state == .ready, history.entries.isEmpty {
                    Text(query.isEmpty ? "No prompts yet on this machine." : "No matching prompts.")
                        .foregroundStyle(.secondaryText)
                }
            }
            .navigationTitle("Prompt History")
            .navigationBarTitleDisplayMode(.inline)
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: "Search prompts")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
                ToolbarItem(placement: .topBarLeading) {
                    Button("Clear", role: .destructive) { confirmsClear = true }
                        .disabled(history.clearing || history.state == .loading || !hosts.isUp(history.host))
                }
            }
            .confirmationDialog("Clear prompt history on \(history.host.name)?", isPresented: $confirmsClear, titleVisibility: .visible) {
                Button("Clear All Prompts", role: .destructive) { Task { await history.clear(query: query) } }
            } message: { Text("This clears all prompts on this machine, including those hidden by your search.") }
            .task(id: "\(history.host.id)|\(query)|\(hosts.isUp(history.host))|\(hosts.instanceID(of: history.host.id) ?? "unknown")") {
                do { try await Task.sleep(for: .milliseconds(200)) } catch { return }
                await history.load(query: query)
            }
        }
        .presentationDetents([.large])
    }

    @ViewBuilder private var notice: some View {
        switch history.state {
        case .loading: ProgressView("Loading prompts…")
        case .offline:
            Text("This machine is offline. Showing any previously loaded prompts.").foregroundStyle(.secondaryText)
            Button("Try Again") { Task { await history.load(query: query) } }
        case .unavailable: Text("This machine does not keep prompt history.").foregroundStyle(.secondaryText)
        case .failed:
            VStack(alignment: .leading, spacing: 8) {
                Text("Could not load or clear prompt history.").foregroundStyle(.secondaryText)
                if let message = history.message { Text(message).font(.caption).foregroundStyle(.secondaryText) }
                Button("Try Again") { Task { await history.load(query: query) } }
            }
        case .ready: EmptyView()
        }
    }

    private func modelTitle(_ entry: HistoryEntry) -> String {
        hosts.models[history.host.id]?.first { $0.name == entry.model }?.headline ?? entry.model
    }
}
