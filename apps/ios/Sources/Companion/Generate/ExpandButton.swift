import MoldClient
import SwiftUI

/// Expand (⌘E): rewrites the prompt in place on the machine, the original kept
/// for Undo; its menu offers "Suggest Other Ways" as a list to pick from.
struct ExpandButton: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @State private var working = false
    @State private var suggestions: [String] = []
    @State private var original: String?

    var body: some View {
        Menu {
            Button { Task { await expand(variations: 1) } } label: {
                Label("Expand", systemImage: "text.badge.star")
            }
            Button { Task { await expand(variations: 4) } } label: {
                Label("Suggest Other Ways", systemImage: "text.bubble")
            }
            if let original {
                Button { generate.draft.prompt = original; self.original = nil } label: {
                    Label("Undo Expand", systemImage: "arrow.uturn.backward")
                }
            }
        } label: {
            if working { ProgressView() } else { Label("Expand", systemImage: "text.badge.star") }
        } primaryAction: {
            Task { await expand(variations: 1) }
        }
        .frame(minWidth: 44, minHeight: 44)
        .disabled(generate.draft.prompt.trimmingCharacters(in: .whitespaces).isEmpty || working
                  || generate.target.flatMap { hosts.capabilities[$0.id]?.expand } == nil)
        .keyboardShortcut("e", modifiers: .command)
        .sheet(isPresented: Binding(get: { !suggestions.isEmpty }, set: { if !$0 { suggestions = [] } })) {
            SuggestionsSheet(suggestions: suggestions) { chosen in
                original = generate.draft.prompt
                generate.draft.prompt = chosen
                suggestions = []
            }
        }
    }

    private func expand(variations: Int) async {
        guard let host = generate.target else { return }
        working = true
        defer { working = false }
        let snapshot = generate.draft
        let model = generate.modelName ?? ""
        let generation = RenderRequest.one(snapshot, model: model)
        var request = ExpandRequest(prompt: snapshot.prompt, modelFamily: generate.model?.family ?? "",
                                    variations: variations, task: ExpandTask.forRequest(family: generate.model?.family, request: generation))
        request.context = ExpandContext(request: generation)
        do {
            let answer = try await hosts.backend(for: host).expand(request)
            guard generate.draft == snapshot, generate.modelName == model, generate.target?.id == host.id else { return }
            if variations == 1, let first = answer.expanded.first {
                original = generate.draft.prompt
                generate.draft.prompt = first
            } else {
                suggestions = answer.expanded
            }
        } catch {
            hosts.report(host, doing: String(localized: "rewrite that prompt"), error)
        }
    }
}

private struct SuggestionsSheet: View {
    @Environment(\.dismiss) private var dismiss
    let suggestions: [String]
    let use: (String) -> Void

    var body: some View {
        NavigationStack {
            List(suggestions, id: \.self) { suggestion in
                Button { use(suggestion) } label: {
                    Text(suggestion).foregroundStyle(.primary).fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityHint("Use this prompt")
            }
            .navigationTitle("Other Ways to Say It")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
        .presentationDetents([.medium, .large])
    }
}
