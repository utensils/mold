import MoldClient
import SwiftUI

/// The files a model needs, and which are here.
struct ComponentsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ModelStore.self) private var models
    let model: Model
    let host: MoldHost
    @State private var components: [ModelComponentStatus]?

    var body: some View {
        NavigationStack {
            List {
                if let components {
                    ForEach(components) { part in
                        AdaptiveRow {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(part.name)
                                Text(verbatim: part.kind).font(.caption.monospaced()).foregroundStyle(.secondaryText)
                            }
                        } value: {
                            Label(part.present ? String(localized: "Present") : String(localized: "Missing"),
                                  systemImage: part.present ? "checkmark.circle" : "exclamationmark.circle")
                        }
                        .accessibilityElement(children: .combine)
                    }
                } else {
                    ProgressView()
                }
            }
            .navigationTitle(model.headline)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .task { components = await models.components(of: model, on: host.id) }
        }
        .presentationDetents([.medium, .large])
    }
}

/// A gated model's terms, before the pull: what they say, where to read them
/// in full, and Accept and Download.
struct LicenceSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(ModelStore.self) private var models
    let pending: ModelStore.PendingLicense
    @State private var accepting = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(pending.mismatch
                         ? String(localized: "This machine pins different terms for \(pending.refusal.name). Read them before downloading.")
                         : String(localized: "\(pending.refusal.name) has its own licence. Accept it on this machine to download."))
                    Text(pending.refusal.summary).foregroundStyle(.secondaryText)
                    if let url = URL(string: pending.refusal.canonical) ?? URL(string: pending.refusal.url) {
                        Link(destination: url) { Label("Read the Full Licence", systemImage: "arrow.up.forward") }
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .safeAreaBar(edge: .bottom) {
                Button {
                    accepting = true
                    Task { await models.accept(pending); accepting = false; if models.pendingLicense == nil { dismiss() } }
                } label: {
                    Text("Accept and Download").frame(maxWidth: .infinity)
                }
                .prominentAction()
                .disabled(accepting)
                .padding(16)
                .accessibilityIdentifier("bottom-chrome")
            }
            .navigationTitle(pending.refusal.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { models.cancelLicense(); dismiss() } } }
        }
        .presentationDetents([.large])
    }
}
