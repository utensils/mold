import MoldClient
import SwiftUI

/// One provider row in `AccountsSettings`: its current state, a field to
/// replace it, Save, and Clear where there is something on the machine to
/// remove.
struct ProviderSection: View {
    let provider: String
    let name: String
    let state: CatalogCredentialState
    let host: MoldHost
    let footer: String

    @Environment(CatalogStore.self) private var catalog
    @State private var token = ""
    @State private var isSaving = false

    private var row: AccountsRow { .resolve(state) }

    var body: some View {
        Section {
            LabeledContent(name, value: row.summary)
            HStack {
                SecureField("Token…", text: $token)
                Button("Save") { Task { await save() } }
                    .disabled(!AccountsRow.canSave(token: token) || isSaving)
                if row.offersClear {
                    Button("Clear") { Task { await clear() } }
                        .disabled(isSaving)
                }
            }
        } footer: {
            Text(footer).font(.caption).foregroundStyle(.secondary)
        }
        // The row's own Clear, reachable from the row itself -- the same call,
        // offered on the same condition, never a second opinion about when a
        // token can be cleared.
        .contextMenu {
            RowActionMenu(actions: menu) { _ in Task { await clear() } }
        }
    }

    private var menu: [RowAction<String>] {
        guard row.offersClear else { return [] }
        return [RowAction(kind: "clear", title: "Clear \(name) Token", isDestructive: true)]
    }

    /// The field empties on success -- the machine never hands the token
    /// back, only the mask, and that mask arrives through `state` on the
    /// next redraw because it rode the save's own answer.
    private func save() async {
        isSaving = true
        defer { isSaving = false }
        if await catalog.saveCredential(provider, token: token, on: host.id) {
            token = ""
        }
    }

    private func clear() async {
        isSaving = true
        defer { isSaving = false }
        await catalog.clearCredential(provider, on: host.id)
    }
}
