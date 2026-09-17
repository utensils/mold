import MoldClient
import SwiftUI

/// Settings ▸ Accounts -- Hugging Face and Civitai tokens.
///
/// Credentials live on the MACHINE, not on this Mac (design fact 9): a
/// catalog token authenticates that machine to a provider, the same way a
/// machine key authenticates this Mac to that machine, and neither one is a
/// Keychain item. The pane is keyed on the same `selectedMachine` preference
/// every other machine-scoped view shares, and reads through the same
/// `CatalogStore` Discover already uses.
struct AccountsSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(CatalogStore.self) private var catalog
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""

    static let noCatalogSentence = "This machine doesn't browse a catalog."

    /// Whether the pane draws the two provider sections or the sentence
    /// instead -- pure, so `AccountsTests` can check the branch with no view,
    /// the same idiom `ModelScope.available` uses for Discover's own gate.
    static func showsProviders(capabilities: Capabilities?) -> Bool {
        capabilities?.canBrowseCatalog == true
    }

    private var selected: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        Form {
            if hosts.hosts.isEmpty {
                Text("Add a machine in Settings ▸ Machines first.")
                    .foregroundStyle(.secondary)
            } else {
                Picker("Machine", selection: selectedHostID) {
                    ForEach(hosts.hosts) { host in
                        Text(host.name).tag(MoldHost.ID?.some(host.id))
                    }
                }
                if let selected { machine(selected) }
            }
        }
        .formStyle(.grouped)
        .task(id: selected?.id) {
            guard let id = selected?.id else { return }
            await catalog.loadCredentials(on: id)
        }
    }

    @ViewBuilder private func machine(_ host: MoldHost) -> some View {
        if !Self.showsProviders(capabilities: hosts.capabilities(of: host)) {
            Text(Self.noCatalogSentence)
                .foregroundStyle(.secondary)
        } else if let status = catalog.credentials(on: host.id) {
            ProviderSection(
                provider: "hf", name: "Hugging Face", state: status.hf, host: host,
                footer: "A token lets this machine fetch gated and private models from Hugging Face.")
            ProviderSection(
                provider: "civitai", name: "Civitai", state: status.civitai, host: host,
                footer: "A token lets this machine browse Civitai without its rate limits.")
        } else {
            ProgressView().frame(maxWidth: .infinity)
        }
    }

    /// `selectedMachine` read and written the way `HostStore.machine(selected:)`
    /// expects, the same binding `ModelsPane`'s toolbar builds.
    private var selectedHostID: Binding<MoldHost.ID?> {
        Binding(
            get: { hosts.machine(selected: selectedMachine)?.id },
            set: { selectedMachine = $0?.uuidString ?? "" }
        )
    }
}
