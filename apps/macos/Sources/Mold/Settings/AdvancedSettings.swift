import MoldClient
import SwiftUI

/// Settings ▸ Advanced -- every row `GET /api/config` will give up, rendered
/// from each value's own JSON type rather than a client-side schema (design
/// decision 3). Advanced shows every row, including the ones a curated pane
/// owns: a table that filtered would be a second opinion about which keys
/// matter, and the promise "this is everything the machine has" is what
/// makes the curated panes trustworthy rather than a frozen snapshot.
struct AdvancedSettings: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ConfigStore.self) private var store
    @AppStorage("selectedMachine", store: AppStorageSuite.defaults) private var selectedMachine = ""
    @State private var query = ""
    @State private var selection: String?

    /// The three whole-pane states, in the order they are checked -- shared
    /// with Generation and Expansion (S4a) as `SettingsPaneState` /
    /// `SettingsPane.resolve`, since every machine-scoped Settings tab needs
    /// exactly the same switch. Kept as a forwarder so the existing call
    /// sites and tests (`AdvancedSettings.resolve(...) == .unavailable`)
    /// need no changes.
    typealias PaneState = SettingsPaneState

    static func resolve(hosts: HostStore, store: ConfigStore, machine: MoldHost.ID?) -> PaneState {
        SettingsPane.resolve(hosts: hosts, store: store, machine: machine)
    }

    private var machine: MoldHost? { hosts.machine(selected: selectedMachine) }

    var body: some View {
        content
            .task(id: machine?.id) {
                guard let id = machine?.id, !store.hasLoaded(on: id) else { return }
                await store.refresh(on: id)
                await store.refreshProfiles(on: id)
            }
    }

    @ViewBuilder private var content: some View {
        switch Self.resolve(hosts: hosts, store: store, machine: machine?.id) {
        case .noMachines:
            Text("Add a machine in Settings ▸ Machines first.")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .unavailable:
            if let machine {
                Text("""
                     \(machine.name)'s settings database is switched off, so \
                     there is nothing here to read or change.
                     """)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        case .loading:
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        case .loaded:
            if let machine { loaded(machine) }
        }
    }

    @ViewBuilder private func loaded(_ machine: MoldHost) -> some View {
        let listing = store.byHost[machine.id]
        let refusals = (store.refusals[machine.id] ?? [:]).mapValues(\.sentence)
        let rows = Self.rows(listing, query: query, refusals: refusals)
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                SettingsMachineHeader(hosts: hosts, selectedMachine: $selectedMachine)
                Spacer()
                // A plain field, not `.searchable`: a Settings window has no
                // navigation toolbar, so `.searchable` surfaced as a toolbar
                // item that read like a tenth tab (M7 UAT).
                TextField("Search settings", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 220)
                    .accessibilityLabel("Search settings")
            }
            ProfileHeader(profiles: store.profiles[machine.id])
            table(rows, machine: machine)
            Text(Self.subtitle(showing: rows.count, total: listing?.entries.count ?? 0))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
    }

    private func table(_ rows: [Row], machine: MoldHost) -> some View {
        Table(of: Row.self, selection: $selection) {
            TableColumn("Key") { row in keyCell(row) }
            TableColumn("Value") { row in
                ConfigValueField(entry: row.entry) { scalar in
                    await store.set(row.entry.key, to: scalar, on: machine.id)
                }
            }
            TableColumn("Source") { row in SourceBadge(entry: row.entry) }
            TableColumn("") { row in resetButton(row, machine: machine) }
                .width(44)
        } rows: {
            ForEach(rows) { row in TableRow(row) }
        }
        .alternatingRowBackgrounds(.disabled)
    }

    @ViewBuilder private func keyCell(_ row: Row) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(row.entry.key).textSelection(.enabled)
            if let refusal = row.refusal {
                Text(refusal)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .help(refusal)
            }
        }
    }

    @ViewBuilder private func resetButton(_ row: Row, machine: MoldHost) -> some View {
        if row.entry.canReset {
            Button("Reset", systemImage: "arrow.uturn.backward") {
                Task { await store.reset(row.entry.key, on: machine.id) }
            }
            .buttonStyle(.borderless)
            .labelStyle(.iconOnly)
            .help("Reset \(row.entry.key) to its fallback")
            .accessibilityLabel("Reset \(row.entry.key)")
        }
    }
}
