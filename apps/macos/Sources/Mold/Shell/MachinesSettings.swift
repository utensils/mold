import MoldClient
import SwiftUI

/// Adding, editing and removing the servers the app talks to.
struct MachinesSettings: View {
    @Environment(HostStore.self) private var hosts
    @State private var selection: MoldHost.ID?
    @State private var isAdding = false
    @State private var editingID: MoldHost.ID?
    @State private var pendingRemoval: Destruction?

    var body: some View {
        VStack(spacing: 0) {
            if hosts.hosts.isEmpty {
                empty
            } else {
                list
            }
            Divider()
            footer
        }
        .sheet(isPresented: $isAdding) {
            HostEditor { hosts.add(name: $0, url: $1, apiKey: $2) }
        }
        .sheet(item: editing) { host in
            HostEditor(host: host) { name, url, key in
                hosts.update(MoldHost(id: host.id, name: name, baseURL: url, apiKey: key))
            }
        }
        .destructionDialog($pendingRemoval)
        .task { openEditorIfRequested() }
    }

    /// `MOLD_NATIVE_DESTINATION=add-machine` opens an empty host sheet, and
    /// `=edit-machine` opens the first machine in the list. Both exist so a
    /// UAT run can photograph the sheet without a script driving the mouse
    /// across someone's desktop to reach it.
    static let addOnLaunch = "add-machine"
    static let editOnLaunch = "edit-machine"

    private func openEditorIfRequested() {
        switch NativeUAT.destination.value() {
        case Self.addOnLaunch: isAdding = true
        case Self.editOnLaunch: hosts.hosts.first.map(edit)
        default: break
        }
    }

    private var list: some View {
        List(selection: $selection) {
            ForEach(hosts.hosts) { host in
                HostSettingsRow(host: host, reachability: hosts.reachability(of: host))
                    .tag(host.id)
                    .contentShape(.rect)
                    // A simultaneous gesture, so opening the editor does not
                    // cost the row its ordinary click-to-select.
                    .simultaneousGesture(TapGesture(count: 2).onEnded { edit(host) })
                    .rowActionMenu(
                        MachineRowActions.offered(isManaged: isManaged(host),
                                                  isDefault: hosts.defaultMachine == host.id)
                    ) { perform($0, on: host) }
            }
        }
        .alternatingRowBackgrounds()
    }

    private var empty: some View {
        ContentUnavailableView {
            Label("No machines yet", systemImage: "server.rack")
        } description: {
            Text("Add a machine running `mold serve`. Its name or IP is enough.")
        } actions: {
            Button("Add a Machine…") { isAdding = true }
        }
        .frame(maxHeight: .infinity)
    }

    private var footer: some View {
        HStack(spacing: 8) {
            Button("Add a machine", systemImage: "plus") { isAdding = true }
            // The ellipsis is the promise the dialog keeps: this button asks.
            Button("Remove the selected machine…", systemImage: "minus") { removeSelected() }
                .disabled(selected.map(isManaged) != true)
            Button("Edit the selected machine", systemImage: "pencil") {
                if let selected { edit(selected) }
            }
            .disabled(selected.map(isManaged) != true)
            Spacer()
            Button("Check All") { Task { await hosts.refreshAll() } }
                .disabled(hosts.hosts.isEmpty)
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.borderless)
        .padding(8)
    }

    // MARK: - Actions

    private var selected: MoldHost? { hosts.hosts.first { $0.id == selection } }

    /// This Mac's engine is a property of the running app, not a saved row:
    /// its address belongs to whatever port the engine bound, and removing it
    /// from a list it is not stored in would only make it come back.
    private func isManaged(_ host: MoldHost) -> Bool { host.id != MoldEngine.localHostID }

    private var editing: Binding<MoldHost?> {
        Binding(
            get: { hosts.hosts.first { $0.id == editingID } },
            set: { editingID = $0?.id }
        )
    }

    private func edit(_ host: MoldHost) {
        guard isManaged(host) else { return }
        selection = host.id
        editingID = host.id
    }

    private func removeSelected() {
        guard let selected else { return }
        remove(selected)
    }

    /// The right-click menu's one door. Every item ends up in the same call
    /// the inline control makes -- `edit` and `remove` are literally the
    /// footer's own buttons, so the two surfaces cannot drift apart.
    private func perform(_ kind: MachineRowActions.Kind, on host: MoldHost) {
        switch kind {
        case .edit: edit(host)
        case .checkNow: Task { await hosts.refresh(host) }
        case .copyAddress: Clipboard.put(HostAddress.displayString(for: host.baseURL))
        case .setDefault: hosts.setDefault(host)
        case .remove: remove(host)
        }
    }

    /// Asks first, always. The removal takes the machine's stored key with it
    /// and there is no undo on either side (review 05-H6).
    private func remove(_ host: MoldHost) {
        guard isManaged(host) else { return }
        pendingRemoval = MachineRemoval.destruction(of: host) {
            hosts.remove(host)
            if selection == host.id { selection = nil }
        }
    }
}
