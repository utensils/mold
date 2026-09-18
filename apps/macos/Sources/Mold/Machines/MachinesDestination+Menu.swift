import MoldClient
import SwiftUI

// What a machine's card offers and what each item does. Split from
// `MachinesDestination` for size; every one of these is the SAME call the
// other surface makes -- Settings ▸ Machines' own row menu and the Machine
// menu in the menu bar both end up in `HostStore`, and nothing here is a
// second implementation of any of it.
extension MachinesDestination {

    /// The one door. The card's right-click menu, its inline controls and the
    /// Machine menu all arrive here.
    func perform(_ id: MoldHost.ID, _ kind: MachineCardActions.Kind) {
        guard let host = hosts.host(id) else { return }
        switch kind {
        // Opening IS selecting: the stack's path is `selectedMachine`, so the
        // sidebar's machine row lights up with the page.
        case .open: selectedMachine = id.uuidString
        case .checkNow: Task { await hosts.refresh(host) }
        case .setDefault: hosts.setDefault(host)
        case .copyAddress: Clipboard.put(HostAddress.displayString(for: host.baseURL))
        case .edit: edit(host)
        case .remove: remove(host)
        }
    }

    /// This Mac's engine is a property of this launch rather than a saved row:
    /// its address is whatever port the engine bound, so there is no key to
    /// edit and removing it would only make it come back. The card does not
    /// offer either item -- this is the belt for the menu bar's braces.
    private func isManaged(_ host: MoldHost) -> Bool { host.id != MoldEngine.localHostID }

    private func edit(_ host: MoldHost) {
        guard isManaged(host) else { return }
        editing = host
    }

    /// Asks first, always, with the SAME sentence Settings asks: the removal
    /// takes the machine's stored key with it and there is no undo on either
    /// side (`MachineRemoval.swift`, review 05-H6).
    private func remove(_ host: MoldHost) {
        guard isManaged(host) else { return }
        pendingRemoval = MachineRemoval.destruction(of: host) {
            // `HostStore.remove` is the one door: it forgets the machine, its
            // key, its watchers and its default-ness. Re-deciding any of that
            // here would be a second answer.
            hosts.remove(host)
            if selectedMachine == host.id.uuidString { selectedMachine = "" }
            if focusedCard == host.id { focusedCard = nil }
        }
    }

    /// What the Machine menu offers, resolved once per body pass -- the
    /// "items and closures together" shape `ModelSelection` and
    /// `QueueSelection` already take.
    ///
    /// The items are the CARD's items: one declaration, so a right click and
    /// the menu bar can never mean different things. Off no machine at all
    /// they are present and inert, which is the rule the menu already
    /// followed.
    var selection: MachineSelection {
        let target = target
        return MachineSelection(
            machines: hosts.hosts,
            selected: target?.id,
            defaultID: hosts.defaultMachine,
            offered: target.map {
                MachineCardActions.offered(isThisMac: !isManaged($0),
                                           isDefault: hosts.defaultMachine == $0.id)
            } ?? MachineCardActions.unavailable(),
            choose: { id in hosts.host(id).map(hosts.setDefault) },
            perform: { kind in target.map { perform($0.id, kind) } }
        )
    }
}
