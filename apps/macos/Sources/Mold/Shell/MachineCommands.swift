import MoldClient
import SwiftUI

/// The Machine menu.
///
/// `LibraryCommands.swift`'s own reason: the menu bar is what Help ▸ Search
/// searches and VoiceOver reads. Check Now is `HostStore.refresh(_:)`
/// (`+Reachability.swift:34-52`), which already asks one machine what it is
/// and has never had a menu item -- a different, narrower question from ⌘R's
/// own full pane refresh (`MachinesPane.swift`'s `refreshAction`), so it gets
/// its own shortcut rather than a second binding for one. Set as Default is
/// also in the Machines pane's toolbar overflow (`MachinesPane.swift`),
/// calling the same `HostStore.setDefault(_:)` -- one door either way.
struct MachineCommands: Commands {
    @FocusedValue(\.machineSelection) private var selection

    var body: some Commands {
        CommandMenu("Machine") {
            // The CARD's own list (`MachineCardActions`), so a right click on
            // a machine and this menu can never mean different things. Off no
            // machine it is present and inert, the rule `Refresh` set: Help ▸
            // Search finds "Set as Default" from every pane. The chords stay
            // here, because a `RowAction` carries none and a contextual menu
            // shows none.
            RowActionMenu(actions: selection?.offered ?? MachineCardActions.unavailable(),
                          perform: { selection?.perform($0) },
                          shortcut: Self.shortcut)
            if let selection, !selection.machines.isEmpty {
                Divider()
                ForEach(selection.machines) { machine in
                    Button {
                        selection.choose(machine.id)
                    } label: {
                        if machine.id == selection.defaultID {
                            Label(machine.name, systemImage: "checkmark")
                        } else {
                            Text(machine.name)
                        }
                    }
                }
            }
        }
    }

    /// ⇧⌘R is one machine asked what it is -- a narrower question than ⌘R's
    /// whole-pane refresh, which is why it is its own chord and not a second
    /// binding for one. ⌘⌫ is the app's own "this row leaves" chord, the same
    /// one the Library's Move to Trash and the Queue's Cancel Job carry; only
    /// one of the three is ever focus-eligible.
    ///
    /// Open takes NO chord: Return opens the focused card, and binding it here
    /// as well would take Return away from every default button in the app.
    static func shortcut(_ kind: MachineCardActions.Kind) -> KeyboardShortcut? {
        switch kind {
        case .checkNow: KeyboardShortcut("r", modifiers: [.command, .shift])
        case .remove: KeyboardShortcut(.delete, modifiers: .command)
        case .open, .setDefault, .copyAddress, .edit: nil
        }
    }
}

/// What the Machines pane's current machine can do, and how -- resolved once
/// per body pass, the "items and closures together" shape `ModelSelection`
/// and `QueueSelection` already take. `nil` off no machine at all (the pane's
/// own empty state), which is an empty menu.
struct MachineSelection: Equatable {
    let machines: [MoldHost]
    /// The machine the items act on: the one whose page is open, else the
    /// card the keyboard is on. `nil` on the overview with nothing picked,
    /// which is what makes every item inert rather than acting on a machine
    /// nobody pointed at.
    let selected: MoldHost.ID?
    /// The default's id, or `nil` when nothing has been chosen -- no row in
    /// the list below carries the checkmark.
    let defaultID: MoldHost.ID?
    /// The card's own items (`MachineCardActions`), already resolved for the
    /// selected machine -- so Set as Default is absent on the default machine
    /// and Remove… is absent on This Mac, here as well as on the card.
    let offered: [RowAction<MachineCardActions.Kind>]
    /// Sets the default to whichever machine's row this is -- the same call
    /// `perform(.setDefault)` makes for `selected`, parameterized for the list.
    let choose: (MoldHost.ID) -> Void
    let perform: (MachineCardActions.Kind) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.machines == rhs.machines && lhs.selected == rhs.selected
            && lhs.defaultID == rhs.defaultID && lhs.offered == rhs.offered
    }

    /// One row per machine, in list order, with whether it carries the
    /// checkmark -- pulled out of `body` so a test can pin exactly what the
    /// menu offers without rendering it, `QueueSelection.offeredTitles`'s own
    /// idiom (`QueueCommands.swift`).
    struct Row: Equatable {
        let name: String
        let isDefault: Bool
    }

    var rows: [Row] {
        machines.map { Row(name: $0.name, isDefault: $0.id == defaultID) }
    }
}

struct MachineSelectionKey: FocusedValueKey {
    typealias Value = MachineSelection
}

extension FocusedValues {
    var machineSelection: MachineSelection? {
        get { self[MachineSelectionKey.self] }
        set { self[MachineSelectionKey.self] = newValue }
    }
}
