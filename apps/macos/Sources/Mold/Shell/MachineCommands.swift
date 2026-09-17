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
            // Present and disabled off no machine, the rule `Refresh` set:
            // Help ▸ Search finds "Set as Default" from every pane.
            // The words come from `SidebarMachineActions`, which is also what
            // a machine row's right-click menu draws -- one spelling, two
            // surfaces. The shortcut and the disabled-off-nothing rule stay
            // here, because a `RowAction` carries neither.
            Button(SidebarMachineActions.checkNow) { selection?.check() }
                .keyboardShortcut("r", modifiers: [.command, .shift])
                .disabled(selection == nil)
            Button(SidebarMachineActions.setAsDefault) { selection?.setDefault() }
                .disabled(selection == nil)
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
}

/// What the Machines pane's current machine can do, and how -- resolved once
/// per body pass, the "items and closures together" shape `ModelSelection`
/// and `QueueSelection` already take. `nil` off no machine at all (the pane's
/// own empty state), which is an empty menu.
struct MachineSelection: Equatable {
    let machines: [MoldHost]
    let selected: MoldHost.ID
    /// The default's id, or `nil` when nothing has been chosen -- no row in
    /// the list below carries the checkmark.
    let defaultID: MoldHost.ID?
    let check: () -> Void
    /// Sets the default to whichever machine's row this is -- the same call
    /// `setDefault` makes for `selected`, parameterized for the list.
    let choose: (MoldHost.ID) -> Void
    let setDefault: () -> Void

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.machines == rhs.machines && lhs.selected == rhs.selected && lhs.defaultID == rhs.defaultID
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
