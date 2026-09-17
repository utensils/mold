import MoldClient
import SwiftUI

/// One machine in the sidebar.
///
/// Selectable, and it opens that machine's page. It used to be inert, on the
/// reasoning that a machine is something the library is filtered BY rather
/// than a place to go -- which was half right, so the menu still files the
/// library by it. What the reasoning missed is that a machine is also a thing
/// with four GPUs, a memory budget and a switch per card, and none of that
/// fits in a search chip.
struct MachineRow: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryNavigation.self) private var navigation
    let host: MoldHost
    let reachability: HostStore.Reachability
    @Binding var destination: Destination

    var body: some View {
        HStack(spacing: 8) {
            HostStatusDot(reachability: reachability)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                if let detail = reachability.summary {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .help(HostAddress.displayString(for: host.baseURL))
        // Declared once in `SidebarMachineActions`, which is also where the
        // Machine menu's two items get their words -- so a right click and
        // ⇧⌘R can never mean different things.
        .rowActionMenu(SidebarMachineActions.offered(), perform: perform)
    }

    private func perform(_ kind: SidebarMachineActions.Kind) {
        switch kind {
        case .checkNow:
            Task { await hosts.refresh(host) }
        case .setDefault:
            hosts.setDefault(host)
        case .showInLibrary:
            navigation.query.tokens = [.machine(id: host.id, name: host.name)]
            destination = .library
        }
    }
}
