import MoldClient

/// One row of the sidebar, whichever of its three groups it belongs to.
enum SidebarRow: Hashable {
    case destination(Destination)
    case shelf(LibraryScope)
    case machine(MoldHost.ID)
}

/// What picking a row means: where the window goes, and what it takes with it.
struct SidebarPick: Equatable {
    let destination: Destination
    /// The shelf a library row names. `nil` leaves the library where it was.
    let scope: LibraryScope?
    /// The machine a machine row names. `nil` leaves the picked one alone.
    let machine: MoldHost.ID?
}

/// What the sidebar lists, and which ONE row of it is highlighted.
///
/// The library is not a top-level row. Its section below lists All Prints,
/// Favourites, every collection and Recently Deleted, and each of those IS the
/// way in -- the one Library group Photos, Music and Mail all have. A row
/// above them that opened what its first row opens is a duplicate, and while
/// it existed the window had two ideas of where it was.
///
/// Pure, so the rule can be asserted with no view rendered: `selected` returns
/// ONE row for any state the window can be in, which is what makes "exactly
/// one row is highlighted, across both groups" true by construction rather
/// than by two bindings agreeing.
enum SidebarRows {
    /// The destinations that are rows of their own, in menu order.
    static let destinations: [Destination] = Destination.allCases.filter { $0 != .library }

    /// The highlighted row, from where the window is. On the library that is
    /// the shelf showing; on Machines it is the machine picked, falling back
    /// to the section's own row while nothing is picked.
    static func selected(
        destination: Destination, scope: LibraryScope, machine: MoldHost.ID?
    ) -> SidebarRow {
        switch destination {
        case .library: .shelf(scope)
        case .machines: machine.map(SidebarRow.machine) ?? .destination(.machines)
        default: .destination(destination)
        }
    }

    /// What picking a row does. Picking a shelf or a machine also moves to its
    /// pane, because choosing what to look at and choosing to look are the
    /// same act -- making them two clicks would be a bug people report.
    static func pick(_ row: SidebarRow?) -> SidebarPick? {
        switch row {
        case let .destination(item): SidebarPick(destination: item, scope: nil, machine: nil)
        case let .shelf(scope): SidebarPick(destination: .library, scope: scope, machine: nil)
        case let .machine(id): SidebarPick(destination: .machines, scope: nil, machine: id)
        case nil: nil
        }
    }
}
