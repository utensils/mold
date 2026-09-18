import Foundation
import MoldClient
import Testing

@testable import Mold

/// The sidebar had TWO ways into the library: a top-level Library row beside
/// Generate, Queue, Models and Machines, and directly under it a Library
/// SECTION whose first row, All Prints, opened the same thing. Photos, Music
/// and Mail have one Library group whose rows ARE the destinations.
///
/// `SidebarRows` is that list as a value, so what the sidebar draws and which
/// single row is highlighted can be asserted with no view rendered -- the same
/// way `MenuBarTests` pins the two menu resolvers.
@MainActor
struct SidebarRowsTests {
    private func scratch() -> UserDefaults {
        let name = "io.utensils.mold.native.tests.sidebarrows.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return defaults
    }

    private func machineID() -> MoldHost.ID {
        MoldHost(name: "plato", baseURL: URL(string: "http://plato")!).id
    }

    /// **Fails today**: the first section is `Destination.allCases`, so
    /// Library is a row of its own above the section that already lists it.
    @Test func theLibraryIsNotATopLevelRow() {
        #expect(!SidebarRows.destinations.contains(.library))
        #expect(SidebarRows.destinations == [.generate, .queue, .models, .machines])
        #expect(SidebarRows.destinations.count == Destination.allCases.count - 1)
    }

    /// The single-highlight rule, over every destination there is: the
    /// highlighted row is always one the sidebar actually draws, and the
    /// library's is never a top-level row -- it is the shelf you are on.
    ///
    /// **Fails today**: `SidebarRows` does not exist; the sidebar derives this
    /// inside a `Binding` no test can reach.
    @Test func exactlyOneDrawnRowIsHighlighted() {
        let machine = machineID()
        for destination in Destination.allCases {
            let row = SidebarRows.selected(
                destination: destination, scope: .favorites, machine: machine)
            let expected: SidebarRow = switch destination {
            case .library: .shelf(.favorites)
            case .machines: .machine(machine)
            default: .destination(destination)
            }
            #expect(row == expected)
            #expect(row != .destination(.library))
            if case let .destination(item) = row {
                #expect(SidebarRows.destinations.contains(item))
            }
        }
    }

    /// When the library is showing, the highlighted row is its scope --
    /// including a collection, which is a row of the same section.
    @Test func theHighlightedLibraryRowIsTheScopeOnShow() {
        #expect(SidebarRows.selected(destination: .library, scope: .all, machine: nil)
            == .shelf(.all))
        #expect(SidebarRows.selected(destination: .library, scope: .trash, machine: nil)
            == .shelf(.trash))
        #expect(SidebarRows.selected(destination: .library,
                                     scope: .collection(slug: "keepers"), machine: nil)
            == .shelf(.collection(slug: "keepers")))
    }

    /// Machines keeps its two-level answer: the machine you picked, or the
    /// section's own row while nothing is picked.
    @Test func aMachineRowIsHighlightedOverItsSection() {
        let machine = machineID()
        #expect(SidebarRows.selected(destination: .machines, scope: .all, machine: machine)
            == .machine(machine))
        #expect(SidebarRows.selected(destination: .machines, scope: .all, machine: nil)
            == .destination(.machines))
    }

    /// Picking a shelf row IS entering the library at that shelf -- choosing
    /// what to look at and choosing to look are one act.
    @Test func pickingAShelfRowEntersTheLibraryAtThatScope() {
        #expect(SidebarRows.pick(.shelf(.favorites))
            == SidebarPick(destination: .library, scope: .favorites, machine: nil))
        #expect(SidebarRows.pick(.destination(.queue))
            == SidebarPick(destination: .queue, scope: nil, machine: nil))
        let machine = machineID()
        #expect(SidebarRows.pick(.machine(machine))
            == SidebarPick(destination: .machines, scope: nil, machine: machine))
        #expect(SidebarRows.pick(nil) == nil)
    }

    /// ⌘1–⌘5 are `Destination.allCases` in order, one per index
    /// (`MoldCommands.swift`'s View group). Dropping Library from the SIDEBAR
    /// must not move them: ⌘2 is still Library.
    @Test func theDestinationShortcutsAreUnmoved() {
        #expect(Destination.allCases == [.generate, .library, .queue, .models, .machines])
        #expect(Destination.allCases.firstIndex(of: .library) == 1)
    }

    /// View ▸ Library (⌘2) sets `destination = .library` and says nothing
    /// about the scope, so the library opens where you left it -- what
    /// `destination` itself and the picked machine already do. On a library
    /// nobody has moved, that is All Prints.
    @Test func viewLibraryOpensTheShelfYouLeft() {
        let defaults = scratch()
        let navigation = LibraryNavigation(defaults: defaults)
        #expect(SidebarRows.selected(destination: .library, scope: navigation.scope,
                                     machine: nil) == .shelf(.all))

        navigation.scope = .favorites
        let reopened = LibraryNavigation(defaults: defaults)
        #expect(SidebarRows.selected(destination: .library, scope: reopened.scope,
                                     machine: nil) == .shelf(.favorites))
    }

}
