import Foundation
import MoldClient
import Testing

@testable import Mold

/// The fleet overview's menu, its keyboard, and the one preference that says
/// whether a machine is open on it.
///
/// **Fails today**: the Machines destination is one machine's page, so none of
/// these types exist -- there is no card menu, no grid to move around, and the
/// only way to a machine is the sidebar.
@MainActor
struct MachineFleetTests {
    // MARK: - The menu

    /// The same list feeds the card's menu, its inline controls and the
    /// Machine menu in the menu bar. Pinned in draw order, through the one
    /// renderer, so the divider and the destructive-last rule are the ones
    /// `RowAction` decides rather than this surface's own.
    @Test func aCardOffersOpenCheckDefaultCopyEditAndRemove() {
        let offered = RowAction.rendered(
            MachineCardActions.offered(isThisMac: false, isDefault: false))

        #expect(offered.map(\.kind)
            == [.open, .checkNow, .setDefault, .copyAddress, .edit, nil, .remove])
        #expect(offered.map(\.title) == [
            "Open", "Check Now", "Set as Default", "Copy Address", "Edit…", "", "Remove…",
        ])
        #expect(offered.last?.isDestructive == true)
        // The two words the sidebar's row already spells, read from there
        // rather than typed again.
        #expect(offered[1].title == SidebarMachineActions.checkNow)
        #expect(offered[2].title == SidebarMachineActions.setAsDefault)
    }

    /// Absent, not inert: the machine that IS the default has nothing to set.
    @Test func theDefaultMachineIsNotOfferedSetAsDefault() {
        let offered = MachineCardActions.offered(isThisMac: false, isDefault: true)
        #expect(!offered.contains { $0.kind == .setDefault })
        #expect(offered.contains { $0.kind == .remove })
    }

    /// This Mac's engine is a property of this launch, not a saved row: there
    /// is no key to edit and removing it would only make it come back.
    @Test func thisMacsCardOffersOnlyWhatAppliesToIt() {
        let offered = MachineCardActions.offered(isThisMac: true, isDefault: false)
        #expect(offered.map(\.kind) == [.open, .checkNow, .setDefault, .copyAddress])
        #expect(RowAction.offersMenu(offered))
    }

    /// Off no machine the Machine menu still draws every item, inert -- the
    /// rule it already followed, so Help ▸ Search finds "Set as Default" from
    /// every pane.
    @Test func theMenuBarKeepsItsItemsWhenNoMachineIsPicked() {
        let unavailable = MachineCardActions.unavailable()
        #expect(unavailable.allSatisfy { $0.isDisabled })
        #expect(unavailable.map(\.title)
            == MachineCardActions.offered(isThisMac: false, isDefault: false).map(\.title))
        // Disabled commands are still commands: the menu is not empty.
        #expect(RowAction.offersMenu(unavailable))
    }

    /// The overview's Nearby list and a machine's own page draw ONE
    /// declaration, so an "Add…" that promises a sheet promises it on both.
    @Test func aFoundMachineOffersTheSameItemWhereverItIsDrawn() {
        let open = FakeFixtures.discoveryPeer("zeus", url: "http://zeus.local:7680")
        let keyed = FakeFixtures.discoveryPeer("hera", url: "http://hera.local:7680",
                                               authRequired: true)
        let mine = FakeFixtures.discoveryPeer("me", url: "http://me.local:7680",
                                              isThisMachine: true)
        func offered(_ peer: DiscoveryPeer) -> [String] {
            PeerAction.resolve(peer, known: { _ in false }, knownInstance: { _ in false })
                .offered(for: peer).map(\.title)
        }
        #expect(offered(open) == ["Add"])
        #expect(offered(keyed) == ["Add…"])
        // A peer already in the list offers nothing, so it gets no menu.
        #expect(offered(mine) == [])
    }

    // MARK: - The grid and its keyboard

    @Test func theGridIsOneColumnUntilTwoCardsFit() {
        #expect(MachineGrid.columnCount(for: 320) == 1)
        #expect(MachineGrid.columnCount(for: 620) == 2)
        #expect(MachineGrid.columnCount(for: 1240) == 4)
        // A window narrower than one card still draws the column it has.
        #expect(MachineGrid.columnCount(for: 10) == 1)
    }

    @Test func arrowKeysMoveTheFocusAndTheFirstPressPicksSomething() {
        let cards = fleet(5)
        #expect(MachineGrid.move(.right, from: nil, in: cards, columns: 2) == cards[0].id)
        #expect(MachineGrid.move(.right, from: cards[0].id, in: cards, columns: 2) == cards[1].id)
        #expect(MachineGrid.move(.down, from: cards[0].id, in: cards, columns: 2) == cards[2].id)
        #expect(MachineGrid.move(.up, from: cards[2].id, in: cards, columns: 2) == cards[0].id)
        #expect(MachineGrid.move(.left, from: cards[1].id, in: cards, columns: 2) == cards[0].id)
    }

    /// No wrapping, and no sliding sideways: an edge holds.
    @Test func theEdgesOfTheGridHold() {
        let cards = fleet(6)
        #expect(MachineGrid.move(.left, from: cards[0].id, in: cards, columns: 2) == cards[0].id)
        #expect(MachineGrid.move(.up, from: cards[1].id, in: cards, columns: 2) == cards[1].id)
        #expect(MachineGrid.move(.right, from: cards[5].id, in: cards, columns: 2) == cards[5].id)
        // The last row, where straight down is off the end.
        #expect(MachineGrid.move(.down, from: cards[4].id, in: cards, columns: 2) == cards[4].id)
        // A short last row: down lands on the last card, not on nothing.
        let short = fleet(5)
        #expect(MachineGrid.move(.down, from: short[3].id, in: short, columns: 2) == short[4].id)
        #expect(MachineGrid.move(.right, from: nil, in: [], columns: 2) == nil)
    }

    // MARK: - Where the destination opens

    /// The overview and a machine's page are the same destination, one push
    /// apart, and the sidebar's own preference is the whole path.
    @Test func openingACardIsTheSameSelectionTheSidebarWrites() {
        let workstation = host("workstation")
        let hosts = [workstation, host("hal9000")]

        #expect(MachineNavigation.path(selected: "", in: hosts) == [])
        #expect(MachineNavigation.path(selected: workstation.id.uuidString, in: hosts) == [workstation.id])
        // A machine that has been removed cannot be open.
        #expect(MachineNavigation.path(selected: UUID().uuidString, in: hosts) == [])
        #expect(MachineNavigation.path(selected: "not-a-uuid", in: hosts) == [])

        #expect(MachineNavigation.stored(path: [workstation.id]) == workstation.id.uuidString)
        // Back empties it, which is what deselects the sidebar's machine row.
        #expect(MachineNavigation.stored(path: []) == "")
    }

    @Test func theUATHookLandsOnTheOverviewOrOnOneMachinesPage() {
        let workstation = host("workstation")
        let hosts = [workstation, host("hal9000")]

        #expect(MachineLaunch.resolve(destination: "machines", machine: nil, in: hosts)
            == .overview)
        #expect(MachineLaunch.resolve(destination: "machines", machine: "WORKSTATION", in: hosts)
            == .machine(workstation.id))
        // A name matching nothing is the overview, never some other machine.
        #expect(MachineLaunch.resolve(destination: "machines", machine: "zeus", in: hosts)
            == .overview)
        #expect(MachineLaunch.resolve(destination: "machines", machine: "  ", in: hosts)
            == .overview)
        // Another destination -- or none -- says nothing about this one, so
        // what was open stays open.
        #expect(MachineLaunch.resolve(destination: "library", machine: "workstation", in: hosts)
            == .unchanged)
        #expect(MachineLaunch.resolve(destination: nil, machine: nil, in: hosts) == .unchanged)
    }

    // MARK: - Fixtures

    private func host(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name).local:7680")!)
    }

    /// Cards in the order the grid draws them, so an index in a test is a
    /// position on screen.
    private func fleet(_ count: Int) -> [MachineCard] {
        (0..<count).map { index in
            MachineCard(host: host("m\(index)"), reachability: .unknown, isDefault: false,
                        devices: [], snapshot: nil, live: nil, models: nil)
        }
    }
}
