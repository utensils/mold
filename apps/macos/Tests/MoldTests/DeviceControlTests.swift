import Foundation
import MoldClient
import Testing

@testable import Mold

/// What a machine will let you do to one of its cards, and what the pane draws
/// when it will let you do nothing.
///
/// `DeviceControl.resolve` is the whole of the decision and it is pure, so
/// every rule is pinned here rather than inferred from a screenshot. The one
/// that matters most is the first: TWO flags, never one.
@MainActor
struct DeviceControlTests {
    /// Only the three fields the decision reads. `devices.available` rides
    /// along because `DeviceCapabilities` requires it.
    private func capabilities(lifecycle: Bool = false,
                              restartEnable: Bool = false,
                              v2Authoritative: Bool = false) -> Capabilities {
        let json = """
        {"devices": {"available": true, "lifecycle": \(lifecycle),
                     "restart_enable": \(restartEnable)},
         "dispatch": {"v2_authoritative": \(v2Authoritative)}}
        """
        return try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    private func card(_ adminState: String = "enabled", on: Bool = true) -> DeviceInfo {
        FakeFixtures.deviceInfo("cuda:0", ordinal: 0, adminState: adminState, desiredEnabled: on)
    }

    @Test func aHostWithLifecycleButWithoutAnAuthoritativeSchedulerOffersNoSwitch() {
        let caps = capabilities(lifecycle: true, v2Authoritative: false)
        #expect(DeviceControl.resolve(card(), on: caps, isChanging: false) == .readOnly)
    }

    @Test func aHostWithAnAuthoritativeSchedulerButNoLifecycleOffersNoSwitch() {
        let caps = capabilities(lifecycle: false, v2Authoritative: true)
        #expect(DeviceControl.resolve(card(), on: caps, isChanging: false) == .readOnly)
    }

    @Test func bothFlagsTogetherOfferALiveSwitchShowingTheStoredPreference() {
        let caps = capabilities(lifecycle: true, v2Authoritative: true)
        #expect(DeviceControl.resolve(card("disabled", on: false), on: caps, isChanging: false)
            == .live(isOn: false, isEnabled: true))
        #expect(DeviceControl.resolve(card(), on: caps, isChanging: false)
            == .live(isOn: true, isEnabled: true))
    }

    @Test func aCardThatIsDrainingCannotBeFlippedAgain() {
        let caps = capabilities(lifecycle: true, v2Authoritative: true)
        #expect(DeviceControl.resolve(card("draining", on: false), on: caps, isChanging: false)
            == .live(isOn: false, isEnabled: false))
    }

    @Test func aCardThatIsStartingCannotBeFlippedAgain() {
        let caps = capabilities(lifecycle: true, v2Authoritative: true)
        #expect(DeviceControl.resolve(card("starting"), on: caps, isChanging: false)
            == .live(isOn: true, isEnabled: false))
        // And neither can one whose own change is still in flight.
        #expect(DeviceControl.resolve(card(), on: caps, isChanging: true)
            == .live(isOn: true, isEnabled: false))
    }

    @Test func aCardExcludedAtStartupGetsNoControlEvenWhereRestartEnableIsTrue() {
        let caps = capabilities(lifecycle: true, restartEnable: true, v2Authoritative: true)
        #expect(DeviceControl.resolve(card("startup_excluded", on: false),
                                      on: caps, isChanging: false) == .readOnly)
    }

    @Test func restartEnableOffersItsButtonOnlyForACardThatIsOff() {
        let caps = capabilities(restartEnable: true)
        #expect(DeviceControl.resolve(card("disabled", on: false), on: caps, isChanging: false)
            == .enableAtRestart)
        #expect(DeviceControl.resolve(card(), on: caps, isChanging: false) == .readOnly)
    }

    @Test func aMachineThatHasNotSaidWhatItCanDoOffersNothing() {
        #expect(DeviceControl.resolve(card(), on: nil, isChanging: false) == .readOnly)
    }

    // MARK: - The contextual menu

    /// **Fails today**: a GPU row has no contextual menu at all -- its one
    /// control is a `Toggle` and nothing else, so a right click on a card
    /// offers nothing.
    ///
    /// The menu is the SAME `DeviceControl` answer, so it can never offer a
    /// switch the row does not: absent where the row is read-only, absent
    /// while a card is mid-transition (asking again is a second request, not
    /// a second answer), and naming the direction it would move.
    @Test func aCardsMenuOffersExactlyTheControlItsRowDraws() {
        let live = capabilities(lifecycle: true, v2Authoritative: true)
        let name = "NVIDIA L40S  #0"

        #expect(DeviceControl.resolve(card(), on: live, isChanging: false).menu(named: name)
            == [RowAction(kind: false, title: "Stop Using \(name)")])
        #expect(DeviceControl.resolve(card("disabled", on: false), on: live, isChanging: false)
            .menu(named: name) == [RowAction(kind: true, title: "Use \(name)")])

        // Absent, never a disabled placeholder -- so no menu is attached.
        #expect(DeviceControl.resolve(card("draining", on: false), on: live, isChanging: false)
            .menu(named: name).isEmpty)
        #expect(DeviceControl.resolve(card(), on: live, isChanging: true).menu(named: name).isEmpty)
        #expect(DeviceControl.resolve(card(), on: nil, isChanging: false).menu(named: name).isEmpty)

        // The other power, worded exactly as the row's own button is.
        let restart = capabilities(restartEnable: true)
        #expect(DeviceControl.resolve(card("disabled", on: false), on: restart, isChanging: false)
            .menu(named: name) == [RowAction(kind: true, title: "Enable at next restart")])
    }

    /// **Fails today**: `MachineRow` hardcodes "Check Now" and "Set as
    /// Default" and `MachineCommands` hardcodes the same two words ten files
    /// away, with nothing shared and nothing tested. Renaming one leaves the
    /// other saying the old thing.
    @Test func aMachineRowOffersTheMachineMenusOwnItemsFirst() {
        let offered = SidebarMachineActions.offered()

        #expect(offered.map(\.kind) == [.checkNow, .setDefault, .showInLibrary])
        // The Machine menu's own two, in its order, from the one declaration
        // `MachineCommands` reads too.
        #expect(offered.prefix(2).map(\.title)
            == [SidebarMachineActions.checkNow, SidebarMachineActions.setAsDefault])
        #expect(offered.map(\.title) == ["Check Now", "Set as Default", "Show in Library"])
        // Nothing here removes anything, so nothing is destructive and there
        // is always a menu to open.
        #expect(offered.allSatisfy { !$0.isDestructive && !$0.isDisabled })
        #expect(RowAction.offersMenu(offered))
    }

    @Test func aDeviceWithNoReportedTotalHasNoBar() {
        #expect(MemoryReading(used: 12, total: nil) == nil)
        #expect(MemoryReading(used: 12, total: 0) == nil)
        let reading = MemoryReading(used: 25, total: 100)
        #expect(reading?.fraction == 0.25)
        // A machine reporting more used than it has must not draw past the end.
        #expect(MemoryReading(used: 200, total: 100)?.fraction == 1)
    }

    @Test func theSelectedMachineSurvivesBeingRemoved() {
        let workstation = MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
        let hal = MoldHost(name: "hal9000", baseURL: URL(string: "http://hal9000")!)
        let hosts = HostStore(hosts: [workstation, hal])

        #expect(hosts.machine(selected: hal.id.uuidString)?.id == hal.id)
        // The id of a machine that was removed, and a key never written at all.
        #expect(hosts.machine(selected: UUID().uuidString)?.id == hosts.preferredHost?.id)
        #expect(hosts.machine(selected: "")?.id == hosts.preferredHost?.id)
        #expect(hosts.machine(selected: nil)?.id == workstation.id)
    }
}
