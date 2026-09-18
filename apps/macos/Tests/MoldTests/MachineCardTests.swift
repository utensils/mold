import Foundation
import MoldClient
import Testing

@testable import Mold

/// What one machine's card says on the fleet overview.
///
/// **Fails today**: there is no overview and no card -- the Machines
/// destination shows ONE machine's page, so there is nothing that can be asked
/// what the fleet looks like. Pinned at the pure-value level because a card is
/// read at a glance: which rows appear, and which are absent rather than
/// filled with a placeholder, is the whole design.
@MainActor
struct MachineCardTests {
    // MARK: - The GPU line

    @Test func fourIdenticalCardsAreSaidTheWayAPersonSaysThem() {
        #expect(MachineCardFigures.collapse(Array(repeating: "NVIDIA L40S", count: 4))
            == "4× NVIDIA L40S")
        #expect(MachineCardFigures.collapse(["NVIDIA L40S", "NVIDIA A40"]) == "2 GPUs")
        #expect(MachineCardFigures.collapse(["Apple M3 Max"]) == "Apple M3 Max")
        #expect(MachineCardFigures.collapse([]) == nil)
    }

    /// The host editor already collapses a machine's cards into one phrase
    /// (`HostStatus.swift`). A card saying it differently would be the same
    /// machine described two ways in one window.
    @Test func theCardSaysItTheWayTheHostEditorAlreadyDoes() {
        let names = Array(repeating: "NVIDIA L40S", count: 4)
        #expect(MachineCardFigures.collapse(names) == status(gpus: names).hardware)
    }

    /// A machine that does not let this app see its devices still reports its
    /// cards on `/api/status` -- that list is the fallback, not a blank.
    @Test func statusesOwnListAnswersWhenTheDevicesRouteIsShut() {
        let reported = status(gpus: ["NVIDIA L40S", "NVIDIA L40S"]).gpus
        #expect(MachineCardFigures.gpus(devices: [], reported: reported) == "2× NVIDIA L40S")
        #expect(MachineCardFigures.gpus(devices: [], reported: nil) == nil)
    }

    // MARK: - Load and memory

    @Test func aMachineThatReportsNoUtilizationHasNoLoadFigureAtAll() {
        // Metal and the nvidia-smi fallback report none. "0%" would say idle.
        #expect(MachineCardFigures.load(devices: [device(0)], snapshot: nil) == nil)
    }

    @Test func loadIsTheMeanOverTheCardsThatReportOne() {
        let devices = [device(0), device(1)]
        let live = snapshot(gpus: [(0, used: 0, total: 0, mold: nil, load: 80),
                                   (1, used: 0, total: 0, mold: nil, load: 20)])
        #expect(MachineCardFigures.load(devices: devices, snapshot: live) == "50%")
    }

    @Test func videoMemoryAddsUpAcrossCardsAndNamesMoldsShare() {
        let devices = [device(0), device(1)]
        let live = snapshot(gpus: [(0, used: 4_000_000_000, total: 8_000_000_000, mold: 1_000_000_000, load: nil),
                                   (1, used: 2_000_000_000, total: 8_000_000_000, mold: 1_000_000_000, load: nil)])
        let figure = MachineCardFigures.videoMemory(devices: devices, snapshot: live)
        #expect(figure?.reading == MemoryReading(used: 6_000_000_000, total: 16_000_000_000))
        #expect(figure?.text.hasSuffix("of it mold's") == true)
    }

    /// A 0-of-0 bar reads as a FULL one, which is the worst lie to tell about
    /// memory (`MemoryBar.swift`). No figures, no row.
    @Test func aMachineWhoseCardsReportNoSizeGetsNoBar() {
        #expect(MachineCardFigures.videoMemory(devices: [device(0)], snapshot: nil) == nil)
        #expect(MachineCardFigures.systemMemory(nil) == nil)
    }

    // MARK: - The whole card

    @Test func aReadyMachineCarriesEveryFigureItHas() {
        let card = card(reachability: .up(status(gpus: [])),
                        devices: [device(0), device(1)],
                        snapshot: snapshot(gpus: [(0, used: 1, total: 2, mold: nil, load: 10),
                                                  (1, used: 1, total: 2, mold: nil, load: 30)],
                                           ramUsed: 8, ramTotal: 16),
                        live: [], models: [])

        #expect(card.gpus == "2× NVIDIA L40S")
        #expect(card.gpuLoad == "20%")
        #expect(card.systemMemory?.reading == MemoryReading(used: 8, total: 16))
        // The two counts are `MachineFigures`' own sentences, not a second
        // spelling of them.
        #expect(card.work == MachineFigures.workFigure(live: []))
        #expect(card.models == MachineFigures.modelFigure(ready: []))
        #expect(card.reason == nil)
        #expect(!card.isDimmed)
    }

    /// A store that has never answered for this machine leaves the row OUT.
    /// `MachineFigures` prints an em dash on the machine's own page, where the
    /// label is already drawn; a card has no such frame to hang one on.
    @Test func aStoreThatHasNotAnsweredLeavesTheRowOutRatherThanDrawAPlaceholder() {
        let card = card(reachability: .up(status(gpus: [])), live: nil, models: nil)
        #expect(card.work == nil)
        #expect(card.models == nil)
        #expect(card.status == "Ready · 0.29.0")
    }

    /// **Fails today**: nothing keeps a down machine's stale GPU figures off a
    /// card. Its own page refuses to print them (`MachinesPane.unreachable`)
    /// -- `MachineStore` keeps the last device rows on purpose, and printing
    /// them beside a red dot claims they are current.
    @Test func aDownMachineKeepsItsPlaceDimmedAndSaysWhyInsteadOfShowingStaleFigures() {
        let card = card(reachability: .down("Connection refused"),
                        devices: [device(0)],
                        snapshot: snapshot(gpus: [(0, used: 1, total: 2, mold: nil, load: 90)],
                                           ramUsed: 8, ramTotal: 16),
                        live: [], models: [])

        #expect(card.isDimmed)
        #expect(card.reason == "Connection refused")
        #expect(card.status == "Connection refused")
        #expect(card.gpus == nil)
        #expect(card.gpuLoad == nil)
        #expect(card.systemMemory == nil)
        #expect(card.work == nil)
        // The address stands whatever the machine is doing: it is how you
        // recognise the card, not a reading taken from it.
        #expect(card.address.contains("plato"))
    }

    /// Checking is a machine we already know being asked again. Blanking its
    /// card once a second during a refresh would make the grid flicker.
    @Test func aMachineBeingCheckedKeepsTheFiguresItAlreadyHad() {
        let card = card(reachability: .checking, devices: [device(0)],
                        snapshot: snapshot(gpus: [(0, used: 1, total: 2, mold: nil, load: 90)],
                                           ramUsed: 8, ramTotal: 16),
                        live: [], models: [])
        #expect(card.gpus == "NVIDIA L40S")
        #expect(card.reason == "Checking…")
        #expect(!card.isDimmed)
    }

    /// A machine refusing this app's key has told us nothing we may print --
    /// and the fix is a key, not a reboot, so it is not dimmed like a dead one.
    @Test func aMachineThatWantsAKeySaysSoAndPrintsNothingElse() {
        let card = card(reachability: .needsKey, devices: [device(0)], live: [], models: [])
        #expect(card.reason == "This machine is there but wants an API key.")
        #expect(card.gpus == nil)
        #expect(!card.isDimmed)
        // The short form says "Needs an API key" and the long one says what to
        // do about it: both earn their line.
        #expect(card.explanation == card.reason)
    }

    /// A down machine's status line IS its reason. Printing it twice is a card
    /// nobody reads once.
    @Test func theReasonIsNotRepeatedWhenTheStatusLineAlreadySaysIt() {
        #expect(card(reachability: .down("Connection refused"), live: nil, models: nil)
            .explanation == nil)
        #expect(card(reachability: .checking, live: nil, models: nil).explanation == nil)
        #expect(card(reachability: .up(status(gpus: [])), live: nil, models: nil)
            .explanation == nil)
    }

    @Test func thisMacsCardKnowsItIsThisMac() {
        let engine = MoldHost(id: MoldEngine.localHostID, name: "This Mac",
                              baseURL: URL(string: "http://127.0.0.1:7680")!)
        let card = MachineCard(host: engine, reachability: .unknown, isDefault: false,
                               devices: [], snapshot: nil, live: nil, models: nil)
        #expect(card.isThisMac)
        #expect(!MachineCard(host: host(), reachability: .unknown, isDefault: false,
                             devices: [], snapshot: nil, live: nil, models: nil).isThisMac)
    }

    @Test func theDefaultMachineComesFirstAndTheRestGoByName() {
        let cards = MachineCard.sorted([
            named("zeus"), named("alpha"), named("GPU 10"), named("GPU 9"),
            named("workstation", isDefault: true),
        ])
        #expect(cards.map(\.name) == ["workstation", "alpha", "GPU 9", "GPU 10", "zeus"])
    }

    // MARK: - Fixtures

    private func named(_ name: String, isDefault: Bool = false) -> MachineCard {
        MachineCard(host: host(name), reachability: .unknown, isDefault: isDefault,
                    devices: [], snapshot: nil, live: nil, models: nil)
    }

    /// The address is derived from the name so a card's own address line can
    /// be recognised -- spaces squeezed out, because a machine may be called
    /// "GPU 10" and that is not a hostname.
    private func host(_ name: String = "plato") -> MoldHost {
        let address = name.replacingOccurrences(of: " ", with: "-").lowercased()
        return MoldHost(name: name, baseURL: URL(string: "http://\(address).local:7680")!)
    }

    private func card(reachability: HostStore.Reachability, devices: [DeviceInfo] = [],
                      snapshot: ResourceSnapshot? = nil, live: [QueueEntry]?,
                      models: [Model]?) -> MachineCard {
        MachineCard(host: host(), reachability: reachability, isDefault: false,
                    devices: devices, snapshot: snapshot, live: live, models: models)
    }

    /// A status carrying its own GPU list, which `FakeFixtures.serverStatus`
    /// does not plant -- decoded the way the wire produces it.
    private func status(gpus: [String]) -> ServerStatus {
        let rows = gpus.enumerated()
            .map { #"{"ordinal": \#($0.offset), "name": "\#($0.element)"}"# }
            .joined(separator: ",")
        let json = """
        {"version": "0.29.0", "hostname": "plato", "busy": false, "uptime_secs": 10,
         "gpus": [\(rows)]}
        """
        return try! MoldJSON.decoder.decode(ServerStatus.self, from: Data(json.utf8))
    }

    private func device(_ ordinal: Int) -> DeviceInfo {
        let json = """
        {"id": "cuda:\(ordinal)", "name": "NVIDIA L40S", "ordinal": \(ordinal),
         "device_kind": "full_gpu", "memory": {}, "telemetry": {}, "desired_enabled": true,
         "admin_state": "enabled", "health": "healthy", "activity": "idle",
         "schedulable": true, "loaded_models": []}
        """
        return try! MoldJSON.decoder.decode(DeviceInfo.self, from: Data(json.utf8))
    }

    private func snapshot(
        gpus: [(ordinal: Int, used: UInt64, total: UInt64, mold: UInt64?, load: Int?)],
        ramUsed: UInt64 = 1, ramTotal: UInt64 = 2
    ) -> ResourceSnapshot {
        let rows = gpus.map {
            #"""
            {"ordinal": \#($0.ordinal), "vram_total": \#($0.total), "vram_used": \#($0.used),
             "vram_used_by_mold": \#($0.mold.map(String.init) ?? "null"),
             "gpu_utilization": \#($0.load.map(String.init) ?? "null")}
            """#
        }.joined(separator: ",")
        let json = #"""
        {"hostname": "plato", "gpus": [\#(rows)],
         "system_ram": {"total": \#(ramTotal), "used": \#(ramUsed), "used_by_mold": 0}}
        """#
        return try! MoldJSON.decoder.decode(ResourceSnapshot.self, from: Data(json.utf8))
    }
}
