import Foundation
import MoldClient

/// The seam between the four stores a machine's page already reads and the
/// pure cards the overview draws.
///
/// A value, built per body pass from the environment -- not a fifth store.
/// There is nothing here to own: every figure on a card belongs to a store
/// that already holds it, and a copy kept here would be a second answer to go
/// stale.
@MainActor
struct MachineFleet {
    let hosts: HostStore
    let machines: MachineStore
    let queue: QueueStore
    let models: ModelStore
    let activity: ActivityStore
    let upscales: UpscaleStore

    var cards: [MachineCard] {
        MachineCard.sorted(hosts.hosts.map(card(for:)))
    }

    func card(for host: MoldHost) -> MachineCard {
        MachineCard(
            host: host,
            reachability: hosts.reachability(of: host),
            isDefault: hosts.defaultMachine == host.id,
            devices: machines.devices(on: host.id),
            snapshot: machines.resource(on: host.id),
            // `nil` until the store has answered for THIS machine, which is a
            // different fact from an empty answer (`MachineFigures`).
            live: queue.hasLoaded(on: host.id)
                ? queue.entries(on: host.id).filter(\.state.isLive) : nil,
            alsoRunning: alsoRunning(on: host),
            models: models.hasLoaded(on: host.id) ? models.ready(on: host.id) : nil
        )
    }

    /// The Queue pane's own **Also Running** rows for this machine, counted
    /// by the same rule it draws them with.
    func alsoRunning(on host: MoldHost) -> Int {
        let queued = [host.id: Set(queue.entries(on: host.id).map(\.id))]
        return AlsoRunning.rows(reported: activity.rows, queuedIDs: queued,
                                upscales: upscales.live, stills: upscales.liveStills)
            .count { $0.host == host.id && !$0.isSettled }
    }

    /// What the overview asks for when it appears.
    ///
    /// One sample per machine, NOT a stream: the 1 Hz resource stream is
    /// single by construction (`MachineStore+Telemetry`) and belongs to the
    /// machine whose page is open. A fleet of streams would cost a frame per
    /// machine per second for as long as the window is open.
    ///
    /// Only machines that are UP are asked. A machine that is not answering
    /// has nothing to give, and asking anyway would file one failure per
    /// machine into the banner every time you press Back.
    func load() async {
        for host in hosts.hosts where hosts.isUp(host) {
            await machines.refresh(host.id)
            // The queue keeps itself current from `/api/events` once it has
            // loaded (`QueueStore.swift`), and installed models barely move --
            // so these are asked once each rather than on every visit.
            if !queue.hasLoaded(on: host.id) { await queue.refresh(on: host.id) }
            if !models.hasLoaded(on: host.id) { await models.refresh(on: host.id) }
        }
    }

    /// ⌘R on the overview: ask every machine what it is, then reload the
    /// figures of the ones that answered.
    func refreshAll() async {
        // `refreshAll` has already asked every machine what it is, so the
        // per-machine pass below never asks a second time.
        await hosts.refreshAll()
        for host in hosts.hosts where hosts.isUp(host) {
            await figures(of: host)
        }
    }

    /// ⌘R on one machine's page, and its toolbar button.
    func refresh(one host: MoldHost) async {
        await hosts.refresh(host)
        await figures(of: host)
    }

    private func figures(of host: MoldHost) async {
        await machines.refresh(host.id)
        await queue.refresh(on: host.id)
        await models.refresh(on: host.id)
    }
}
