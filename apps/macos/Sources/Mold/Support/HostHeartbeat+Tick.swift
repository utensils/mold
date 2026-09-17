import Foundation
import MoldClient

// What one tick asks of each machine. Split from `HostHeartbeat.swift` past
// the file-size advisory; that file owns the loop and this one owns the
// decision, which is the half worth reading on its own.
@MainActor
extension HostHeartbeat {

    /// What this machine is owed on this tick, if anything.
    enum Probe: Equatable, Sendable {
        case nothing
        /// `/api/status`, and `/api/capabilities` when it has not answered
        /// that yet. `HostStore.refresh(_:)` is both, and reconciles the
        /// event streams after.
        case machine
        /// A listing, for a machine that is up and cannot stream one.
        case queue
    }

    /// One pass over the fleet, every machine asked AT ONCE.
    ///
    /// Sequentially, a machine that is off held the loop for its whole
    /// connect timeout and every machine behind it waited -- so the
    /// "ten seconds" this promises became ten seconds plus every dead
    /// machine's timeout. `HostStore.refreshAll` already uses a task group
    /// for exactly this reason.
    ///
    /// Not `private`: the tests drive a single tick rather than racing the
    /// loop, and `@testable` needs it visible.
    func tick() async {
        guard !Task.isCancelled else { return }
        ticks &+= 1
        let plan = hosts.hosts.map { ($0, probe(for: $0)) }.filter { $0.1 != Probe.nothing }
        guard !plan.isEmpty else { return }
        await withTaskGroup(of: Void.self) { group in
            for (host, probe) in plan {
                group.addTask { await self.ask(probe, of: host) }
            }
        }
        // After the answers are in, and on the main actor: whether each
        // machine that was asked for its capabilities actually produced any.
        for (host, probe) in plan where probe == .machine { noteCapabilities(of: host) }
    }

    /// Pure enough to reason about: what a machine is owed, given only what
    /// this app already knows about it.
    func probe(for host: MoldHost) -> Probe {
        // Not answering: ask it again. This is the only thing that lets a
        // machine that was off at launch join without anybody pressing ⌘R.
        guard hosts.isUp(host) else { return .machine }
        // Answering, but it has never said what it can do. `/api/capabilities`
        // is fetched in one place and with `try?`, so one transient failure
        // leaves this nil -- and nil means no event stream (`wantsEvents`)
        // AND a full-rate poll (`wantsPoll`) for the rest of the session.
        // Asking again is the whole repair.
        if hosts.capabilities[host.id] == nil {
            return mayAskForCapabilities(host.id) ? .machine : .queue
        }
        // It is up and it streams: it already hears everything, and a second
        // authority asking anyway would only argue with the first.
        guard queue.wantsPoll(host.id) else { return .nothing }
        return .queue
    }

    private func ask(_ probe: Probe, of host: MoldHost) async {
        switch probe {
        case .nothing: return
        case .machine: await hosts.refresh(host)
        case .queue: await queue.refresh(on: host.id)
        }
    }

    private func mayAskForCapabilities(_ id: MoldHost.ID) -> Bool {
        ticks >= capabilityRetryTick[id] ?? 0
    }

    /// Doubling, capped at 32 ticks -- about five minutes at the default
    /// interval. A machine that answers `/api/status` and keeps failing
    /// `/api/capabilities` must not cost a request every ten seconds for the
    /// life of the app, and it must not be given up on either.
    private func noteCapabilities(of host: MoldHost) {
        // A machine that is still down is a different question; its own
        // probe already runs every tick.
        guard hosts.isUp(host) else { return }
        guard hosts.capabilities[host.id] == nil else {
            capabilityFailures[host.id] = nil
            capabilityRetryTick[host.id] = nil
            return
        }
        let failures = (capabilityFailures[host.id] ?? 0) + 1
        capabilityFailures[host.id] = failures
        capabilityRetryTick[host.id] = ticks + min(1 << min(failures, 5), 32)
    }
}
