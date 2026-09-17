import Foundation
import MoldClient

// One live connection per machine, replacing a poll per pane.
//
// It lives on `HostStore` because that is the one object that knows which
// machines exist, which are up, and what each advertises. Everything that
// cares -- the library today, the queue and the models pane later -- listens
// here rather than opening a stream of its own. `HostStore` also DRIVES it:
// `refresh` and `refreshAll` reconcile when they are done, and `forget`
// reconciles after it clears a machine, so no view's lifecycle decides what
// the app is connected to.
@MainActor
extension HostStore {

    /// Starts watching every reachable machine and stops watching the rest.
    ///
    /// Idempotent: call it whenever the machine list or its reachability
    /// changes. A machine already being watched is left alone rather than
    /// reconnected, because tearing down a working stream to rebuild it loses
    /// every event in between.
    func reconcileEventStreams() {
        let wanted = Set(hosts.filter(wantsEvents).map(\.id))
        for id in watchers.keys where !wanted.contains(id) {
            watchers.removeValue(forKey: id)?.cancel()
        }
        for host in hosts where wanted.contains(host.id) && watchers[host.id] == nil {
            watchers[host.id] = watch(host)
        }
    }

    /// Whether this machine should be watched right now.
    ///
    /// `events.available` absent means an older server that has no such route;
    /// polling still works, so this is a quiet fallback and not a failure.
    private func wantsEvents(_ host: MoldHost) -> Bool {
        guard capabilities[host.id]?.events?.available == true else { return false }
        switch reachability(of: host) {
        case .up: return true
        // Mid-check, keep whatever answer we had. Tearing down a working
        // stream because we are asking the machine again loses every event in
        // between -- and `refreshAll` checks every machine at once, so one
        // machine's reconcile would otherwise cancel another's live watcher.
        case .checking: return watchers[host.id] != nil
        case .unknown, .needsKey, .down: return false
        }
    }

    /// The fleet identity this machine last announced.
    ///
    /// The stream's opening frame when there has been one, and what
    /// `/api/status` said until then -- ONE answer, so a retry is fenced on
    /// the same identity the event stream fences on.
    func instanceID(of id: MoldHost.ID) -> String? {
        if let announced = instanceIDs[id] { return announced }
        if case let .up(status) = reachability[id] { return status.instanceId }
        return nil
    }

    /// No `[weak self]`: the store lives as long as the app, and `forget`
    /// cancels this task before anything about the machine goes away.
    private func watch(_ host: MoldHost) -> Task<Void, Never> {
        Task {
            // A cancelled task was already taken out of `watchers` by whoever
            // cancelled it, and may already have a successor -- it must not
            // clear that one's entry. Any OTHER exit means nobody knows the
            // stream is gone, so it says so rather than leaving `reconcile`
            // counting a corpse as a watcher and never reconnecting.
            defer { if !Task.isCancelled { watchers[host.id] = nil } }
            // Reconnect with a widening wait. A machine that is rebooting
            // comes back in seconds; one that is off should not be hammered.
            var attempt = 0
            while !Task.isCancelled {
                do {
                    for try await event in backend(for: host).events() {
                        attempt = 0
                        deliver(event, from: host.id)
                    }
                    // A clean end is the server shutting down, which is a
                    // reconnect like any other.
                } catch {
                    guard !Task.isCancelled else { return }
                }
                attempt = min(attempt + 1, 5)
                try? await Task.sleep(for: .seconds(pow(2.0, Double(attempt))))
            }
        }
    }

    private func deliver(_ event: MoldEvent, from host: MoldHost.ID) {
        // The opening frame is the machine's identity, not news. A DIFFERENT
        // identity at the same address is a different library, though, so
        // whoever is caching per-host state is told to start again.
        if case let .authority(instanceID) = event {
            let known = instanceIDs[host]
            instanceIDs[host] = instanceID
            guard let known, known != instanceID else { return }
            listeners.forEach { $0(host, .resyncRequired) }
            return
        }
        listeners.forEach { $0(host, event) }
    }

    /// Registers a listener for the life of the app.
    ///
    /// There is no way to stop one, because nothing wants to: every listener
    /// is a store built by the composition root and outlives every view.
    func onEvent(_ handler: @escaping (MoldHost.ID, MoldEvent) -> Void) {
        listeners.append(handler)
    }
}
