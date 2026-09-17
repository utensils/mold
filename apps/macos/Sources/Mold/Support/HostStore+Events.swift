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

    /// Drops every live stream and opens it again.
    ///
    /// For the one case reconciling cannot fix: a stream that is dead but
    /// whose watcher does not know it yet. After a sleep the socket is gone
    /// and the watcher is either blocked reading it or waiting out a backoff
    /// of up to 32 s, so `reconcileEventStreams` sees a watcher and leaves it
    /// alone. Cancelling first is what makes the machine answer NOW -- and
    /// the reconnect's own opening `authority` frame is what tells every
    /// listener to start again (`deliver`).
    func reconnectEventStreams() {
        for id in watchers.keys { watchers.removeValue(forKey: id)?.cancel() }
        reconcileEventStreams()
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
        // The opening frame is the machine's identity, and `/api/events`
        // sends exactly one per connection (`routes.rs:11777`). So a SECOND
        // one from the same machine is not news about the machine -- it is
        // proof that this app was disconnected, and everything cached about
        // that machine is as old as the gap.
        //
        // A CHANGED identity is not the test. `instance_id` is persisted per
        // data-dir-and-port (`instance.rs:20-28`) and survives a restart, so
        // the two cases this most needs to catch -- the Mac slept while three
        // jobs finished, and `mold serve` restarted and parked every row as
        // paused -- both come back with the identity they left with. Asking
        // only about a change left the Queue pane drawing rows that no longer
        // existed until somebody pressed ⌘R.
        //
        // Repairing from `GET /api/queue`, `GET /api/devices` and
        // `GET /api/gallery` is what the server itself prescribes for a gap
        // (`routes.rs:11751-11755`), and that is exactly what a listener does
        // with `.resyncRequired`.
        if case let .authority(instanceID) = event {
            let reconnected = instanceIDs[host] != nil
            instanceIDs[host] = instanceID
            guard reconnected else { return }
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
