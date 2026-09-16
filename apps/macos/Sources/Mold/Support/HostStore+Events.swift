import Foundation
import MoldClient

// One live connection per machine, replacing a poll per pane.
//
// It lives on `HostStore` because that is the one object that knows which
// machines exist, which are up, and what each advertises. Everything that
// cares -- the library today, the queue and the models pane later -- listens
// here rather than opening a stream of its own.
@MainActor
extension HostStore {

    /// Starts watching every reachable machine and stops watching the rest.
    ///
    /// Idempotent: call it whenever the machine list or its reachability
    /// changes. A machine already being watched is left alone rather than
    /// reconnected, because tearing down a working stream to rebuild it loses
    /// every event in between.
    func reconcileEventStreams() {
        let wanted = Set(hosts.filter { isUp($0) && watchesEvents($0) }.map(\.id))
        for id in watchers.keys where !wanted.contains(id) {
            watchers.removeValue(forKey: id)?.cancel()
        }
        for host in hosts where wanted.contains(host.id) && watchers[host.id] == nil {
            watchers[host.id] = watch(host)
        }
    }

    /// Whether this machine has an event stream at all.
    ///
    /// `events.available` absent means an older server that has no such route;
    /// polling still works, so this is a quiet fallback and not a failure.
    private func watchesEvents(_ host: MoldHost) -> Bool {
        capabilities[host.id]?.events?.available == true
    }

    private func watch(_ host: MoldHost) -> Task<Void, Never> {
        Task { [weak self] in
            // Reconnect with a widening wait. A machine that is rebooting
            // comes back in seconds; one that is off should not be hammered.
            var attempt = 0
            while !Task.isCancelled {
                guard let client = self?.backend(for: host) else { return }
                do {
                    for try await event in client.events() {
                        attempt = 0
                        self?.deliver(event, from: host.id)
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
            listeners.values.forEach { $0(host, .resyncRequired) }
            return
        }
        listeners.values.forEach { $0(host, event) }
    }

    /// Registers a listener. The returned token stops it; dropping the token
    /// on the floor leaves the listener running, which is what a store that
    /// lives as long as the app wants.
    @discardableResult
    func onEvent(_ handler: @escaping (MoldHost.ID, MoldEvent) -> Void) -> UUID {
        let id = UUID()
        listeners[id] = handler
        return id
    }

    func stopListening(_ id: UUID) { listeners.removeValue(forKey: id) }
}
