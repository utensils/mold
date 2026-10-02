import Foundation
import MoldClient

// One live `/api/events` stream per machine that answers, while the app is in
// the foreground. iOS suspends a backgrounded app's sockets without a word, so
// the supervisor stops every stream on the way out and the store reconciles
// on the way back in, rather than trusting a connection that silently died.
extension HostStore {
    /// Other stores hear every machine's events through this. Registered once
    /// each by the composition root.
    func listen(_ listener: @escaping (MoldHost.ID, MoldEvent) -> Void) {
        listeners.append(listener)
    }

    func startWatching() {
        watching = true
        reconcileWatchers()
    }

    func stopWatching() {
        watching = false
        for id in Array(watchers.keys) { stopWatching(id) }
    }

    func stopWatching(_ id: MoldHost.ID) {
        watchers.removeValue(forKey: id)?.cancel()
    }

    /// A watcher for every machine that is up, none for any that is not.
    func reconcileWatchers() {
        guard watching else { return }
        for host in hosts where isUp(host) && watchers[host.id] == nil {
            watchers[host.id] = Task { await watch(host) }
        }
        for id in Array(watchers.keys) where host(id).map(isUp) != true {
            stopWatching(id)
        }
    }

    private func watch(_ host: MoldHost) async {
        var delay: Duration = .seconds(2)
        while !Task.isCancelled {
            do {
                for try await event in backend(for: self.host(host.id) ?? host).events() {
                    delay = .seconds(2)
                    deliver(event, from: host)
                }
            } catch {
                // A dropped stream is not yet a down machine; the re-check
                // after the pause says which it is.
            }
            guard !Task.isCancelled else { return }
            try? await Task.sleep(for: delay)
            delay = min(delay * 2, .seconds(30))
            guard !Task.isCancelled, let current = self.host(host.id) else { return }
            // Every reconnect asks the machine again rather than trusting the
            // deltas it missed while the stream was down.
            await refresh(current)
            deliver(.resyncRequired, from: current)
            guard isUp(current) else { return }
        }
    }

    private func deliver(_ event: MoldEvent, from host: MoldHost) {
        switch event {
        case .deviceStateChanged, .resyncRequired:
            Task { await refreshStatus(host) }
        case .queue(.paused), .queue(.resumed):
            Task { await refreshStatus(host) }
        default:
            break
        }
        for listener in listeners { listener(host.id, event) }
    }

    /// Status only: an event is not a reason to re-read capabilities.
    private func refreshStatus(_ host: MoldHost) async {
        if case let .up(status) = await check(host) {
            setReachability(.up(status), for: host.id)
            setLastAnswered(.now, for: host.id)
        }
    }
}
