import AppKit
import Foundation
import MoldClient

/// The app asking the machines how they are, without anybody pressing ⌘R.
///
/// Everything else in this app is event-driven, which is right for a machine
/// whose stream is live and says nothing at all about the three that are not:
/// a machine that was off when Mold launched is never probed again, a machine
/// too old to advertise `/api/events` is read exactly once
/// (`QueueStore.wantsPoll` named that fallback and had no production caller),
/// and a Mac that sleeps wakes with dead sockets its watchers have not
/// noticed. Desktop's own answer is a 10 s loop (`hosts.ts:49,1573-1586`);
/// this is the same interval, narrowed to the machines that need it.
///
/// Deliberately NOT a poll for every machine: one with a live stream already
/// hears everything, and asking it anyway would be a second authority
/// competing with the first.
@MainActor
final class HostHeartbeat {
    private let hosts: HostStore
    private let queue: QueueStore
    /// A constructor parameter, never a constant, so a test drives ten ticks
    /// in a few milliseconds instead of sleeping through them.
    private let interval: Duration
    private var ticker: Task<Void, Never>?
    private var wakeObserver: (any NSObjectProtocol)?

    /// `NSWorkspace`'s own centre in production; a test hands in a plain one
    /// and posts `wakeName` itself -- the `LandedPrints` idiom, where the
    /// observed centre is the injected `UserDefaults`'.
    init(
        hosts: HostStore, queue: QueueStore, interval: Duration = .seconds(10),
        wakeCenter: NotificationCenter = NSWorkspace.shared.notificationCenter,
        wakeName: Notification.Name = NSWorkspace.didWakeNotification
    ) {
        self.hosts = hosts
        self.queue = queue
        self.interval = interval
        wakeObserver = wakeCenter.addObserver(forName: wakeName, object: nil, queue: nil) { [weak self] _ in
            Task { @MainActor in self?.wake() }
        }
    }

    deinit {
        // Cancelling from `deinit` is the one thing that cannot hop actors,
        // and neither of these needs to: `Task.cancel()` and
        // `removeObserver` are both safe from any thread.
        ticker?.cancel()
    }

    /// Starts ticking. Idempotent -- a second call while one is running is
    /// the app becoming active twice, not a reason for two loops.
    func start() {
        guard ticker == nil else { return }
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                // `try?` here would swallow the cancellation and spin this
                // loop at full speed on the main actor -- the exact shape of
                // `DownloadStore.awaitSettlement`'s own defect.
                do { try await Task.sleep(for: interval) } catch { return }
                guard !Task.isCancelled else { return }
                await tick()
            }
        }
    }

    /// Stops ticking. Called when Mold is no longer the active app: a machine
    /// with no event route is the only thing this loop learns from, and
    /// nothing is drawing it while somebody is in another app.
    func stop() {
        ticker?.cancel()
        ticker = nil
    }

    /// One pass over the fleet. Not `private`: the tests drive a single tick
    /// rather than racing the loop, and `@testable` needs it visible.
    func tick() async {
        for host in hosts.hosts {
            // A machine that is not answering is asked again -- this is the
            // only thing that lets one that was off at launch join without
            // the person pressing ⌘R. `refresh` reconciles the event streams
            // itself, so coming back also opens its stream.
            guard hosts.isUp(host) else {
                await hosts.refresh(host)
                continue
            }
            // It IS up: the only thing left to ask is the queue, and only of
            // a machine that cannot stream it.
            guard queue.wantsPoll(host.id) else { continue }
            await queue.refresh(on: host.id)
        }
    }

    /// The Mac woke up. Every socket opened before the sleep is dead, and the
    /// watcher holding it is either blocked on it or waiting out a backoff of
    /// up to 32 s, so nothing reconnects for a long time and nothing
    /// reconciles when it finally does. Dropping the streams makes each
    /// machine answer now, and the reconnect's own `authority` frame is what
    /// tells every listener the gap happened.
    ///
    /// Not `private`, same reason as `tick()`.
    func wake() {
        hosts.reconnectEventStreams()
        Task { await hosts.refreshAll() }
    }
}
