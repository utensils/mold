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
    // Not `private`: `+Tick.swift` is where every decision about them lives,
    // and `private` does not cross a file boundary even within one type.
    let hosts: HostStore
    let queue: QueueStore
    /// A constructor parameter, never a constant, so a test drives ten ticks
    /// in a few milliseconds instead of sleeping through them.
    private let interval: Duration
    private var ticker: Task<Void, Never>?
    /// Held so it can be given back. `addObserver(forName:…)`'s token is the
    /// only handle on that registration, and a centre outliving this object
    /// would otherwise keep calling a block for a heartbeat nobody has.
    private let wakeCenter: NotificationCenter
    private var wakeObserver: WakeObserver?

    /// The registration token, boxed. Swift 6 refuses to let a nonisolated
    /// `deinit` touch a stored property of a non-`Sendable` type, and
    /// `NSObjectProtocol` is not one -- but this token is created in exactly
    /// one place and only ever handed straight back to the centre it came
    /// from, which is what the box says out loud.
    /// `nonisolated` because this file's default isolation is `MainActor`
    /// and a `deinit` is not.
    private nonisolated final class WakeObserver: @unchecked Sendable {
        let token: any NSObjectProtocol
        init(_ token: any NSObjectProtocol) { self.token = token }
    }

    /// How many ticks have run, and when each machine may be asked for its
    /// capabilities again. Not `private`: `+Tick.swift` is the only reader,
    /// and `private` does not cross a file boundary.
    var ticks = 0
    var capabilityRetryTick: [MoldHost.ID: Int] = [:]
    var capabilityFailures: [MoldHost.ID: Int] = [:]

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
        self.wakeCenter = wakeCenter
        wakeObserver = WakeObserver(
            wakeCenter.addObserver(forName: wakeName, object: nil, queue: nil) { [weak self] _ in
                Task { @MainActor in self?.wake() }
            })
    }

    deinit {
        // Neither of these needs to hop actors, which is what makes them
        // safe from a `deinit`: `Task.cancel()` and `removeObserver` are
        // both callable from any thread.
        ticker?.cancel()
        if let wakeObserver { wakeCenter.removeObserver(wakeObserver.token) }
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
