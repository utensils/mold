import AppKit
import Foundation
import MoldClient

/// Everything the machines are doing, including the work that never becomes a
/// queue row.
///
/// `/api/queue` answers for generations and nothing else. Preparation, a
/// prompt rewrite, a standalone upscale, a durable sequence and a download
/// are all work a machine is genuinely busy with and none of them appears
/// there -- so the Queue pane showed an idle-looking list while the GPU was
/// flat out. `/api/activity` is the machine's own answer to "what are you
/// doing", and this follows it.
///
/// It POLLS because there is no event for it. The tick runs only while Mold
/// is the frontmost app, for `HostHeartbeat`'s reason: nobody is reading a
/// pane in another app, and the first tick on becoming active is what catches
/// up.
@MainActor
@Observable
final class ActivityStore {
    let hosts: HostStore

    /// Desktop's own interval (`liveActivity.ts:11`). A constructor
    /// parameter, never a constant, so a test drives ten ticks in a
    /// millisecond instead of sleeping fifty seconds.
    private let interval: Duration

    /// What each machine last said, reconciled. Not `private`:
    /// `ActivityStore+Poll.swift` writes it.
    internal(set) var byHost: [MoldHost.ID: ActivityHostSnapshot] = [:]

    /// One number per machine. An answer arriving under an older number is an
    /// answer about a read this app has already replaced -- `QueueStore`'s
    /// own coalescing rule, expressed as a fence because these reads are
    /// started by a timer rather than by an event.
    var epochs: [MoldHost.ID: Int] = [:]

    /// One read in flight per machine, at most one queued behind it -- the
    /// same throttle the queue listing takes, and the same type, rather than
    /// a second mechanism.
    let reads = SingleFlight()

    // `@ObservationIgnored` on all three: `@Observable` turns a stored
    // property into an accessor pair that is MainActor-isolated, and a
    // `deinit` is not -- which is what `HostHeartbeat` sidesteps by not being
    // observable at all. None of these is ever drawn, so none needs to be
    // observed.
    @ObservationIgnored private var ticker: Task<Void, Never>?
    @ObservationIgnored private let activeCenter: NotificationCenter
    /// The registration tokens, boxed. Swift 6 refuses a nonisolated `deinit`
    /// access to a stored property of a non-`Sendable` type, and
    /// `NSObjectProtocol` is not one -- but these are made in one place and
    /// only ever handed straight back to the centre they came from, which is
    /// what the box says out loud. `HostHeartbeat.WakeObserver`'s own reason.
    @ObservationIgnored private var observers: ActiveObservers?

    private nonisolated final class ActiveObservers: @unchecked Sendable {
        let tokens: [any NSObjectProtocol]
        init(_ tokens: [any NSObjectProtocol]) { self.tokens = tokens }
    }

    /// The centre is injected for the same reason `HostHeartbeat`'s wake
    /// centre is: a test posts the notifications itself rather than
    /// activating an application.
    init(hosts: HostStore, interval: Duration = .seconds(5),
         activeCenter: NotificationCenter = NotificationCenter.default,
         activeName: Notification.Name = NSApplication.didBecomeActiveNotification,
         inactiveName: Notification.Name = NSApplication.didResignActiveNotification) {
        self.hosts = hosts
        self.interval = interval
        self.activeCenter = activeCenter
        observers = ActiveObservers([
            activeCenter.addObserver(forName: activeName, object: nil, queue: nil) { [weak self] _ in
                Task { @MainActor in self?.start() }
            },
            activeCenter.addObserver(forName: inactiveName, object: nil, queue: nil) { [weak self] _ in
                Task { @MainActor in self?.stop() }
            },
        ])
    }

    deinit {
        // Both are callable from any thread, which is what makes them safe
        // from a nonisolated `deinit`.
        ticker?.cancel()
        for token in observers?.tokens ?? [] { activeCenter.removeObserver(token) }
    }

    /// Starts ticking, and reads once straight away. Idempotent: becoming
    /// active twice is not a reason for two loops.
    func start() {
        guard ticker == nil else { return }
        ticker = Task {
            while !Task.isCancelled {
                await refresh()
                // `try?` would swallow the cancellation and spin this loop at
                // full speed on the main actor.
                do { try await Task.sleep(for: interval) } catch { return }
            }
        }
    }

    func stop() {
        ticker?.cancel()
        ticker = nil
    }

    var isTicking: Bool { ticker != nil }

    /// Every machine's rows in one list, ordered by submission time alone.
    var rows: [FleetActiveWork] {
        ActivityReconcile.merged(
            hosts.hosts.compactMap { host in
                byHost[host.id].map { (host: host.id, snapshot: $0) }
            })
    }
}
