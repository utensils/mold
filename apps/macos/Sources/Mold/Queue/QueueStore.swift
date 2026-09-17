import Foundation
import MoldClient

/// Queued work across every machine, and the actions on it.
@MainActor
@Observable
final class QueueStore {
    let hosts: HostStore
    /// `internal(set)`: `QueueStore+Fixture.seed(from:)` writes it too, and
    /// `private(set)` does not cross a file boundary.
    internal(set) var byHost: [MoldHost.ID: [QueueEntry]] = [:]
    private(set) var isLoading = false

    /// SOMEBODY paused this machine's whole queue -- names the QUEUE, not a
    /// row. `internal(set)`: `QueueStore+Live` writes it, and `private(set)`
    /// does not cross a file boundary.
    internal(set) var queuePaused: [MoldHost.ID: Bool] = [:]

    /// The typed half of a queue row, which `/api/queue` does not carry --
    /// per machine, then per batch id. See `QueueStore+Batches`.
    internal(set) var children: [MoldHost.ID: [String: [BatchChild]]] = [:]

    /// Set once by `seed(from:)` -- a UAT fixture, never a live host. Every
    /// mutation below checks `refuseIfFixture` first, and `poll`/`hydrate`
    /// return immediately, so a seeded store never overwrites its own
    /// fixture with whatever a real machine of the same name answers (design
    /// M6 decision 27, `QueueStore+Fixture.swift`).
    internal(set) var isSeeded = false

    /// One coalescing task per machine. Not `private`: `QueueStore+Live`
    /// reads and writes it too.
    var coalescers: [MoldHost.ID: Task<Void, Never>] = [:]

    /// The last batch hydration started for each machine, so the next one
    /// runs AFTER it rather than beside it -- see `QueueStore+Batches
    /// .hydrate(on:)`. Not `private` for the same cross-file reason.
    var hydrations: [MoldHost.ID: Task<Void, Never>] = [:]

    /// How long a burst of job frames waits before the one re-read it earns
    /// -- a stored value, not a fixed constant, so a test can shrink it
    /// instead of sleeping 250 ms per case.
    let coalesceDelay: Duration

    /// A child this store already knew settling badly -- `.failed`, or a
    /// hold the host says trying again would not help. Fired from
    /// `QueueStore+Batches`'s `hydrate`, the only thing that compares a
    /// child's state against what this store held before (decision 24).
    var onOutcome: ((MoldHost.ID, QueueEntry, String) -> Void)?

    init(hosts: HostStore, coalesceDelay: Duration = .milliseconds(250)) {
        self.hosts = hosts
        self.coalesceDelay = coalesceDelay
        // For the life of the app -- `LibraryStore.swift:56`'s shape. The
        // queue used to be poll-only, so a job that started, ran and
        // finished between two visits to this pane was never seen at all.
        hosts.onEvent { [weak self] host, event in self?.apply(event, from: host) }
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: Void.self) { group in
            for host in hosts.hosts {
                group.addTask { await self.refresh(on: host.id) }
            }
        }
    }

    /// One machine's queue, then its batches -- what every fallback calls.
    func refresh(on host: MoldHost.ID) async {
        await poll(host)
        await hydrate(on: host)
    }

    /// The one listing read: the first listing for a machine, a person
    /// asking again, a machine `wantsPoll` says cannot stream, and after any
    /// mutation this app makes (`act`, below) -- the row's new position is
    /// the server's to state.
    func poll(_ host: MoldHost.ID) async {
        guard !isSeeded, let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.queue().merged
            hosts.succeeded(on: host, doing: "list its queue")
        } catch {
            hosts.report(error, on: host, doing: "list its queue")
        }
    }

    /// Whether this machine cannot stream and needs an explicit poll instead.
    /// Absent means an older host with no event route at all.
    func wantsPoll(_ host: MoldHost.ID) -> Bool {
        hosts.capabilities[host]?.hasEvents != true
    }

    /// Whether this host has ever answered a queue listing -- distinct from
    /// an empty answer, which means it truly has nothing queued. `nil` from
    /// `byHost` is "not yet asked", not "asked and got nothing".
    func hasLoaded(on host: MoldHost.ID) -> Bool { byHost[host] != nil }

    var all: [QueueEntry] { byHost.values.flatMap(\.self) }

    func entries(on host: MoldHost.ID) -> [QueueEntry] { byHost[host] ?? [] }
}

// The ACTIONS on a row -- cancel, pause, resume, retry -- are
// `QueueStore+Actions.swift`, and a whole batch's are
// `QueueStore+GroupAction.swift`; this file holds the state and the one
// listing read. `NoInstanceKnown`, `NotADurableBatchChild` and the fixture's
// own `FixtureRefusal` live in `QueueStore+Fixture.swift`, not `private`
// here -- `private` does not cross a file boundary.
