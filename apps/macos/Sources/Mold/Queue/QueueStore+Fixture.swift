import Foundation
import MoldClient

/// The UAT-only seed straight into the store's own dictionaries, the switch
/// that turns every mutation into a report instead of a request, and the
/// reasons a mutation reports rather than reaching a backend at all. See
/// `QueuePane+UAT.swift` for where the file comes from (design M6 decision
/// 27): without this, every screenshot in this milestone is of an empty
/// pane (facts 1-12 of "cannot exercise"), and with it they are of real
/// wire bytes, drawn by the real views, with every action provably inert.
@MainActor
extension QueueStore {
    /// `{"hosts": {"<machine name>": {"queue": <QueueListing>, "batches":
    /// <BatchStatusListing>}}}` -- keyed by NAME, never an id a fixture
    /// author cannot know. A name matching no configured machine is
    /// silently skipped, the same "absent means nothing to show" rule every
    /// other listing here follows.
    struct Fixture: Decodable {
        struct HostFixture: Decodable {
            let queue: QueueListing
            let batches: BatchStatusListing?
        }
        let hosts: [String: HostFixture]
    }

    /// Seeds `byHost` and `children` directly -- never through `poll`, which
    /// would be the network round trip this fixture exists to replace -- and
    /// sets `isSeeded`, which `poll`, `hydrate` and every mutating method
    /// check first.
    func seed(from fixture: Fixture) {
        for host in hosts.hosts {
            guard let seed = fixture.hosts[host.name] else { continue }
            byHost[host.id] = seed.queue.merged
            if let batches = seed.batches {
                children[host.id] = Dictionary(uniqueKeysWithValues: batches.batches.map { ($0.id, $0.children) })
            }
        }
        isSeeded = true
    }

    /// The one gate every mutation checks first: a fixture-seeded store
    /// sends nothing, ever, and reports through the SAME funnel a real
    /// refusal would, so a screenshot of a held row's buttons after a click
    /// shows the same banner shape a live refusal would.
    @discardableResult
    func refuseIfFixture(_ host: MoldHost.ID, doing verb: String) -> Bool {
        guard isSeeded else { return false }
        hosts.report(FixtureRefusal(host: hosts.name(of: host) ?? "that machine"), on: host, doing: verb)
        return true
    }
}

/// "This is a fixture; nothing was sent to `<machine>`." -- `report`'s own
/// template already names the machine and the verb, so this supplies only
/// the reason.
struct FixtureRefusal: LocalizedError {
    let host: String
    var errorDescription: String? { "this is a fixture; nothing was sent to \(host)" }
}

/// Not a network failure -- the host just hasn't told us which run it is yet.
/// `report` still names the machine, because "refresh and try again" is what
/// the person needs whichever produced the sentence.
struct NoInstanceKnown: LocalizedError {
    var errorDescription: String? {
        "This machine hasn't said which run it is; refresh and try again."
    }
}

/// A held row with no batch to retry against -- `QueueEntry.authority(instanceId:)`
/// came back `nil`. Retry has nothing to send; this is what surfaces that as a
/// report rather than a silently ignored tap.
struct NotADurableBatchChild: LocalizedError {
    var errorDescription: String? {
        "This job isn't a durable batch child, so it can't be retried this way."
    }
}
