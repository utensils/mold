import Foundation
import MoldClient

/// The UAT-only seed straight into `byHost`, and the switch that turns
/// every mutation into a report instead of a request -- the same class of
/// hook as `QueueStore+Fixture.swift` (design M6 decision 27, here decision
/// 25): without it, a pairing screenshot is either empty or of a real
/// operator key sending a real revoke to a real machine.
@MainActor
extension PairingStore {
    /// `{"hosts": {"<machine name>": <PairedClients>}}` -- keyed by NAME,
    /// never an id a fixture author cannot know, exactly `QueueStore.Fixture`'s
    /// own rule. A name matching no configured machine is silently skipped.
    struct Fixture: Decodable {
        let hosts: [String: PairedClients]
    }

    /// Seeds `byHost` directly -- never through `refresh`, which is the
    /// network round trip this fixture exists to replace -- and sets
    /// `isSeeded`, which `refresh` and every mutating method check first.
    func seed(from fixture: Fixture) {
        for host in hosts.hosts {
            guard let seed = fixture.hosts[host.name] else { continue }
            byHost[host.id] = seed
        }
        isSeeded = true
    }

    /// The one gate `createSession` and `revoke` both check first: a
    /// fixture-seeded store sends nothing, ever, and reports through the
    /// SAME funnel a real refusal would.
    @discardableResult
    func refuseIfFixture(_ host: MoldHost.ID, doing verb: String) -> Bool {
        guard isSeeded else { return false }
        hosts.report(PairingFixtureRefusal(host: hosts.name(of: host) ?? "that machine"), on: host, doing: verb)
        return true
    }
}

/// "This is a fixture; nothing was sent to `<machine>`." -- named apart from
/// `FixtureRefusal` (`QueueStore+Fixture.swift`) only because two types
/// cannot share one name in one module; the sentence is identical on
/// purpose.
struct PairingFixtureRefusal: LocalizedError {
    let host: String
    var errorDescription: String? { "this is a fixture; nothing was sent to \(host)" }
}
