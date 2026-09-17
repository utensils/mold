import Foundation
import MoldClient

/// The UAT-only seed straight into `byHost`, and the switch that turns
/// every mutation into a report instead of a request -- the same class of
/// hook as `QueueStore+Fixture.swift` (design M6 decision 27, here decision
/// 25): without it, a pairing screenshot is either empty or of a real
/// operator key sending a real revoke to a real machine.
@MainActor
extension PairingStore {
    /// `{"hosts": {"<machine name>": <HostFixture>}}` -- keyed by NAME,
    /// never an id a fixture author cannot know, exactly `QueueStore.Fixture`'s
    /// own rule. A name matching no configured machine is silently skipped.
    struct Fixture: Decodable {
        let hosts: [String: HostFixture]
    }

    /// One host's pairing fixture. `HostFixture.init(from:)` accepts either
    /// the bare `PairedClients` shape a fixture predating this addition
    /// still carries, or the keyed shape below that also seeds an in-flight
    /// session and an operator-required marker -- so an existing fixture
    /// file needs no rewrite.
    struct HostFixture: Decodable {
        let clients: PairedClients
        /// Seeds `PairingStore.session`/`sessionHost` so `PairingSheet` can
        /// draw a code and count down from the fixture's own `expiresAt`,
        /// with no live host at all.
        let session: PairingSession?
        /// `true` seeds `authority[host] = .paired`, the 403
        /// `PAIRING_OPERATOR_REQUIRED` state `PairingSection` draws.
        let operatorRequired: Bool?

        init(clients: PairedClients, session: PairingSession? = nil, operatorRequired: Bool? = nil) {
            self.clients = clients
            self.session = session
            self.operatorRequired = operatorRequired
        }

        init(from decoder: Decoder) throws {
            if let bare = try? PairedClients(from: decoder) {
                clients = bare
                session = nil
                operatorRequired = nil
                return
            }
            let container = try decoder.container(keyedBy: CodingKeys.self)
            clients = try container.decode(PairedClients.self, forKey: .clients)
            session = try container.decodeIfPresent(PairingSession.self, forKey: .session)
            operatorRequired = try container.decodeIfPresent(Bool.self, forKey: .operatorRequired)
        }

        // No explicit rawValue: `MoldJSON.decoder`'s `.convertFromSnakeCase`
        // matches "operator_required" against the camelCase case name
        // itself, not the other way round.
        private enum CodingKeys: String, CodingKey {
            case clients, session, operatorRequired
        }
    }

    /// Seeds `byHost`, and where the fixture names them, `session`/
    /// `sessionHost`/`authority` too -- never through `refresh` or
    /// `createSession`, which are the network round trips this fixture
    /// exists to replace -- and sets `isSeeded`, which `refresh` and every
    /// mutating method check first.
    func seed(from fixture: Fixture) {
        for host in hosts.hosts {
            guard let seed = fixture.hosts[host.name] else { continue }
            byHost[host.id] = seed.clients
            if let session = seed.session {
                self.session = session
                sessionHost = host.id
            }
            if seed.operatorRequired == true {
                authority[host.id] = .paired
            }
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
