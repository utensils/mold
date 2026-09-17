import Foundation
import MoldClient

/// Mobile pairing, per machine -- for a KEYED host this app holds an
/// operator key for (design fact 7, decision 12). This Mac's own engine is
/// keyless and never appears here; `LocalEngineSettings` says why.
///
/// One store, not folded into `MachineStore`: pairing is its own wire
/// surface (`MoldConfigBackend`, beside `config()` and `configProfiles()`),
/// and `PairingSection` needs a place to hold the one in-flight session
/// that is not shaped like anything `MachineStore` already tracks.
@MainActor
@Observable
final class PairingStore {
    let hosts: HostStore
    /// Not `private(set)`: `+Fixture.swift`'s `seed(from:)` writes it too,
    /// and `private` does not cross a file boundary even within one type.
    internal(set) var byHost: [MoldHost.ID: PairedClients] = [:]
    /// What THIS app's own key can do on that machine. Absent means either
    /// "not asked yet" or "asked and it answered cleanly" -- `.paired` is
    /// the only positive fact worth keeping, set from a 403 caught by CODE
    /// (`PAIRING_OPERATOR_REQUIRED`), never by status alone, and cleared the
    /// moment a listing actually succeeds.
    private(set) var authority: [MoldHost.ID: Authority] = [:]
    /// At most one pairing session in flight, and which machine it is for --
    /// a phone scanning a code from the wrong machine would be pairing to
    /// nothing.
    private(set) var session: PairingSession?
    private(set) var sessionHost: MoldHost.ID?
    /// See `+Fixture.swift`: set by `seed(from:)`, and checked by every
    /// method below that would otherwise reach a backend. Not `private(set)`
    /// for the same cross-file reason as `byHost`.
    internal(set) var isSeeded = false

    enum Authority: Equatable {
        case `operator`
        case paired
    }

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// `GET /api/pairing/clients`. A keyless host answers cleanly with
    /// `authRequired: false` -- `PairingSection.resolve` is what turns that
    /// into "nothing to show", not a refusal here.
    func refresh(on host: MoldHost.ID) async {
        guard !isSeeded, let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.pairedClients()
            authority[host] = nil
            hosts.succeeded(on: host, doing: "list its paired clients")
        } catch let MoldClientError.http(status, code, _)
            where status == 403 && code == "PAIRING_OPERATOR_REQUIRED" {
            // Signed in, but not as the operator -- a real state the section
            // draws, not a machine-level failure.
            authority[host] = .paired
        } catch {
            hosts.report(error, on: host, doing: "list its paired clients")
        }
    }

    /// `POST /api/pairing/sessions`: a two-minute, one-use handoff. Replaces
    /// whatever session was in flight, for this host or another -- a New
    /// Code press means the old code is dead even if nobody rescans it.
    func createSession(on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "start a pairing session") else { return }
        guard let client = hosts.backend(for: host) else { return }
        do {
            session = try await client.pairingSession()
            sessionHost = host
            hosts.succeeded(on: host, doing: "start a pairing session")
        } catch {
            hosts.report(error, on: host, doing: "start a pairing session")
        }
    }

    /// Revokes, then re-reads -- never trims the row locally, the same rule
    /// `ConfigStore.set` follows for a write that might touch more than the
    /// one thing it named.
    func revoke(_ client: PairedClient, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "revoke a paired client") else { return }
        guard let backend = hosts.backend(for: host) else { return }
        do {
            try await backend.revokePairedClient(client.id)
            hosts.succeeded(on: host, doing: "revoke a paired client")
            await refresh(on: host)
        } catch {
            hosts.report(error, on: host, doing: "revoke a paired client")
        }
    }
}
