import Foundation
import MoldClient

/// What each machine remembers being asked for.
///
/// Per host and never merged: `mold.db` is per server, so a prompt typed
/// against plato is plato's -- the same rule models follow.
@MainActor
@Observable
final class PromptHistoryStore {
    private let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: [HistoryEntry]] = [:]
    /// The machines that answered 503 because their metadata DB is off. A
    /// machine with an EMPTY history and one with no history feature look
    /// identical in `byHost`, and the section says different things.
    private(set) var unavailable: Set<MoldHost.ID> = []

    /// Fifty is the server's own default and its listing is newest-first.
    static let limit = 50

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// A `503 HISTORY_UNAVAILABLE` is caught by CODE, not by prose, and goes
    /// into `unavailable` rather than through `hosts.report` -- a host built
    /// without a metadata DB is not failing, and a permanent banner saying so
    /// on every visit would be noise.
    func refresh(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.history(limit: Self.limit).entries
            unavailable.remove(host)
            hosts.succeeded(on: host, doing: "list what it was last asked for")
        } catch let MoldClientError.http(status, code, _) where status == 503 && code == "HISTORY_UNAVAILABLE" {
            unavailable.insert(host)
        } catch {
            hosts.report(error, on: host, doing: "list what it was last asked for")
        }
    }

    /// Clears the whole history, then re-reads so what is shown is what the
    /// machine actually kept.
    func clear(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            try await client.clearHistory(keeping: nil)
        } catch {
            hosts.report(error, on: host, doing: "clear its prompt history")
            return
        }
        await refresh(on: host)
    }

    func entries(on host: MoldHost.ID) -> [HistoryEntry] { byHost[host] ?? [] }

    /// Whether this host has ever answered -- with rows, with none, or with
    /// "this machine can't". `nil` in `byHost` and absence from
    /// `unavailable` together mean "not yet asked".
    func hasLoaded(on host: MoldHost.ID) -> Bool {
        byHost[host] != nil || unavailable.contains(host)
    }
}
