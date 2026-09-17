import Foundation
import MoldClient

/// What each machine has been told a model's controls should start at.
///
/// One `GET /api/config` per machine answers for EVERY model, which is why
/// this holds a listing rather than asking per key: a per-key GET 404s for a
/// model nobody has configured yet (`config_keys.rs:586-593`), and turning
/// "never set" into an error report is how a feature ends up shouting on a
/// fresh machine.
@MainActor
@Observable
final class ModelDefaultsStore {
    private let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: ConfigListing] = [:]
    /// The machines that answered 503 because their metadata DB is off --
    /// same reason `PromptHistoryStore` keeps one.
    private(set) var unavailable: Set<MoldHost.ID> = []

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func refresh(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.config()
            unavailable.remove(host)
            hosts.succeeded(on: host, doing: "read its configured defaults")
        } catch let MoldClientError.http(status, code, _) where status == 503 && code == "CONFIG_UNAVAILABLE" {
            unavailable.insert(host)
        } catch {
            hosts.report(error, on: host, doing: "read its configured defaults")
        }
    }

    /// Whether this host has ever answered -- with a listing, or with "this
    /// machine can't". `nil` in `byHost` and absence from `unavailable`
    /// together mean "not yet asked".
    func hasLoaded(on host: MoldHost.ID) -> Bool {
        byHost[host] != nil || unavailable.contains(host)
    }

    /// A model nobody has ever configured on this host, or whose listing
    /// hasn't been read yet, has no defaults -- `ModelDefaults()` is empty,
    /// so adopting it changes nothing and the recipe's own numbers stand.
    func defaults(for model: String, on host: MoldHost.ID) -> ModelDefaults {
        guard let listing = byHost[host] else { return ModelDefaults() }
        return ModelDefaults(from: listing, model: model)
    }

    /// Writes the fields this draft has a control for, one PUT each, and
    /// re-reads the listing afterwards so what is shown is what the machine
    /// stored rather than what was sent. A partial failure reports once for
    /// this machine -- a retry replaces it the same way every other store's
    /// failure does.
    func save(_ draft: RenderDraft, for model: String, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        var reported = false
        for (key, value) in ModelDefaults().writes(for: draft, model: model) {
            do {
                try await client.setConfig(key, to: value)
            } catch {
                if !reported {
                    hosts.report(error, on: host, doing: "save the defaults for \(model)")
                    reported = true
                }
            }
        }
        await refresh(on: host)
    }

    /// Drops every one of the eight rows for this model, then re-reads.
    func clear(for model: String, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        var reported = false
        for key in ModelDefaults.keys(for: model) {
            do {
                try await client.resetConfig(key)
            } catch {
                if !reported {
                    hosts.report(error, on: host, doing: "clear the defaults for \(model)")
                    reported = true
                }
            }
        }
        await refresh(on: host)
    }
}
