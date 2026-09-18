import Foundation
import MoldClient

/// One `GET /api/config` listing per machine -- the whole of what a host will
/// tell this app about itself: per-model render defaults (`+Defaults.swift`,
/// M3's `ModelDefaultsStore`), every row the Advanced table draws (S3), and
/// what each machine most recently refused to change (`+Keys.swift`).
///
/// Widened from `ModelDefaultsStore` rather than joined by a second store
/// that also caches `GET /api/config` -- two caches of the same listing could
/// disagree, the same defect the app already avoids for its two remembered
/// machines (`defaultMachine` / `selectedMachine`).
@MainActor
@Observable
final class ConfigStore {
    let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: ConfigListing] = [:]
    /// The machines that answered 503 because their metadata DB is off --
    /// same reason `PromptHistoryStore` keeps one.
    private(set) var unavailable: Set<MoldHost.ID> = []
    /// `GET /api/config/profiles`, read-only (design decision 9): switching
    /// writes the active profile without touching the running server's
    /// loaded config, so a table that could switch would show one profile's
    /// values while an edit landed in another's.
    private(set) var profiles: [MoldHost.ID: ConfigProfiles] = [:]
    /// The last refusal per (machine, key), so a 409 or a 422 sits under the
    /// row it is about instead of in the fleet-wide banner, where it would be
    /// about nothing in particular.
    private(set) var refusals: [MoldHost.ID: [String: ConfigRefusal]] = [:]

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

    /// A host that predates `GET /api/config/profiles` answers 404 -- that
    /// is "never had the route", not a failure worth a banner over, the same
    /// "never said" reading `LibraryTags` gives a fresh tag's 404.
    func refreshProfiles(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            profiles[host] = try await client.configProfiles()
        } catch let MoldClientError.http(status, code, _) where status == 503 && code == "CONFIG_UNAVAILABLE" {
            unavailable.insert(host)
        } catch let MoldClientError.http(status: 404, _, _) {
            profiles[host] = nil
        } catch {
            hosts.report(error, on: host, doing: "read its config profiles")
        }
    }

    /// Whether this host has ever answered -- with a listing, or with "this
    /// machine can't". `nil` in `byHost` and absence from `unavailable`
    /// together mean "not yet asked".
    func hasLoaded(on host: MoldHost.ID) -> Bool {
        byHost[host] != nil || unavailable.contains(host)
    }

    /// Every row for the Advanced table, sorted by key.
    func entries(on host: MoldHost.ID) -> [ConfigEntry] {
        (byHost[host]?.entries ?? []).sorted { $0.key < $1.key }
    }

    func entry(_ key: String, on host: MoldHost.ID) -> ConfigEntry? {
        byHost[host]?.entries.first { $0.key == key }
    }

    func refusal(for key: String, on host: MoldHost.ID) -> ConfigRefusal? {
        refusals[host]?[key]
    }

    func clearRefusal(_ key: String, on host: MoldHost.ID) {
        refusals[host]?[key] = nil
    }

    /// Puts the machine's own answer into the listing in place, so the row
    /// does not flicker back to its old value while the follow-up re-read is
    /// in flight. `key` is not necessarily already present -- a
    /// `models.<name>.<field>` write CREATES that model's row.
    func applyEntry(_ entry: ConfigEntry, on host: MoldHost.ID) {
        guard let listing = byHost[host] else { return }
        var entries = listing.entries
        if let index = entries.firstIndex(where: { $0.key == entry.key }) {
            entries[index] = entry
        } else {
            entries.append(entry)
        }
        byHost[host] = ConfigListing(profile: listing.profile, entries: entries)
    }

    /// Records a refusal against its key, or marks the machine `unavailable`
    /// for the one refusal that is not about the key at all -- 503
    /// `CONFIG_UNAVAILABLE`, exactly as `refresh` already handles it. Any
    /// other error (unreachable, a bad key, a reply this build can't parse)
    /// is a machine-level failure and goes through the usual funnel.
    func recordFailure(_ error: Error, for key: String, on host: MoldHost.ID, doing verb: String) {
        switch error {
        case let MoldClientError.http(status, code, _) where status == 503 && code == "CONFIG_UNAVAILABLE":
            unavailable.insert(host)
        case let MoldClientError.http(status, code, message):
            let rebuilt = MoldClientError.http(status: status, code: code, message: message)
            refusals[host, default: [:]][key] = ConfigRefusal(code: code, sentence: rebuilt.reasonSentence)
        default:
            hosts.report(error, on: host, doing: verb)
        }
    }
}

/// The last refusal a machine gave for one key -- shown beside the row
/// itself (Advanced, S3), never through the fleet-wide banner: a 422 on
/// `expand.max_tokens` is about that row, not about the machine.
struct ConfigRefusal: Hashable, Sendable {
    let code: String?
    let sentence: String
}
