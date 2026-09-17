import Foundation
import MoldClient

/// Third-party licences a machine gates, and THIS machine's acceptance of
/// each.
///
/// Acceptance is per Mold data root (`types.rs:12517-12521`), so a fleet
/// holds one of these rows per machine and they never merge: a licence
/// accepted on plato is not accepted on hal9000.
@MainActor
@Observable
final class LicenseStore {
    private let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: [ThirdPartyLicense]] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// Guarded on `capabilities.hasLicenses`: an older host that never gates
    /// a model behind a licence has nothing to answer, and asking anyway
    /// would report a failure nobody caused.
    func refresh(on host: MoldHost.ID) async {
        guard hosts.capabilities[host]?.hasLicenses == true else { return }
        guard let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.licenses()
            hosts.succeeded(on: host, doing: "list its licences")
        } catch {
            hosts.report(error, on: host, doing: "list its licences")
        }
    }

    /// The licence gating this model on this machine, read from its own
    /// `requiredBy` list. `nil` means either nothing gates it or this host
    /// has never answered.
    func licence(gating model: String, on host: MoldHost.ID) -> ThirdPartyLicense? {
        byHost[host]?.first { $0.requiredBy.contains(model) }
    }

    func isAccepted(_ id: String, on host: MoldHost.ID) -> Bool {
        byHost[host]?.first { $0.id == id }?.accepted ?? false
    }

    /// Records consent on ONE machine and stores the refreshed listing the
    /// route answers with, so nothing has to re-read.
    @discardableResult
    func accept(_ license: ThirdPartyLicense, on host: MoldHost.ID) async -> Bool {
        await accept(license.acceptance, name: license.name, on: host)
    }

    @discardableResult
    func accept(_ refusal: LicenseRefusal, on host: MoldHost.ID) async -> Bool {
        await accept(refusal.acceptance, name: refusal.name, on: host)
    }

    private func accept(_ acceptance: LicenseAcceptance, name: String, on host: MoldHost.ID) async -> Bool {
        guard let client = hosts.backend(for: host) else { return false }
        do {
            byHost[host] = try await client.acceptLicenses([acceptance])
            hosts.succeeded(on: host, doing: "accept the licence for \(name)")
            return true
        } catch {
            hosts.report(error, on: host, doing: "accept the licence for \(name)")
            return false
        }
    }
}
