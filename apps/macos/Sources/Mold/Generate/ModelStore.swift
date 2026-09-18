import Foundation
import MoldClient

/// What each machine can render with.
///
/// Models are per host, not merged: a model installed on workstation is not
/// available on hal9000, and offering it would produce a request that host
/// must refuse.
@MainActor
@Observable
final class ModelStore {
    let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: [Model]] = [:]
    private(set) var isLoading = false
    /// Rows mid-mutation, per host and model name -- delete, load or unload,
    /// whichever is running -- so one row can spin without the whole pane
    /// going busy. Read and written from `ModelStore+Manage.swift`, which
    /// cannot declare a stored property of its own.
    var busy: [MoldHost.ID: Set<String>] = [:]
    /// A component listing, per (host, model), held until that host's next
    /// `refresh(on:)` -- the route's own answer is a one-second server-side
    /// cache (`model_manager.rs:230`), so re-asking inside that window would
    /// only ever repeat the same rows.
    var componentsCache: [MoldHost.ID: [String: ModelComponentsResponse]] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: (MoldHost.ID, Result<[Model], Error>).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                group.addTask {
                    do { return (host.id, .success(try await client.models())) }
                    catch { return (host.id, .failure(error)) }
                }
            }
            for await (id, result) in group {
                switch result {
                case let .success(models):
                    byHost[id] = models
                    hosts.succeeded(on: id, doing: "list its models")
                case let .failure(error):
                    hosts.report(error, on: id, doing: "list its models")
                }
            }
        }
    }

    /// One machine's models, for a caller that only needs this host rather
    /// than the whole fleet's -- the Machines pane opening on one machine.
    /// Same failure-report shape as `refresh()`'s per-host branch.
    func refresh(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            byHost[host] = try await client.models()
            componentsCache[host] = nil
            hosts.succeeded(on: host, doing: "list its models")
        } catch {
            hosts.report(error, on: host, doing: "list its models")
        }
    }

    /// Whether this host has ever answered a models listing -- distinct from
    /// an empty answer, which means it truly has none installed. `nil` from
    /// `byHost` is "not yet asked", not "asked and got nothing".
    func hasLoaded(on host: MoldHost.ID) -> Bool { byHost[host] != nil }

    /// Only things a person would pick to make a picture: no prompt-expansion
    /// LLMs, no upscalers, no ControlNets -- and no mesh families, which this
    /// app cannot yet draw (`Model.isPictureMaker`).
    func generators(on host: MoldHost.ID) -> [Model] {
        (byHost[host] ?? []).filter(\.isPictureMaker)
    }

    /// Installed and complete, which is what can run right now.
    func ready(on host: MoldHost.ID) -> [Model] {
        generators(on: host).filter(\.isReady)
    }

    /// Grouped by family so a picker reads as families of models rather than
    /// 170 flat rows of quantization tags.
    func families(on host: MoldHost.ID) -> [(family: String, models: [Model])] {
        Dictionary(grouping: ready(on: host), by: \.family)
            .map { (family: $0.key, models: $0.value.sorted { $0.name < $1.name }) }
            .sorted { $0.family < $1.family }
    }

    func model(named name: String, on host: MoldHost.ID) -> Model? {
        (byHost[host] ?? []).first { $0.name == name }
    }

    /// Every model this host has ever reported, whatever its install state
    /// or family -- the raw listing, for a caller that cross-references
    /// against the catalog rather than managing what is already here.
    func all(on host: MoldHost.ID) -> [Model] { byHost[host] ?? [] }

    /// Every model this machine holds, of every family.
    ///
    /// Deliberately not `ready(on:)`: that is the PICKER's question -- what
    /// could render right now -- and it drops an upscaler, a
    /// prompt-expansion LLM and every half-installed model. A management
    /// surface that hid a broken download would hide it in the one place
    /// somebody goes to fix it.
    func installed(on host: MoldHost.ID) -> [Model] {
        all(on: host).filter {
            switch $0.installState {
            case .available: false
            case .needsRepair, .installed, .loaded: true
            }
        }
    }
}
