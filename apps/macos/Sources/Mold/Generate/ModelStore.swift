import Foundation
import MoldClient

/// What each machine can render with.
///
/// Models are per host, not merged: a model installed on plato is not
/// available on hal9000, and offering it would produce a request that host
/// must refuse.
@MainActor
@Observable
final class ModelStore {
    private let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: [Model]] = [:]
    private(set) var isLoading = false

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
    /// LLMs, no upscalers, no ControlNets.
    func generators(on host: MoldHost.ID) -> [Model] {
        (byHost[host] ?? []).filter(\.isGenerator)
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
}
