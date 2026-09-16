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
    private(set) var byHost: [MoldHost.ID: [Model]] = [:]
    private(set) var isLoading = false

    func refresh(hosts: [MoldHost], using backend: (MoldHost) -> any MoldBackend) async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: (MoldHost.ID, [Model]?).self) { group in
            for host in hosts {
                let client = backend(host)
                group.addTask { (host.id, try? await client.models()) }
            }
            for await (id, models) in group where models != nil {
                byHost[id] = models
            }
        }
    }

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
