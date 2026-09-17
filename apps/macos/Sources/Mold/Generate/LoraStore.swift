import Foundation
import MoldClient

/// Installed adapters a model can take, asked per (machine, model) pair.
///
/// The compatibility decision is entirely the SERVER's
/// (`catalog_api.rs:1098-1115`): it resolves the model's family, answers `[]`
/// for a family with no adapter support, and refuses an unknown model with
/// `400 UNKNOWN_MODEL`. This store never matches families itself -- it holds
/// exactly what each machine already said about each model it was asked.
@MainActor
@Observable
final class LoraStore {
    private let hosts: HostStore
    /// Machine, then model -- the answer differs per model on one machine.
    private(set) var byHost: [MoldHost.ID: [String: [LoraInfo]]] = [:]
    /// Models a machine has already said it does not recognize, per host --
    /// asked once and never retried, the reason `refresh` is guarded on it
    /// the same way it is guarded on already having rows.
    private(set) var unknownModels: [MoldHost.ID: Set<String>] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// Asks the machine what this model can take, unless it already has an
    /// answer for this exact pair -- rows, or "unrecognized".
    func refresh(model: String, on host: MoldHost.ID) async {
        guard rows(for: model, on: host) == nil,
              !unknownModels[host, default: []].contains(model)
        else { return }
        guard let client = hosts.backend(for: host) else { return }
        do {
            let list = try await client.loras(compatibleWith: model)
            byHost[host, default: [:]][model] = list
            hosts.succeeded(on: host, doing: "list this model's adapters")
        } catch let MoldClientError.http(status, code, _) where status == 400 && code == "UNKNOWN_MODEL" {
            unknownModels[host, default: []].insert(model)
        } catch {
            hosts.report(error, on: host, doing: "list this model's adapters")
        }
    }

    /// `nil` means never answered for this pair -- neither with rows nor
    /// with "unrecognized". `[]` is a real answer: this family takes no
    /// adapters.
    func rows(for model: String, on host: MoldHost.ID) -> [LoraInfo]? {
        byHost[host]?[model]
    }
}
