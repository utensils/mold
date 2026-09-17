import Foundation
import MoldClient

// Deleting, warming, cooling and inspecting one model already on a machine.
// Split from `ModelStore.swift` for size; `busy` and `componentsCache` are
// declared there because an extension cannot add a stored property.
extension ModelStore {
    /// Whether a row has a mutation running against it right now, so a view
    /// can spin the one row rather than the whole pane.
    func isBusy(with model: Model, on host: MoldHost.ID) -> Bool {
        busy[host]?.contains(model.name) ?? false
    }

    /// Removes a model's files. Answers the removal summary so the caller
    /// can say what was kept; `nil` on failure, which is already in
    /// `hosts.failures` by the time this returns -- including a
    /// `409 MODEL_LOADED`, reported with the machine's own sentence, which
    /// already names `DELETE /api/models/unload` as the fix (design decision
    /// 13, M5: this store never unloads first).
    @discardableResult
    func delete(_ model: Model, on host: MoldHost.ID) async -> ModelRemoval? {
        guard let client = hosts.backend(for: host) else { return nil }
        busy[host, default: []].insert(model.name)
        defer { busy[host]?.remove(model.name) }
        do {
            let removal = try await client.deleteModel(model.name)
            hosts.succeeded(on: host, doing: "delete \(model.name)")
            await refresh(on: host)
            return removal
        } catch {
            hosts.report(error, on: host, doing: "delete \(model.name)")
            return nil
        }
    }

    /// Warms a model onto a GPU. `gpu` is `nil` for the machine's own
    /// placement choice.
    func load(_ model: Model, gpu: Int?, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        busy[host, default: []].insert(model.name)
        defer { busy[host]?.remove(model.name) }
        do {
            try await client.loadModel(model.name, gpu: gpu)
            hosts.succeeded(on: host, doing: "load \(model.name)")
            await refresh(on: host)
        } catch {
            hosts.report(error, on: host, doing: "load \(model.name)")
        }
    }

    /// Cools a model off every GPU it happens to be loaded on. No queue
    /// check: `unload_model` does not inspect the queue either (design
    /// decision 14, M5), so a client-side guard here would be a rule the
    /// server does not have.
    func unload(_ model: Model, on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host) else { return }
        busy[host, default: []].insert(model.name)
        defer { busy[host]?.remove(model.name) }
        do {
            try await client.unloadModel(model: model.name, gpu: nil)
            hosts.succeeded(on: host, doing: "unload \(model.name)")
            await refresh(on: host)
        } catch {
            hosts.report(error, on: host, doing: "unload \(model.name)")
        }
    }

    /// Per-component presence -- one row per manifest file. Cached until
    /// this host's next `refresh(on:)`, so opening the same components sheet
    /// twice does not ask the machine twice.
    func components(of model: Model, on host: MoldHost.ID) async -> ModelComponentsResponse? {
        if let cached = componentsCache[host]?[model.name] { return cached }
        guard let client = hosts.backend(for: host) else { return nil }
        do {
            let response = try await client.modelComponents(model.name)
            componentsCache[host, default: [:]][model.name] = response
            hosts.succeeded(on: host, doing: "read \(model.name)'s components")
            return response
        } catch {
            hosts.report(error, on: host, doing: "read \(model.name)'s components")
            return nil
        }
    }
}
