import Foundation

// Deleting, repairing, loading and unloading models on one machine, and
// reading what every download on it is doing right now. A new file rather
// than a line in `+Work.swift`: `HTTPBackend`'s files already total 804
// against the 600-line advisory `make lint-type-size` tracks, and this
// milestone reports the honest new total rather than raising the threshold
// -- the `HTTPBackend+Adapters.swift` precedent from M4 S3.
public extension HTTPBackend {
    /// Removes a model's files. Refuses `409 MODEL_LOADED` while an engine is
    /// GPU-resident or mid-generation anywhere on the machine
    /// (`routes.rs:5766-5793`); a merely PARKED engine is evicted by the
    /// route itself (`routes.rs:5872-5878`), so the app never unloads first.
    @discardableResult
    func deleteModel(_ model: String) async throws -> ModelRemoval {
        let data = try await bytes(for: request(modelPath(model), method: "DELETE"))
        do {
            return try MoldJSON.decoder.decode(ModelRemoval.self, from: data)
        } catch {
            throw MoldClientError.malformedResponse
        }
    }

    /// Per-component presence. One row per manifest file, so a sharded model
    /// reports a row per shard (`model_manager.rs:1411-1426`).
    func modelComponents(_ model: String) async throws -> ModelComponentsResponse {
        try await get(modelComponentsPath(model))
    }

    /// Warms a model. The name travels in the BODY -- there is no `:model`
    /// path segment, unlike delete (`routes.rs:5341-5346`). `accept_licenses`
    /// exists on this body and is documented as IGNORED here
    /// (`routes.rs:5330-5337`), so it is not on this signature.
    func loadModel(_ model: String, gpu: Int?) async throws {
        try await send("/api/models/load", method: "POST", body: LoadModelWireBody(model: model, gpu: gpu))
    }

    /// `nil` model and `nil` gpu unloads everything on the machine
    /// (`routes.rs:5672-5680`). Answers 200 with a sentence even when
    /// nothing was loaded -- not an error.
    func unloadModel(model: String?, gpu: Int?) async throws {
        try await send("/api/models/unload", method: "DELETE", body: UnloadModelWireBody(model: model, gpu: gpu))
    }

    /// Every job this machine is running, queued or has finished, whoever
    /// asked for it -- from this app, the CLI, or another client on the same
    /// machine (`types.rs:13274-13285`).
    func downloads() async throws -> DownloadsListing {
        try await get("/api/downloads")
    }
}

extension HTTPBackend {
    /// Split out so the URL can be pinned without a network call, the same
    /// precedent as `loraPath(model:)`.
    func modelPath(_ model: String) -> String { "/api/models/\(escaped(model))" }
    func modelComponentsPath(_ model: String) -> String { "\(modelPath(model))/components" }
}

/// `POST /api/models/load`'s body (`routes.rs:5324-5338`). Named, not local
/// to `loadModel`, so its wire shape can be pinned without a network call.
struct LoadModelWireBody: Encodable {
    let model: String
    let gpu: Int?
}

/// `DELETE /api/models/unload`'s optional body (`routes.rs:5651-5660`).
struct UnloadModelWireBody: Encodable {
    let model: String?
    let gpu: Int?
}
