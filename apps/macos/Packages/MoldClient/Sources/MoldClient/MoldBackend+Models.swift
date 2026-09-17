import Foundation

/// Managing models already on a machine: repair status, deletion, and the
/// engine's warm/cold state. New in M5 S1b.
public protocol MoldModelsBackend: Sendable {
    /// Removes a model's files. Refuses `409 MODEL_LOADED` while an engine is
    /// GPU-resident or mid-generation anywhere on the machine
    /// (`routes.rs:5766-5793`); a merely PARKED engine is evicted by the
    /// route itself, so the app never unloads first.
    @discardableResult
    func deleteModel(_ model: String) async throws -> ModelRemoval
    /// Per-component presence. One row per manifest file, so a sharded model
    /// reports a row per shard.
    func modelComponents(_ model: String) async throws -> ModelComponentsResponse
    /// Warms a model. `accept_licenses` exists on the wire body and is
    /// documented as IGNORED here, so it is not on this signature.
    func loadModel(_ model: String, gpu: Int?) async throws
    /// `nil` model and `nil` gpu unloads everything on the machine. Answers
    /// 200 with a sentence even when nothing was loaded -- not an error.
    func unloadModel(model: String?, gpu: Int?) async throws
}
