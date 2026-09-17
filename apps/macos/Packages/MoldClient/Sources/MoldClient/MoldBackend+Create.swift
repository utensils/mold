import Foundation

/// Prompt transforms, prompt history, per-model config, and adapters.
public protocol MoldCreateBackend: Sendable {
    /// Rewrites a prompt. A family that reads no prompt is answered with the
    /// family guide's own advice rather than an LLM rewrite.
    func expand(_ request: ExpandRequest) async throws -> ExpandResponse
    /// Subject-preserving alternatives. Separate from `expand` so a host too
    /// old to remix fails closed instead of silently expanding.
    func remix(_ request: RemixRequest) async throws -> RemixResponse
    /// Newest first. What somebody typed, not what was made.
    func history(limit: Int) async throws -> HistoryListing
    /// `nil` clears everything; otherwise trims to the most recent N. There
    /// is no per-row delete -- a `HistoryEntry` carries no id to name one.
    func clearHistory(keeping keep: Int?) async throws
    /// Every model's config surface, as `/api/config` reports it. An absent
    /// per-model key means "never configured"; a present key with a null
    /// value means the same thing.
    func config() async throws -> ConfigListing
    /// Sets one key. `models.<name>.<field>` CREATES the model's row.
    @discardableResult
    func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry
    /// Drops the DB row so the key falls back to file/env/default.
    @discardableResult
    func resetConfig(_ key: String) async throws -> ConfigEntry
    /// Installed adapters a model can take, filtered by the machine
    /// (`catalog_api.rs:1098-1115`). Refuses an unknown model.
    func loras(compatibleWith model: String) async throws -> [LoraInfo]
}

public extension MoldBackend {
    /// Clears the whole history. A protocol default rather than a parameter
    /// default: an existential call can't see a default argument, and every
    /// conformance (including the fake) gets this for free.
    func clearHistory() async throws { try await clearHistory(keeping: nil) }
}
