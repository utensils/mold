import Foundation

/// Prompt transforms, prompt history, and adapters. Per-model config moved
/// out to `MoldConfigBackend` in M7 S1, beside profiles and pairing --
/// config was always an odd fit here.
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
