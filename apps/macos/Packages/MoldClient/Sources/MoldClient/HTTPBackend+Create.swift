import Foundation

// Prompt transforms and prompt history. A new file rather than growing
// `+Generation.swift` (66 lines): `HTTPBackend`'s files already total 643
// against the 600-line advisory `make lint` tracks, and this is a Create
// concern, not a batch-lifecycle one. `config()`, `setConfig` and
// `resetConfig` moved out to `+Config.swift` in M7 S1 -- config, profiles
// and pairing are a Settings concern, not this one.
public extension HTTPBackend {
    /// A rewrite is one LLM completion, but the first one on an idle machine
    /// loads the expander before it can answer. Measured cold on plato:
    /// past the transport's 10 s. Three minutes is the same order as a
    /// clip download, and a machine that cannot answer in that time has
    /// something else wrong with it.
    static let promptTransformTimeout: TimeInterval = 180

    /// Rewrites a prompt. A family that reads no prompt is answered with the
    /// family guide's own advice, at 200, WITHOUT an expansion model being
    /// created, activated or pulled (`routes.rs:4066-4072`) -- so an
    /// `expanded` of one entry is the whole answer whatever `variations` said.
    func expand(_ request: ExpandRequest) async throws -> ExpandResponse {
        try await post("/api/expand", body: request, timeout: Self.promptTransformTimeout)
    }

    /// Subject-preserving alternatives. A SEPARATE route from expand on
    /// purpose, so a host too old to remix fails closed rather than silently
    /// expanding (`types.rs:782-784`).
    func remix(_ request: RemixRequest) async throws -> RemixResponse {
        try await post("/api/remix", body: request, timeout: Self.promptTransformTimeout)
    }

    /// Newest first. 503 `HISTORY_UNAVAILABLE` where the metadata DB is off.
    func history(limit: Int) async throws -> HistoryListing {
        try await get(historyPath(limit: limit))
    }

    /// Clears it. `keep` trims to the most recent N instead; there is no
    /// per-row delete and a `HistoryEntry` carries no id to name one with.
    func clearHistory(keeping keep: Int?) async throws {
        try await delete(clearHistoryPath(keeping: keep))
    }
}

extension HTTPBackend {
    /// Split out so the URL can be pinned without a network call, the same
    /// precedent as `deviceMutationPath`.
    func historyPath(limit: Int) -> String { "/api/history?limit=\(limit)" }

    func clearHistoryPath(keeping keep: Int?) -> String {
        keep.map { "/api/history?keep=\($0)" } ?? "/api/history"
    }
}
