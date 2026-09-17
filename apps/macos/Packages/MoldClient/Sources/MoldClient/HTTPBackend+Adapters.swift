import Foundation

// The adapter catalog. A new file rather than a line in `+Generation.swift`:
// `HTTPBackend`'s files already total 732 against the 600-line advisory
// `make lint-type-size` tracks, and this slice reports the honest new total
// rather than raising the threshold.
public extension HTTPBackend {
    /// Installed adapters this MODEL can take, filtered by the machine.
    ///
    /// `model` is the whole compatibility decision (`catalog_api.rs:1098-1115`):
    /// the host resolves the family, answers `[]` for a family with no
    /// adapter support, and refuses an unknown model with `400
    /// UNKNOWN_MODEL`. No client matches families itself.
    func loras(compatibleWith model: String) async throws -> [LoraInfo] {
        try await get(loraPath(model: model))
    }
}

extension HTTPBackend {
    /// Split out so the URL can be pinned without a network call, the same
    /// precedent as `historyPath(limit:)`.
    func loraPath(model: String) -> String { "/api/loras?model=\(escaped(model))" }
}
