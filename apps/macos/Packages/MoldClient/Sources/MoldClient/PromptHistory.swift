import Foundation

/// One prompt a machine remembers being asked for.
///
/// Recorded at ADMISSION, before anything is dispatched, and consecutive
/// identical `(prompt, model, negative)` rows are collapsed
/// (`crates/mold-server/src/routes.rs:2346-2378`). So this is what somebody
/// TYPED, not what was made -- a cancelled job leaves a row and a batch of
/// four leaves one. `types.rs:12792-12797`.
public struct HistoryEntry: Codable, Hashable, Sendable, Identifiable {
    public let prompt: String
    public let model: String
    /// Unix epoch milliseconds when the prompt was recorded.
    public let usedAt: Int64

    public init(prompt: String, model: String, usedAt: Int64) {
        self.prompt = prompt
        self.model = model
        self.usedAt = usedAt
    }

    /// There is no server-side id. Rows are identified by their content,
    /// which is sound precisely BECAUSE consecutive duplicates are collapsed.
    public var id: String { "\(usedAt)|\(model)|\(prompt)" }

    public var usedAtDate: Date { Date(timeIntervalSince1970: Double(usedAt) / 1000) }
}

/// `GET /api/history`'s whole-listing envelope. `types.rs:12799-12805`.
public struct HistoryListing: Codable, Hashable, Sendable {
    public let entries: [HistoryEntry]

    public init(entries: [HistoryEntry]) {
        self.entries = entries
    }
}
