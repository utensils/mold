import Foundation

/// One thing a machine is doing right now, as `GET /api/activity` reports it
/// (`crates/mold-core/src/types.rs:5030-5066`).
///
/// It carries identification and progress and NOTHING else: prompts and
/// source media deliberately never cross this boundary just because another
/// client can authenticate to the host.
public struct ActiveWorkItem: Codable, Hashable, Sendable, Identifiable {
    /// Stable within this machine for the life of the work.
    public let id: String
    /// An OPEN class -- `generation`, `sequence`, `download`, or a
    /// scheduler-owned kind added after this build. A `String`, not an enum,
    /// because a kind this app has never heard of must still draw a row.
    public let kind: String
    /// `chain` identifies an auto-chained generation whose durable lifecycle
    /// lives under `/api/chain-jobs`. It is not a second kind of work.
    public let execution: String?
    /// Open too: `queued`, `preparing`, `loading`, `running`, `downloading`,
    /// `held`, `paused`, `cancelling`, or something newer.
    public let phase: String
    public let model: String?
    public let createdAtUnixMs: Int
    public let updatedAtUnixMs: Int
    public let position: Int?
    public let current: Int?
    public let total: Int?
    /// The machine's own words for where it has got to.
    public let stage: String?
    public let preparationProgress: PreparationProgress?
    /// Server-confirmed for THIS item. Absent is a definitive no.
    public let canCancel: Bool

    /// Spelled in camelCase on purpose: `MoldJSON.decoder` converts the
    /// wire's snake_case for us, so a key written `created_at_unix_ms` here
    /// would be looked up AFTER that conversion and never match.
    private enum CodingKeys: String, CodingKey {
        case id, kind, execution, phase, model
        case createdAtUnixMs, updatedAtUnixMs
        case position, current, total, stage
        case preparationProgress, canCancel
    }

    /// `can_cancel` is `#[serde(default)]` on the wire and every other field
    /// past `phase` is optional, so this decodes an older host's row rather
    /// than losing the whole snapshot over one absent key.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        kind = try container.decode(String.self, forKey: .kind)
        execution = try container.decodeIfPresent(String.self, forKey: .execution)
        phase = try container.decode(String.self, forKey: .phase)
        model = try container.decodeIfPresent(String.self, forKey: .model)
        createdAtUnixMs = try container.decode(Int.self, forKey: .createdAtUnixMs)
        updatedAtUnixMs = try container.decode(Int.self, forKey: .updatedAtUnixMs)
        position = try container.decodeIfPresent(Int.self, forKey: .position)
        current = try container.decodeIfPresent(Int.self, forKey: .current)
        total = try container.decodeIfPresent(Int.self, forKey: .total)
        stage = try container.decodeIfPresent(String.self, forKey: .stage)
        preparationProgress = try container.decodeIfPresent(
            PreparationProgress.self, forKey: .preparationProgress)
        canCancel = try container.decodeIfPresent(Bool.self, forKey: .canCancel) ?? false
    }

    /// Which authority OWNS this row, which is not always its `kind`: an
    /// ephemeral chain reports `kind: "generation"` with `execution: "chain"`
    /// and is retained or replaced with the CHAIN authority
    /// (`studio/api/activity.ts:56-58`).
    public var authorityKind: String {
        execution == "chain" ? "chain_generation" : kind
    }
}

/// A dependency the scheduler is fetching before the work can run.
public struct PreparationProgress: Codable, Hashable, Sendable {
    public let component: String
    public let bytesDone: Int
    public let bytesTotal: Int

    // No `CodingKeys`: `MoldJSON.decoder` converts `bytes_done` for us.
}

/// One machine's whole answer.
public struct ActiveWorkSnapshot: Codable, Hashable, Sendable {
    /// Fences a remembered machine whose URL or box has changed since the
    /// last snapshot was cached.
    public let instanceId: String
    public let observedAtUnixMs: Int
    public let items: [ActiveWorkItem]
    /// Work kinds whose backing authority could not be READ. A client keeps
    /// its last verified rows of these kinds and replaces the healthy ones --
    /// an unreadable database is not evidence that the work has gone.
    public let unavailableKinds: [String]

    private enum CodingKeys: String, CodingKey {
        case instanceId, observedAtUnixMs, items, unavailableKinds
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        instanceId = try container.decode(String.self, forKey: .instanceId)
        observedAtUnixMs = try container.decode(Int.self, forKey: .observedAtUnixMs)
        items = try container.decodeIfPresent([ActiveWorkItem].self, forKey: .items) ?? []
        unavailableKinds = try container.decodeIfPresent(
            [String].self, forKey: .unavailableKinds) ?? []
    }
}
