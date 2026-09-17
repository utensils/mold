import Foundation

/// One manifest file's presence on this machine, as `GET
/// /api/models/:model/components` reports it (`model_manager.rs:1411-1426`).
/// A sharded model reports one row per shard.
public struct ModelComponentStatus: Codable, Hashable, Sendable, Identifiable {
    public let kind: String
    public let name: String
    public let present: Bool
    public let path: String?
    /// What to re-install to repair a MISSING component: the resolved model
    /// name, re-enqueued through the same route that installs it fresh
    /// (`model_manager.rs:1422`).
    public let repairModel: String?
    /// Every file of this coarse KIND the machine holds anywhere
    /// (`model_manager.rs:1423` -> `component_options_for_kind`). Measured on
    /// plato: 103 entries on one `transformer` slot, led by an unrelated
    /// OpenCLIP checkpoint. This is the candidate list for a
    /// `models.<name>.<component>_path` override, NOT a list of things that
    /// serve this component -- decoded here so the type round-trips, and
    /// READ BY NOBODY. The components sheet draws one row per component and
    /// never this field (design fact 4, M5).
    public let options: [ModelComponentOption]?

    public var id: String { "\(kind)/\(name)" }
}

/// One candidate file for a `models.<name>.<component>_path` override.
public struct ModelComponentOption: Codable, Hashable, Sendable {
    public let label: String
    public let path: String
    public let present: Bool
}

public struct ModelComponentsResponse: Codable, Hashable, Sendable {
    public let model: String
    public let components: [ModelComponentStatus]
}

/// What `DELETE /api/models/:model` actually did (`types.rs:12816-12827`).
public struct ModelRemoval: Codable, Hashable, Sendable {
    public let removed: [String]
    /// Shared files kept because another installed model still names them --
    /// the ref-count that makes a sum of `Model.diskUsageBytes` meaningless
    /// (design fact 3, M5).
    public let kept: [KeptComponent]
    public let freedBytes: Int64
}

/// One shared component `DELETE` kept because another installed model still
/// references it (`types.rs:12805-12813`).
public struct KeptComponent: Codable, Hashable, Sendable, Identifiable {
    public let component: String
    public let usedBy: [String]
    public var id: String { component }
}
