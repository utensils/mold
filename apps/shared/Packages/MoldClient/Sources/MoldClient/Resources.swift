import Foundation

/// One 1 Hz sample from `GET /api/resources` and `/api/resources/stream`.
public struct ResourceSnapshot: Codable, Hashable, Sendable {
    public let hostname: String
    public let gpus: [GpuSample]
    public let systemRam: RamSample
    /// Absent until the aggregator has two samples -- CPU use is a delta, and
    /// the first snapshot would always read zero.
    public let cpu: CpuSample?
}

/// The moving half of a device's memory. Matched to a `DeviceInfo` by
/// `ordinal` -- legitimate because BOTH come from the same server process in
/// the same second, which is the only scope in which an ordinal means
/// anything. Never matched across hosts, and never used as an identity.
public struct GpuSample: Codable, Hashable, Sendable {
    public let ordinal: Int
    public let vramTotal: UInt64
    public let vramUsed: UInt64
    public let vramUsedByMold: UInt64?
    public let gpuUtilization: Int?
}

public struct RamSample: Codable, Hashable, Sendable {
    public let total: UInt64
    public let used: UInt64
    /// `MemAvailable`. Absent on an older host -- fall back to total - used.
    public let available: UInt64?
    public let usedByMold: UInt64
}

public struct CpuSample: Codable, Hashable, Sendable {
    public let cores: Int
    public let usagePercent: Double
}

public extension RamSample {
    /// `available`, or the total-minus-used fallback for a host that predates
    /// the field.
    var availableOrComputed: UInt64 {
        available ?? (total > used ? total - used : 0)
    }
}
