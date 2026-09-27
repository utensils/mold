import Foundation

/// One compute device, as `GET /api/devices` reports it.
///
/// `id` is OPAQUE -- `cuda:<32 hex>`, `metal:default` -- and is never parsed.
/// It is the durable identity; `ordinal` is a process-local display hint and
/// means nothing across a restart.
public struct DeviceInfo: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let ordinal: Int?
    public let deviceKind: DeviceKind
    public let memory: DeviceMemory
    public let telemetry: DeviceTelemetry
    public let desiredEnabled: Bool
    /// The preference is stored, but the machine must restart before it
    /// takes effect. `#[serde(default)]` on the server: absent means false,
    /// which is an older host and not a device stuck waiting.
    public let restartRequired: Bool?
    public let adminState: DeviceAdminState
    public let health: DeviceHealth
    public let activity: DeviceActivity
    public let schedulable: Bool
    /// The machine's own sentence. Shown verbatim -- it is the only thing that
    /// explains a device that is up, healthy and still taking no work.
    public let unschedulableReason: String?
    public let loadedModels: [String]

    public var needsRestart: Bool { restartRequired ?? false }
}

public struct DeviceMemory: Codable, Hashable, Sendable {
    /// Null where the backend does not report it. Never draw a bar without it
    /// -- a 0-of-0 bar reads as a full one.
    public let totalBytes: UInt64?
    public let usedBytes: UInt64?
    /// What this mold is holding, as against everything else on the card.
    public let moldUsedBytes: UInt64?
    public let otherUsedBytes: UInt64?
}

public struct DeviceTelemetry: Codable, Hashable, Sendable {
    /// Null on Metal and on the `nvidia-smi` fallback -- only NVML has it.
    public let utilizationPercent: Int?
}

public enum DeviceKind: String, OpenWireEnum {
    case fullGpu = "full_gpu"
    case mig
    case unknownCuda = "unknown_cuda"
    case metal
    case unknown
}

public enum DeviceAdminState: String, OpenWireEnum {
    case startupExcluded = "startup_excluded"
    case starting, enabled, draining, disabled, unknown
}

public enum DeviceHealth: String, OpenWireEnum {
    case healthy, degraded, unavailable, poisoned, unknown
}

public enum DeviceActivity: String, OpenWireEnum {
    case idle, loading, generating, upscaling
    case adminLoading = "admin_loading"
    case stopping, unknown
}

/// The route's envelope. `planVersion` is 0 on every host in this fleet and
/// nothing draws it, but the body is an object and a type has to exist.
public struct DeviceState: Codable, Hashable, Sendable {
    public let devices: [DeviceInfo]
    public let planVersion: UInt64
}

/// Body of `PATCH /api/devices/{id}`.
public struct DeviceMutation: Codable, Hashable, Sendable { public let enabled: Bool }
