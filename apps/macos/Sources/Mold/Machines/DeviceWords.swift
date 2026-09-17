import Foundation
import MoldClient

/// Every sentence the pane says about a device, in one file, because lexicon
/// decisions are decisions.
///
/// `.generating` is "Rendering" -- the app's own word, the one already in
/// "Lost contact while rendering" and "Renders you start appear here".
enum DeviceWords {
    /// The word a row leads with. An admin state the person did something to
    /// wins over the activity, because "Idle" on a card that is draining is
    /// true and useless. `.unknown` says nothing rather than saying "unknown".
    static func state(_ device: DeviceInfo) -> String? {
        switch device.adminState {
        case .startupExcluded: return "Left out at startup"
        case .draining: return "Finishing what it has"
        case .starting: return "Starting up"
        case .disabled: return "Off"
        case .enabled, .unknown: break
        }
        switch device.activity {
        case .idle: return "Idle"
        case .loading: return "Loading a model"
        case .generating: return "Rendering"
        case .upscaling: return "Upscaling"
        case .adminLoading: return "Preparing"
        case .stopping: return "Winding down"
        case .unknown: return nil
        }
    }

    /// Printed only when it is NOT healthy: a row that says "healthy" on every
    /// card trains you to stop reading it.
    static func health(_ health: DeviceHealth) -> String? {
        switch health {
        case .healthy, .unknown: nil
        case .degraded: "Degraded"
        case .unavailable: "Unavailable"
        case .poisoned: "Faulted"
        }
    }

    /// Never `0%`: a missing utilization is not a busy-ness of zero, and only
    /// NVML reports one at all.
    static func utilization(_ percent: Int?) -> String? {
        percent.map { "\($0)%" }
    }

    /// "15.36 GB of 45 GB, 0.5 GB of it mold's", or the absence said plainly.
    ///
    /// Takes the figures rather than a `DeviceMemory`, because a row prefers
    /// the live 1 Hz sample where there is one and the card's own last answer
    /// where there is not.
    static func memory(used: UInt64?, total: UInt64?, mold: UInt64?) -> String {
        guard let total, total > 0 else { return "Memory not reported" }
        let sentence = "\(bytes(used ?? 0)) of \(bytes(total))"
        guard let mold, mold > 0 else { return sentence }
        return "\(sentence), \(bytes(mold)) of it mold's"
    }

    /// The system row's figure, which reads the other way round because a
    /// person asks how much of the machine is gone, not how much a card holds.
    static func capacity(used: UInt64, total: UInt64) -> String {
        "\(bytes(used)) of \(bytes(total)) used"
    }

    static func holding(_ models: [String]) -> String? {
        models.isEmpty ? nil : "Holding \(models.joined(separator: ", "))"
    }

    static let startupExcluded =
        "Left out when this machine started. It needs a restart to come back."
    static let needsRestart = "This machine has to restart before that takes effect."

    static func bytes(_ count: UInt64) -> String {
        Int64(clamping: count).formatted(.byteCount(style: .memory))
    }
}
