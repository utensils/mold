import Foundation
import MoldClient

/// Only a current running row may display sampling progress. A departed or
/// paused job must not keep showing the last preview response as live work.
enum QueueDetailPresentation {
    struct Status: Equatable {
        let title: String
        let message: String?
        let stepText: String?
        let fraction: Double?
    }

    static func status(current: QueueEntry?, progress: JobProgress?) -> Status {
        guard let current else {
            return Status(title: "No longer queued", message: "This job is no longer in the queue.", stepText: nil, fraction: nil)
        }
        let title: String = switch current.state {
        case .queued: "Queued"
        case .running: "Rendering"
        case .paused: "Paused"
        case .held: "Waiting"
        case .cancelling: "Stopping"
        case .cancelled: "Cancelled"
        case .complete: "Finished"
        case .failed: "Failed"
        case .unknown: "Status unavailable"
        }
        guard current.state == .running else {
            let message = current.waitDescription
            return Status(title: title, message: message == title ? nil : message, stepText: nil, fraction: nil)
        }
        let stage = progress?.stage?.trimmingCharacters(in: .whitespacesAndNewlines)
        let message = stage.flatMap { $0.isEmpty ? nil : $0 } ?? "Getting ready…"
        guard let step = progress?.step, let total = progress?.total, total > 0, step >= 0 else {
            return Status(title: title, message: message, stepText: nil, fraction: nil)
        }
        let completed = min(step, total)
        return Status(title: title, message: message, stepText: "Step \(completed) of \(total)",
                      fraction: Double(completed) / Double(total))
    }

    static func showsLabel(_ row: PrintDetailRow, in group: PrintDetailGroup) -> Bool {
        row.label != group.title
    }
}
