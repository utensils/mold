import Foundation

/// Explicitly requested diagnostics, distinct from a row's concise explanation.
public enum QueueFailureDetails {
    public static func diagnostic(_ entry: QueueEntry, child: BatchChild? = nil) -> String? {
        guard entry.state == .held || entry.state == .failed else { return nil }
        return [entry.errorDetail, child?.errorDetail, entry.heldReason, entry.error, child?.error]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
    }

    public static func copyText(_ entry: QueueEntry, child: BatchChild? = nil, machine: String) -> String {
        "Machine: \(machine)\nJob: \(entry.id)\nModel: \(entry.model ?? "Unknown")\n\n\(diagnostic(entry, child: child) ?? "The machine did not provide failure details.")"
    }
}
