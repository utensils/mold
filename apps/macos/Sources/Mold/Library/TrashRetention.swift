import Foundation
import MoldClient

/// How long a deleted print survives, across machines that need not agree.
///
/// Retention is a per-machine setting, so a merged trash can be promising two
/// different things at once. Saying "30 days" over a mixed fleet would be a
/// promise about somebody else's machine.
enum TrashRetention {
    /// The sentence under Recently Deleted, or nil when nothing can be said.
    static func sentence(for hosts: [MoldHost], capabilities: [MoldHost.ID: Capabilities])
        -> String?
    {
        let answers = hosts.compactMap { host -> Int?? in
            guard let caps = capabilities[host.id], caps.trashEnabled else { return nil }
            // Double optional on purpose: the inner nil is "kept forever",
            // which is an answer, and the outer is "this machine has no trash".
            return .some(caps.trashRetentionDays)
        }
        guard !answers.isEmpty else { return nil }

        let distinct = Set(answers.map { $0.map(String.init) ?? "forever" })
        guard distinct.count == 1 else {
            return "Each machine deletes these on its own schedule."
        }
        guard let days = answers[0] else {
            return "These are kept until you empty the trash."
        }
        return "These are deleted after \(days) \(days == 1 ? "day" : "days")."
    }

    /// What one print has left, for its own machine's countdown.
    ///
    /// `purge_at` is derived by the host from the retention in force RIGHT
    /// NOW, never stored — so it moves when somebody changes the setting, and
    /// it is the only number worth showing per print.
    static func remaining(for print: GalleryPrint, now: Date = .now) -> String? {
        guard let purgeAt = print.purgeAt else { return nil }
        let due = Date(timeIntervalSince1970: TimeInterval(purgeAt))
        guard due > now else { return "Deleting soon" }
        let days = Calendar.current.dateComponents([.day], from: now, to: due).day ?? 0
        if days < 1 { return "Deleting today" }
        return "Deleting in \(days) \(days == 1 ? "day" : "days")"
    }
}
