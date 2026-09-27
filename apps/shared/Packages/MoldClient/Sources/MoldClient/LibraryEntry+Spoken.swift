import Foundation

public extension LibraryEntry {
    /// What VoiceOver reads for this print.
    ///
    /// A tile draws a picture, a star, a play badge, a machine name and a
    /// purge countdown, and a sighted person takes in all five at a glance.
    /// The label carries all five for the same reason -- a label that is only
    /// the prompt describes a different, simpler tile than the one on screen.
    ///
    /// Phrases separated by commas, because VoiceOver pauses at one, which is
    /// what turns a run of facts into something a person can follow.
    func spokenDescription(showsHost: Bool, now: Date = .now) -> String {
        var parts = [spokenName]
        parts.append(print.kind.spoken)
        if print.isFavorite { parts.append("favourite") }
        if showsHost {
            parts.append("on " + ListFormatter.localizedString(byJoining: hostNames))
        }
        if print.trashedAt != nil {
            parts.append("deleted")
            parts.append(remaining(now))
        }
        return parts.joined(separator: ", ")
    }

    /// Their own title if they gave one; otherwise what they asked for, which
    /// says far more than a generated filename; otherwise the filename.
    private var spokenName: String {
        if print.title?.trimmingCharacters(in: .whitespaces).isEmpty == false {
            return print.displayName
        }
        if let prompt = print.metadata.prompt, !prompt.isEmpty { return prompt }
        return print.filename
    }

    private func remaining(_ now: Date) -> String {
        guard let days = TrashCountdown.days(until: print.purgeAt, now: now) else {
            return "no purge date"
        }
        if days <= 0 { return "today" }
        return days == 1 ? "1 day left" : "\(days) days left"
    }
}

extension PrintKind {
    /// The word for the badge, said aloud. "mp4" is a container, not a thing
    /// somebody made.
    var spoken: String {
        switch self {
        case .picture: "picture"
        case .clip: "clip"
        case .mesh: "mesh"
        }
    }
}
