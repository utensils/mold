import Foundation

// What a machine's retained-media refusal MEANS, in this app's own words.
//
// The host's sentences are written for an API caller ("retained-media reuse
// session is missing, expired, or already consumed"). A person pressing
// Develop needs two other things: what happened to the picture, and what to
// do next -- and next is always the same, because the authority is consumed
// by the submit that took it, so pressing Develop again renders without it.
public extension RetainedSourceMedia {

    /// The codes a machine answers with (`gallery_source_media.rs`).
    enum Refusal: String, Sendable {
        case invalid = "RETAINED_MEDIA_REUSE_INVALID"
        case scopeMismatch = "RETAINED_MEDIA_REUSE_SCOPE_MISMATCH"
        case archiveChanged = "RETAINED_MEDIA_REUSE_ARCHIVE_CHANGED"
        case targetConflict = "RETAINED_MEDIA_REUSE_TARGET_CONFLICT"
        case roleUnsupported = "RETAINED_MEDIA_REUSE_ROLE_UNSUPPORTED"
        case batchAmbiguous = "RETAINED_MEDIA_REUSE_BATCH_AMBIGUOUS"
        case authRequired = "RETAINED_MEDIA_REUSE_AUTH_REQUIRED"
        case unavailable = "RETAINED_SOURCE_MEDIA_UNAVAILABLE"
        case tooLarge = "RETAINED_SOURCE_MEDIA_TOO_LARGE"

        /// The two a RE-MINT clears, because both describe the handle rather
        /// than the archive: the 120 s TTL elapsing, and the print being
        /// re-published between the probe and the submit
        /// (`gallery_source_media.rs:552-575`). Everything else is a real
        /// answer about the media and asking again would get it again.
        public var isWorthOneMoreAttempt: Bool {
            self == .invalid || self == .archiveChanged
        }

        /// What this app says. Never the host's own wording: these are read
        /// by somebody who pressed Develop, not by a caller reading a body.
        public var sentence: String {
            switch self {
            case .invalid:
                "That print's source media was held for too long and its machine has let it go."
            case .scopeMismatch:
                "The machine couldn't match this render to the print's source media."
            case .archiveChanged:
                "The print changed on its machine while this render was being prepared."
            case .targetConflict:
                "Something is already attached where this print's source media goes."
            case .roleUnsupported:
                "This print kept something this version of Mold can't reuse."
            case .batchAmbiguous:
                "A machine restores a print's source media for one render at a time."
            case .authRequired:
                "Connect this machine with an API key to restore its private source media."
            case .unavailable:
                "That print's machine no longer has its source media."
            case .tooLarge:
                "This print's source media is larger than a machine will accept in one request."
            }
        }
    }

    /// The whole refusal: what happened, and what to do about it. The way
    /// forward is always the same one, because the authority is spent by the
    /// submit that took it -- so the next press renders without the picture
    /// rather than failing the same way again.
    static func refusalSentence(for code: String?) -> String? {
        guard let code, let refusal = Refusal(rawValue: code) else { return nil }
        return refusal.sentence
            + " Nothing was queued \u{2014} press Develop again to make it without the picture."
    }
}
