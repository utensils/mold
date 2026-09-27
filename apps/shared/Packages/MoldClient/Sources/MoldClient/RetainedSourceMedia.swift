import Foundation

/// The private conditioning media a host kept when it published a print
/// (`crates/mold-server/src/gallery_source_media.rs`).
///
/// `GET /api/gallery/source-media/:filename` answers this. The SERVER is the
/// only authority on what it retained: `OutputMetadata` under-reports it --
/// inline `source_video`, `audio_file` and `mask_image` bytes leave no marker
/// at all -- so every surface ALWAYS asks, whatever the print's metadata
/// says, and the metadata decides only whether an unavailable answer is worth
/// a sentence (`RetainedSourceMedia.disclosable`).
public enum RetainedSourceMedia {}

public extension RetainedSourceMedia {
    /// `RetainedSourceMediaAvailability` (`types.rs:10784-10791`).
    ///
    /// Open, like every wire enum here: a host newer than this build could
    /// name a fifth state, and throwing would turn a restorable print into a
    /// failed reuse. `.unknown` is not `available`, and has no sentence --
    /// studio's exhaustive `switch` answers `undefined` for the same case
    /// (`gallerySourceMedia.ts:175-188`), because there is nothing honest to
    /// say about a word this build does not know.
    enum Availability: String, OpenWireEnum {
        case available
        case unavailableLegacy = "unavailable_legacy"
        case unavailableMissingOrCorrupt = "unavailable_missing_or_corrupt"
        case unavailableAuth = "unavailable_auth"
        case unknown
    }

    /// One retained file. `memberId` is opaque and item-scoped -- never a
    /// server path, a pin id or a media-set id (`types.rs:10793-10800`).
    struct Member: Codable, Hashable, Sendable, Identifiable {
        public let memberId: String
        public let role: String
        public let displayName: String
        public let sizeBytes: Int

        public var id: String { memberId }

        public init(memberId: String, role: String, displayName: String, sizeBytes: Int) {
            self.memberId = memberId
            self.role = role
            self.displayName = displayName
            self.sizeBytes = sizeBytes
        }
    }

    /// What the host retained for one print.
    struct Inventory: Codable, Hashable, Sendable {
        public let availability: Availability
        public let members: [Member]

        public init(availability: Availability, members: [Member] = []) {
            self.availability = availability
            self.members = members
        }

        /// `members` is `skip_serializing_if = "Vec::is_empty"` on the Rust
        /// side, so it is ABSENT on every unavailable answer -- which is the
        /// commonest answer there is. Required, this would fail to decode a
        /// legacy print's reply.
        public init(from decoder: any Decoder) throws {
            let body = try decoder.container(keyedBy: CodingKeys.self)
            self.init(
                availability: try body.decode(Availability.self, forKey: .availability),
                members: try body.decodeIfPresent([Member].self, forKey: .members) ?? [])
        }
    }

    /// The opaque handle `POST …/reuse-sessions` mints
    /// (`gallery_source_media.rs:439-502`). One use, 120 s, and bound to the
    /// server instance, the credential, the archive identity and the exact
    /// request digest -- which is what stops a DIFFERENT request consuming
    /// it, and why an edit after minting means minting again.
    struct ReuseSession: Codable, Hashable, Sendable {
        public let instanceId: String
        public let expiresAt: Int
        public let requestSha256: String
        public let sessionHandle: String
    }

    /// The header the handle rides on (`REUSE_SESSION_HEADER`,
    /// `gallery_source_media.rs:24`).
    static let sessionHeader = "x-mold-retained-media-session"

    /// `MAX_REUSE_SESSION_MEMBERS` (`gallery_source_media.rs:23`).
    static let maxSessionMembers = 64
}
