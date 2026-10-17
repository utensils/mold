import Foundation
import os

/// One page of `GET /api/gallery/changes` (protocol version 1).
///
/// The feed is the machine's gallery change log: every name whose row changed
/// since `since`, with the row as it is NOW (or `removed`). No `since` is a full
/// snapshot paged by sequence; `reset` says the cursor the client held is no
/// longer answerable (another epoch, a restored database, a tombstone horizon
/// it fell behind) and the response restarts from sequence zero. The next
/// request carries `cursor`, and a client keeps asking while `more`.
public struct GalleryChangesPage: Decodable, Sendable, Equatable {
    public let protocolVersion: Int
    public let instanceId: String
    public let cursor: String
    public let more: Bool
    public let reset: Bool
    public let changes: [GalleryChange]

    public init(protocolVersion: Int = 1, instanceId: String, cursor: String, more: Bool = false,
                reset: Bool = false, changes: [GalleryChange]) {
        self.protocolVersion = protocolVersion
        self.instanceId = instanceId
        self.cursor = cursor
        self.more = more
        self.reset = reset
        self.changes = changes
    }
}

/// One name in the log. `image` is the row exactly as `GET /api/gallery` would
/// list it (a trashed row carries `trashed_at`); absent means the name is gone.
///
/// A row this client cannot hold -- a filename that is not one safe path
/// component, or a shape a newer machine wrote -- decodes as a removal of that
/// name, never as a failure of the whole page: the listing's rule
/// (`GalleryListing`), applied one change at a time.
public struct GalleryChange: Decodable, Sendable, Equatable {
    public let filename: String
    public let removed: Bool
    public var image: GalleryPrint?

    public init(filename: String, removed: Bool = false, image: GalleryPrint? = nil) {
        self.filename = filename
        self.removed = removed || image == nil
        self.image = removed ? nil : image
    }

    private enum CodingKeys: String, CodingKey { case filename, removed, image }

    public init(from decoder: any Decoder) throws {
        let row = try decoder.container(keyedBy: CodingKeys.self)
        filename = try row.decode(String.self, forKey: .filename)
        let gone = try row.decodeIfPresent(Bool.self, forKey: .removed) ?? false
        var image: GalleryPrint?
        if !gone {
            do {
                image = try row.decodeIfPresent(GalleryPrint.self, forKey: .image)
            } catch is SafeFilename.Rejected {
                image = nil
            } catch let error as DecodingError {
                Self.log.error(
                    "dropped a gallery change this build cannot decode: \(DecodingFailure.summary(error), privacy: .public)")
                image = nil
            }
        }
        // A row that names another file than its change is not the row for it.
        if let decoded = image, decoded.filename != filename { image = nil }
        self.image = image
        removed = image == nil
    }

    private static let log = Logger(subsystem: "io.utensils.mold.native", category: "gallery")
}

/// `capabilities.gallery.changes_feed`. Absent on an older machine, which is
/// read as "use the full listing", never as a refusal.
public struct GalleryChangesFeedCapability: Codable, Hashable, Sendable {
    public let protocolVersion: Int

    public init(protocolVersion: Int) { self.protocolVersion = protocolVersion }
}

public extension Capabilities {
    /// The only protocol this build speaks. A newer major version is a feed
    /// this client cannot follow, so it keeps the legacy listing.
    var hasGalleryChangesFeed: Bool { gallery?.changesFeed?.protocolVersion == 1 }
}
