import CryptoKit
import Foundation

public extension GalleryPrint {
    /// SHA-256 of the exact recipe bytes the server sent, or nil when the row
    /// carried none. Cheap next to canonicalizing (no parse, no re-encode),
    /// so it can key a cache of canonical-recipe digests without collisions.
    var rawMetadataDigest: String? {
        rawMetadataJSON.map { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }
}

public extension GalleryPrint {
    /// The same row carrying exact raw recipe bytes, as a listing read from
    /// the wire does. For fixtures that build rows by hand.
    func carryingRawMetadata(_ raw: Data) -> GalleryPrint {
        var copy = self
        copy.rawMetadataJSON = raw
        return copy
    }
}
