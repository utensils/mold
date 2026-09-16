import Foundation

/// What a machine says while you are watching it.
///
/// **The frame name is not the type.** `GET /api/events` opens with
/// `event: authority`, and everything after it arrives as the literal
/// `event: event` with the real tag in the payload's `type` field. Routing on
/// the SSE name finds "event" for every gallery change and decodes none of
/// them, which is a stream that looks connected and reports nothing.
public enum MoldEvent: Hashable, Sendable {
    /// The opening frame, naming the machine. It is the fleet identity, so
    /// cached per-host state can be fenced on it: the same address answering
    /// with a different `instance_id` is a different library.
    case authority(instanceID: String)
    /// The server's buffer overran and this client missed deltas. Repair from
    /// the listings; do not carry on.
    case resyncRequired
    case gallery(Gallery)

    public enum Gallery: Hashable, Sendable {
        /// `row` present means insert without asking again; absent means the
        /// metadata DB did not record it and the caller must go and read.
        case added(filename: String, row: GalleryPrint?)
        case updated(filename: String, row: GalleryPrint?)
        case restored(filename: String, row: GalleryPrint?)
        case removed(filename: String)
        case trashed(filename: String)
        case collectionsChanged
    }

    /// Decodes one frame, or nothing.
    ///
    /// Nothing is the normal answer: job lifecycle, chain jobs and whatever
    /// mold adds next all come down this stream, and a client that treats an
    /// unrecognised tag as a failure breaks on a server upgrade.
    public init?(name: String?, data: String) {
        guard let bytes = data.data(using: .utf8) else { return nil }
        if name == "authority" {
            guard let frame = try? MoldJSON.decoder.decode(Authority.self, from: bytes)
            else { return nil }
            self = .authority(instanceID: frame.instanceId)
            return
        }
        if name == "resync_required" {
            self = .resyncRequired
            return
        }
        guard let frame = try? MoldJSON.decoder.decode(Frame.self, from: bytes) else { return nil }
        switch frame.type {
        case "gallery_added": self = .gallery(.added(filename: frame.name, row: frame.image))
        case "gallery_updated": self = .gallery(.updated(filename: frame.name, row: frame.image))
        case "gallery_restored": self = .gallery(.restored(filename: frame.name, row: frame.image))
        case "gallery_removed": self = .gallery(.removed(filename: frame.name))
        case "gallery_trashed": self = .gallery(.trashed(filename: frame.name))
        case "gallery_collections_changed": self = .gallery(.collectionsChanged)
        default: return nil
        }
    }

    private struct Authority: Decodable {
        // Spelled as the wire spells it: `MoldJSON.decoder` converts from
        // snake case, and `instance_id` arrives as `instanceId`.
        let instanceId: String
    }

    private struct Frame: Decodable {
        let type: String
        let filename: String?
        let image: GalleryPrint?

        /// Collections changing names no print, and the gallery verbs all do.
        var name: String { filename ?? "" }
    }
}
