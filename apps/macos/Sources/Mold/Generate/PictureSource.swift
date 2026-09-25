import AppKit
import CoreTransferable
import Foundation
import MoldClient
import UniformTypeIdentifiers

/// What lands in a picture well: a file from the Finder or a print dragged
/// out of the Library (M8 decision 5). One `Transferable` importing either,
/// so a well needs one `dropDestination`.
enum PictureDrop: Transferable {
    case file(URL)
    case print(PrintID)

    static var transferRepresentation: some TransferRepresentation {
        // `PrintID` FIRST: a print dragged out of the Library also carries a
        // file representation (`DraggablePrint`), and SwiftUI picks the
        // first importing representation that matches -- so an in-app drag
        // prefers the identity, which costs nothing, over the file, which
        // costs a download.
        ProxyRepresentation { PictureDrop.print($0) }
        ProxyRepresentation { PictureDrop.file($0) }
    }
}

/// The bytes for a picture from either place a well accepts one.
enum PictureSource {
    /// THE `NSOpenPanel` for a picture: png/jpeg/webP/heic/tiff. Several files
    /// only where the well behind it feeds a LIST -- the reference strip and
    /// the identity group -- because everywhere else a second file would
    /// silently overwrite the first.
    ///
    /// The panel offers more than any one well accepts on the wire on purpose:
    /// HEIC is the format every iPhone photograph arrives in, and
    /// `PictureImport` transcodes it rather than the panel pretending the
    /// picture does not exist.
    static func choose(allowsMultiple: Bool = false) -> [URL] {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = allowsMultiple
        guard panel.runModal() == .OK else { return [] }
        return panel.urls
    }

    /// A print's bytes come from the machine that holds it, through the same
    /// route Quick Look uses (`MoldBackend.media`) -- never a second copy
    /// kept on this Mac.
    ///
    /// Both doors answer the WELL's own acceptance policy. A print's bytes are
    /// something mold made, which is not the same as something this well's
    /// path can decode: the identity encoder reads a PNG signature and then
    /// JPEG markers and nothing else, so a WebP print is transcoded here
    /// exactly as a HEIC file is, rather than uploaded whole and refused.
    static func bytes(
        of drop: PictureDrop, accepting: Set<String>,
        hosts: HostStore, library: LibraryStore
    ) async throws -> ImportedPicture {
        switch drop {
        case let .file(url):
            return try await PictureImport.load(url, accepting: accepting)
        case let .print(id):
            guard let entry = library.entry(id) else {
                throw MoldClientError.malformedResponse
            }
            guard let backend = hosts.backend(for: id.host) else {
                throw MoldClientError.malformedResponse
            }
            let trashed = entry.print.trashedAt != nil
            let data = try await backend.media(entry.print.filename, trashed: trashed)
            return try await PictureImport.conforming(
                data, name: entry.print.filename, accepting: accepting)
        }
    }
}
