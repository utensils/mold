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
    /// The `NSOpenPanel` every well already opened for "Choose File…":
    /// png/jpeg/webP/heic/tiff, one file.
    static func chooseFile() -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    /// A print's bytes come from the machine that holds it, through the same
    /// route Quick Look uses (`MoldBackend.media`) -- never a second copy
    /// kept on this Mac.
    static func bytes(
        of drop: PictureDrop, hosts: HostStore, library: LibraryStore
    ) async throws -> (data: Data, name: String) {
        switch drop {
        case let .file(url):
            return (try Data(contentsOf: url), url.lastPathComponent)
        case let .print(id):
            guard let entry = (library.items + library.trashed).first(where: { $0.id == id }) else {
                throw MoldClientError.malformedResponse
            }
            guard let backend = hosts.backend(for: id.host) else {
                throw MoldClientError.malformedResponse
            }
            let trashed = entry.print.trashedAt != nil
            let data = try await backend.media(entry.print.filename, trashed: trashed)
            return (data, entry.print.filename)
        }
    }
}
