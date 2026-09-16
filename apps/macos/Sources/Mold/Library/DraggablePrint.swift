import Foundation
import MoldClient
import SwiftUI
import UniformTypeIdentifiers

/// A print being dragged out of the library.
///
/// The file is fetched when the drop happens, not when the drag starts: the
/// bytes live on another machine and a clip can be hundreds of megabytes, so
/// pulling one down on mouse-down would stall every drag that was never
/// dropped anywhere.
struct DraggablePrint: Transferable, Sendable {
    let id: PrintID
    let filename: String
    let format: String
    /// Hands back a file on this disk. The drag does not own the file and
    /// does not copy it -- the materializer's cache is where it lives, and the
    /// Finder copies out of it.
    let file: @Sendable () async -> URL?

    static var transferRepresentation: some TransferRepresentation {
        // Two representations, and the order is the offer: another app takes
        // the file, and Mold itself takes the identity -- which is what makes
        // dropping a print on a collection cost nothing, while dropping the
        // same print on the Finder still costs a download.
        ProxyRepresentation(exporting: \.id)
        FileRepresentation(exportedContentType: .data) { print in
            guard let file = await print.file() else {
                throw CocoaError(.fileNoSuchFile)
            }
            // The materializer already keeps the print's own name inside a
            // keyed directory, so what lands in the Finder is named the way
            // it is on the machine that made it.
            return SentTransferredFile(file)
        }
        .suggestedFileName { $0.filename }
    }
}

extension LibraryActions {
    func draggable(_ entry: LibraryEntry) -> DraggablePrint {
        DraggablePrint(
            id: entry.id,
            filename: entry.print.filename,
            format: entry.print.format ?? "png",
            file: { [self] in await files(for: [entry]).first?.url }
        )
    }
}
