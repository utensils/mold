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
    let filename: String
    let format: String
    let fetch: @Sendable () async -> Data?

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .data) { print in
            guard let data = await print.fetch() else {
                throw CocoaError(.fileNoSuchFile)
            }
            // A real file with the print's own name, so what lands in the
            // Finder is named the way it is on the machine that made it.
            let url = FileManager.default.temporaryDirectory
                .appending(path: "mold-drag-\(UUID().uuidString)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            let file = url.appending(path: print.filename)
            try data.write(to: file)
            return SentTransferredFile(file)
        }
        .suggestedFileName { $0.filename }
    }
}

extension LibraryActions {
    func draggable(_ entry: LibraryEntry) -> DraggablePrint {
        DraggablePrint(
            filename: entry.print.filename,
            format: entry.print.format ?? "png",
            fetch: { [self] in await data(for: entry) }
        )
    }
}
