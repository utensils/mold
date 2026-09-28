import CoreTransferable
import MoldClient
import UniformTypeIdentifiers

/// A print dragged out of the grid on iPad (DESIGN.md §E): fetched from its
/// machine only when something accepts the drop, as the file itself -- a
/// picture as an image, a clip as a movie, a 3-D object as a file.
struct DraggedPrint: Transferable {
    let filename: String
    let kind: PrintKind
    let trashed: Bool
    let backend: (any MoldBackend)?
    let drop: PrintDrop

    init(_ entry: LibraryEntry, backend: (any MoldBackend)?) {
        filename = entry.print.filename
        kind = entry.print.kind
        trashed = entry.print.trashedAt != nil
        self.backend = backend
        drop = PrintDrop(host: entry.hostID, filename: entry.print.filename)
    }

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .image) { print in
            SentTransferredFile(try await print.file())
        }
        .exportingCondition { $0.kind == .picture }
        FileRepresentation(exportedContentType: .movie) { print in
            SentTransferredFile(try await print.file())
        }
        .exportingCondition { $0.kind == .clip }
        FileRepresentation(exportedContentType: .data) { print in
            SentTransferredFile(try await print.file())
        }
        // Last, so anything asking for data gets the file: within the app,
        // which print it is (a sidebar collection files it).
        ProxyRepresentation(exporting: \.drop)
    }

    private func file() async throws -> URL {
        guard let backend else { throw CocoaError(.fileNoSuchFile) }
        return try await backend.mediaFile(filename, trashed: trashed)
    }
}
