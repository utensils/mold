import Foundation
import MoldClient

// What the 3-D view needs from the Library: the stored bytes, and the
// sentence to put on the poster when they do not arrive.
@MainActor
extension LibraryActions {
    /// The stored GLB, bounded by the reader's own cap.
    ///
    /// THROWS rather than reporting, unlike `data(for:)`: a mesh that will not
    /// load lands on the poster with one line saying why, and a toast beside
    /// an empty frame would say it twice and explain neither.
    ///
    /// `data(for:)`'s route -- `media`, which carries the host's key -- and
    /// never `playableURL`, whose ticket is for a player that builds its own
    /// requests.
    func meshBytes(for entry: LibraryEntry) async throws -> Data {
        guard let backend = hosts.backend(for: entry.hostID) else {
            throw MeshViewFailure.transport("That machine isn't connected.")
        }
        let bytes = try await backend.media(entry.print.filename,
                                            trashed: entry.print.trashedAt != nil)
        return try ResponseCeiling.checked(
            bytes, ceiling: min(ResponseCeiling.media, GLB.maximumBytes),
            what: "that mesh")
    }

    /// A mesh's file for Quick Look: the host's POSTER, named `.png`.
    ///
    /// macOS ships no GLB preview generator, so materializing the stored
    /// bytes gave the panel a container it draws a generic icon for -- after
    /// downloading however many megabytes it was. The poster is the picture
    /// the gallery tile already shows and the 3-D view's own home frame, so
    /// previewing a mesh shows the mesh.
    ///
    /// The `.png` extension is the whole point: Quick Look routes on it, and
    /// a poster written under the print's `.glb` name previews no better than
    /// the mesh did.
    func meshPosterFile(for entry: LibraryEntry) async -> (url: URL, title: String)? {
        guard let materializer, let backend = hosts.backend(for: entry.hostID) else {
            return nil
        }
        let stem = MeshExport.filename(entry.print.filename, format: "png")
        let url = await materializer.url(for: entry, named: stem) {
            try? await backend.thumbnail(entry.print.filename, size: 1024,
                                         trashed: entry.print.trashedAt != nil)
        }
        return url.map { ($0, entry.print.displayName) }
    }
}
