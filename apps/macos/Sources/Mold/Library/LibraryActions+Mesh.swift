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
}
