import Foundation

/// The gallery: listing, mutating, trashing, exporting and playing prints.
public protocol MoldGalleryBackend: Sendable {
    /// Pass the previous `etag` to let the host answer `.notModified`.
    func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]>
    func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]>
    func patch(_ filename: String, with patch: GalleryPatch) async throws
    /// Replay-safe by `operationId`, so a retry cannot double-apply.
    func mutate(_ mutation: GalleryBulkMutation) async throws
    func trash(_ filenames: [String]) async throws
    func restoreFromTrash(_ filenames: [String]) async throws
    /// Permanent. There is no undo on the host side.
    func deleteForever(_ filenames: [String]) async throws
    @discardableResult
    func importPrint(_ item: GalleryImport, as filename: String) async throws -> String
    /// The stored bytes. A trashed print lives behind the trash view, exactly
    /// as the listing does.
    func media(_ filename: String, trashed: Bool) async throws -> Data
    /// Downloaded into a caller-owned temporary file, for large Library mirrors.
    func mediaFile(_ filename: String, trashed: Bool) async throws -> URL
    /// The host's rendered poster for a print, at its own size.
    func thumbnail(_ filename: String, size: Int, trashed: Bool) async throws -> Data
    func exportOptions() async throws -> ExportOptions
    /// Converts on the machine that holds the print, so the app needs no
    /// decoder for every container mold can write.
    func export(_ filename: String, format: String) async throws -> Data
    /// The same route carrying the optional controls a mesh export takes --
    /// the geometry knobs, or a turntable's frames and fps, never both.
    func export(_ filename: String, request: MeshExportRequest) async throws -> Data
    /// A URL a player can open directly, ticketed where the host needs it.
    /// Throws rather than falling back to an unticketed URL: on a keyed host
    /// a failed ticket means the player would 401, not play silently wrong.
    func playableURL(for filename: String) async throws -> URL
}
