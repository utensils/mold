import AppKit
import Foundation

/// The pasteboard, as a picture well needs it.
///
/// A trivial composition of what the wells already do -- bytes in, base64 out
/// through `PictureImport` -- so Paste offers nothing the app could not
/// already do from a file, and is hidden outright when the pasteboard holds
/// no picture.
enum PicturePaste {
    /// Whether Paste is worth offering at all right now.
    @MainActor static var hasPicture: Bool {
        NSPasteboard.general.canReadObject(forClasses: [NSImage.self], options: nil)
    }

    /// Whatever is on the pasteboard, conformed to what the WELL accepts and
    /// encoded off the main actor exactly as an imported file is.
    static func read(_ data: Data?, accepting: Set<String>) async throws -> ImportedPicture? {
        guard let data else { return nil }
        // Named for what the pasteboard actually hands over. `conform` renames
        // only what it TRANSCODES, so calling these bytes a PNG shipped TIFF
        // under a PNG name to every well that reads TIFF.
        return try await PictureImport.conforming(
            data, name: "Pasted picture.tiff", accepting: accepting)
    }

    /// The pasteboard's own bytes, read on the main actor because `NSPasteboard`
    /// is not thread-safe.
    @MainActor static func pasteboardData() -> Data? {
        guard let image = NSPasteboard.general.readObjects(
            forClasses: [NSImage.self], options: nil)?.first as? NSImage,
            let tiff = image.tiffRepresentation
        else { return nil }
        return tiff
    }
}
