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

    /// PNG bytes for whatever is on the pasteboard, conformed and encoded off
    /// the main actor exactly as an imported file is.
    static func read(_ data: Data?) async throws -> ImportedPicture? {
        guard let data else { return nil }
        return try await Task.detached(priority: .userInitiated) {
            try PictureImport.conform(
                data, name: "Pasted picture.png", accepting: PictureImport.engineReadable)
        }.value
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
