import Foundation

public extension PrintKind {
    /// What a finished file is, from its own extension.
    ///
    /// `GalleryPrint.kind` answers the same question from the wire's `format`
    /// field; a batch child carries only the stored filename, so the Generate
    /// canvas reads it here rather than guessing that every result decodes as
    /// a picture (the parity report's §5.1: `NSImage(data:)` on MP4 bytes is
    /// nil, and the pane then spun "Fetching your picture…" for ever).
    ///
    /// The clip set is deliberately the containers a PLAYER can open. mold
    /// also delivers a clip as GIF, APNG or WebP where the recipe advertises
    /// them, and those are still images to AVFoundation -- they decode, so
    /// they belong on the picture arm here whatever the Library calls them.
    init(filename: String) {
        switch (filename as NSString).pathExtension.lowercased() {
        case "mp4", "webm", "mov", "m4v": self = .clip
        case "glb": self = .mesh
        default: self = .picture
        }
    }
}

public extension BatchResult {
    /// What this child produced, or nil before the host has named a file.
    var kind: PrintKind? { filename.map(PrintKind.init(filename:)) }
}
