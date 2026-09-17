import Foundation

public extension PrintKind {
    /// What a stored file is, from its own EXTENSION and nothing else -- the
    /// question a surface has to ask when a filename is all it holds.
    ///
    /// A `BatchResult` (`BatchStatus.swift`) carries `filename`, `seed`,
    /// `generationTimeMs` and `gpu`: no `format`, no `frames`. So the Generate
    /// canvas has no other input, and one authority genuinely cannot serve
    /// both it and `GalleryPrint.kind`, which answers the richer question from
    /// the serialized `format` (and, where the host sent it, `frames`).
    ///
    /// The two are named apart deliberately. This one asks "what can PLAY
    /// this", so the clip set is the containers a player can open. mold also
    /// delivers a clip as GIF, APNG or WebP where the recipe advertises them;
    /// those decode as images and are still pictures HERE, whatever the
    /// Library calls them, because handing one to `AVPlayer` shows nothing at
    /// all. Where `frames` IS available the caller should prefer it -- this
    /// table is the fallback, and the one place the extensions live.
    init(playbackOf filename: String) {
        switch (filename as NSString).pathExtension.lowercased() {
        case "mp4", "webm", "mov", "m4v": self = .clip
        case "glb": self = .mesh
        default: self = .picture
        }
    }

    /// The containers a player can open, exposed so `GalleryPrint.kind` can
    /// fall back to the same table when the wire carried no `format`.
    static let playableClipExtensions: Set<String> = ["mp4", "webm", "mov", "m4v"]
}

public extension BatchResult {
    /// What this child produced, as the CANVAS has to ask it: by container.
    /// `nil` before the host has named a file.
    var playbackKind: PrintKind? { filename.map(PrintKind.init(playbackOf:)) }
}
