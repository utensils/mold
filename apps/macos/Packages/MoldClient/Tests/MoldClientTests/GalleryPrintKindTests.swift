import Foundation
import Testing

@testable import MoldClient

// What a print IS, from the container it was delivered in.
//
// `format` on the wire is the serialized `OutputFormat`
// (`types.rs:3667-3689`), not a file extension -- so `apng` arrives as
// `"apng"` even though the file on disk is a `.png`.

private func made(_ format: String?, frames: Int? = nil) -> GalleryPrint {
    PrintFixtures.print("p", format: format, frames: frames)
}

/// **Fails today**: `isVideo` is `["mp4", "webm", "mov"]`. Every recipe with a
/// `temporal` block advertises `[Mp4, Gif, Apng, Webp]`
/// (`generation_profile.rs:2140-2152`) and the Generate inspector offers that
/// list, so picking GIF for an LTX-2 render is a supported one-click choice --
/// and the print it makes gets no player, no video export, no `is:video`
/// match and a Quick Look that opens it as a still.
@Test func aClipRenderedAsAnAnimatedStillContainerIsStillAClip() {
    for format in ["mp4", "gif", "apng"] {
        #expect(made(format).isVideo, "\(format)")
        #expect(made(format).kind == .clip, "\(format)")
    }
}

/// `webp` is the one container the server offers for BOTH a still recipe and
/// a temporal one (`generation_profile.rs:2140-2157`), so the container alone
/// cannot answer and the print's own frame count does.
@Test func aWebpAnswersFromItsFrameCountRatherThanItsContainer() {
    #expect(made("webp", frames: 97).isVideo)
    #expect(made("webp", frames: 97).kind == .clip)
    #expect(!made("webp", frames: 1).isVideo)
    #expect(!made("webp").isVideo)
    #expect(made("webp").kind == .picture)
}

/// A still stays a still, whatever it records: a single-frame render is not
/// a one-frame clip.
@Test func aPictureIsNotAClip() {
    for format in ["png", "jpeg", "webp"] {
        #expect(!made(format, frames: 1).isVideo, "\(format)")
        #expect(made(format, frames: 1).kind == .picture, "\(format)")
    }
}

/// The known limit, written down rather than left to be rediscovered: an
/// animated WebP imported through `GalleryImport` carries no `frames` and no
/// other signal exists for it on the wire, so it reads as a still. A GIF or
/// an MP4 imported the same way is right, because there the container IS the
/// answer. `GalleryPrint.isVideo`'s doc comment says what closing it takes.
@Test func anImportedAnimatedWebpIsAKnownMisclassification() {
    let imported = PrintFixtures.print("clip.webp", format: "webp", prompt: "Imported — clip.webp")
    #expect(!imported.isVideo)
    #expect(PrintFixtures.print("clip.gif", format: "gif").isVideo)
    #expect(PrintFixtures.print("clip.mp4", format: "mp4").isVideo)
}

/// And a mesh is neither. `glb` is the only stored 3-D form.
@Test func aMeshIsNeither() {
    #expect(made("glb").kind == .mesh)
    #expect(!made("glb").isVideo)
}

/// An older host, or a file the scanner adopted with no record, may send no
/// format at all. That is a still, which is what it always was -- not a
/// crash and not a clip.
@Test func aPrintWithNoFormatIsAPicture() {
    #expect(!made(nil).isVideo)
    #expect(!made(nil).isMesh)
    #expect(made(nil).kind == .picture)
}
