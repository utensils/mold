import Foundation
import Testing

@testable import MoldClient

/// **Fails today**: a batch child's kind was never asked -- the Generate
/// canvas decoded EVERY result with `NSImage(data:)`, and MP4 bytes are nil,
/// so a finished clip spun "Fetching your picture…" for ever with the result
/// bar and the failure summary trapped beside it (parity report §5.1).
@Test func aFinishedChildIsClassifiedByItsOwnContainer() {
    #expect(PrintKind(playbackOf: "2026-09-17-a.mp4") == .clip)
    #expect(PrintKind(playbackOf: "clip.MOV") == .clip)
    #expect(PrintKind(playbackOf: "clip.webm") == .clip)
    #expect(PrintKind(playbackOf: "shape.glb") == .mesh)
    #expect(PrintKind(playbackOf: "a.png") == .picture)
    #expect(PrintKind(playbackOf: "a.jpeg") == .picture)
    // A clip delivered as an animated container still DECODES, so it belongs
    // on the picture arm: handing it to a player would show nothing at all.
    #expect(PrintKind(playbackOf: "a.gif") == .picture)
    #expect(PrintKind(playbackOf: "a.webp") == .picture)
    // No extension at all is not a reason to refuse to draw it.
    #expect(PrintKind(playbackOf: "anonymous") == .picture)
}

@Test func aChildWithNoFileYetHasNoKind() throws {
    let listing = try MoldJSON.decoder.decode(BatchResult.self, from: Data("""
    {"filename": null, "seed": 1, "generation_time_ms": null, "gpu": null}
    """.utf8))
    #expect(listing.playbackKind == nil)

    let finished = try MoldJSON.decoder.decode(BatchResult.self, from: Data("""
    {"filename": "x.mp4", "seed": 1, "generation_time_ms": null, "gpu": null}
    """.utf8))
    #expect(finished.playbackKind == .clip)
}
