import Foundation
import Testing

@testable import MoldClient

/// Which live frames are this app's own edit coming back.
///
/// **Fails today**: `LibraryStore+Live.swift:38` skips every frame from a
/// machine with anything queued, so a render landing during a star is
/// discarded -- there is no per-row decision to test.
@MainActor
struct GalleryEchoTests {
    private let plato = UUID()

    private func pending(_ filenames: [String]) -> [MutationOutbox.Entry] {
        var outbox = MutationOutbox()
        return outbox.enqueue(PrintEdit(change: .favorite(true),
                                        targets: [plato: filenames]))
    }

    @Test func aFrameNamingARowWeAreSendingIsOurOwnEcho() {
        var echo = GalleryEcho()
        let ours = MoldEvent.Gallery.updated(filename: "star.png", row: nil)
        let skipped = echo.isEcho(ours, on: plato, pending: pending(["star.png"]))
        #expect(skipped)
    }

    @Test func aFrameNamingAnyOtherRowIsTheMachineTellingUsSomethingNew() {
        var echo = GalleryEcho()
        let landed = MoldEvent.Gallery.added(filename: "new.png", row: nil)
        let skipped = echo.isEcho(landed, on: plato, pending: pending(["star.png"]))
        let owed = echo.takeStale(plato)
        #expect(!skipped)
        #expect(!owed)
    }

    @Test func aFrameWithNothingQueuedIsNeverAnEcho() {
        var echo = GalleryEcho()
        let change = MoldEvent.Gallery.trashed(filename: "star.png")
        let skipped = echo.isEcho(change, on: plato, pending: [])
        #expect(!skipped)
    }

    /// The one gap the row rule leaves: another client touching the same row
    /// inside our window. The machine is remembered so the drain re-lists it,
    /// and remembered ONCE.
    @Test func aSkippedFrameLeavesItsMachineOwedARelist() {
        var echo = GalleryEcho()
        _ = echo.isEcho(.updated(filename: "star.png", row: nil), on: plato,
                        pending: pending(["star.png"]))

        let owed = echo.takeStale(plato)
        let owedAgain = echo.takeStale(plato)
        #expect(owed)
        #expect(!owedAgain)
    }

    @Test func aCollectionsFrameNamesNoRowAndSoIsNeverAnEcho() {
        var echo = GalleryEcho()
        let skipped = echo.isEcho(.collectionsChanged, on: plato, pending: pending(["a.png"]))
        #expect(GalleryEcho.names(.collectionsChanged) == nil)
        #expect(!skipped)
    }
}
