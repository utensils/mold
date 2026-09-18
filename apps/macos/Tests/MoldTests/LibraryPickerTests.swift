import Foundation
import MoldClient
import Testing

@testable import Mold

/// `LibraryPicker.rows` is what "From Library…" (M8 design, decision 5)
/// offers: pictures only, never trashed, newest first, optionally folded on
/// a query -- pure, so the sheet's grid needs no store to test.
@MainActor
struct LibraryPickerTests {
    private func host(_ name: String = "workstation") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// `FakeFixtures.print` has no `format` or `trashed_at`; this is the same
    /// decode-from-JSON shape with the two extra fields this suite needs.
    private func print(_ filename: String, format: String? = nil, trashedAt: UInt64? = nil) -> GalleryPrint {
        let json = """
        {"filename": "\(filename)", "metadata": {"prompt": "a picture"}, "timestamp": 1000,
         "format": \(format.map { "\"\($0)\"" } ?? "null"),
         "trashed_at": \(trashedAt.map(String.init) ?? "null")}
        """
        return try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
    }

    private func entry(_ filename: String, format: String? = nil, trashedAt: UInt64? = nil) -> LibraryEntry {
        LibraryEntry(host: host(), print: print(filename, format: format, trashedAt: trashedAt))
    }

    @Test func aClipIsExcluded() {
        let entries = [entry("a.png"), entry("b.mp4", format: "mp4")]
        #expect(LibraryPicker.rows(entries, query: "").map(\.print.filename) == ["a.png"])
    }

    @Test func aMeshIsExcluded() {
        let entries = [entry("a.png"), entry("b.glb", format: "glb")]
        #expect(LibraryPicker.rows(entries, query: "").map(\.print.filename) == ["a.png"])
    }

    @Test func aTrashedPictureIsExcluded() {
        let entries = [entry("a.png"), entry("b.png", trashedAt: 2000)]
        #expect(LibraryPicker.rows(entries, query: "").map(\.print.filename) == ["a.png"])
    }

    @Test func aFoldedQueryMatchesCaseInsensitively() {
        let entries = [entry("a.png"), entry("sunset.png")]
        #expect(LibraryPicker.rows(entries, query: "SUNSET").map(\.print.filename) == ["sunset.png"])
    }

    @Test func orderIsPreservedNotResorted() {
        // The store's `items` are already newest first -- `rows` must not
        // re-sort into filename order.
        let entries = [entry("z.png"), entry("a.png")]
        #expect(LibraryPicker.rows(entries, query: "").map(\.print.filename) == ["z.png", "a.png"])
    }
}
