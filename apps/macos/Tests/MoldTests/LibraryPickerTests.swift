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
    @Test func allMachinesIncludesBothHostsAndSearchesCopies() {
        let first = host("alpha"), second = host("beta")
        var merged = LibraryEntry(host: first, print: print("shared.png"))
        merged.copies = [LibraryEntry(host: second, print: print("shared.png"))]
        #expect(LibraryPicker.rows([merged], query: "beta").count == 1)
        let selected = LibraryPicker.rows([merged], query: "", machineID: second.id)
        #expect(selected.first?.hostID == second.id)
        #expect(selected.first?.copies.isEmpty == true)
    }

    @Test func removedMachinesCannotRemainActionTargets() {
        let first = host("alpha"), second = host("beta")
        var merged = LibraryEntry(host: first, print: print("shared.png"))
        merged.copies = [LibraryEntry(host: second, print: print("shared.png"))]
        let rows = LibraryPicker.rows([merged], query: "", availableHosts: [second.id])
        #expect(rows.first?.hostID == second.id)
    }

    @Test func sheetFitsSmallScreensAndGrowsOnDesktop() {
        let small = LibraryPicker.sheetSize(available: CGSize(width: 800, height: 600))
        #expect(small.width <= 768 && small.height <= 568)
        let large = LibraryPicker.sheetSize(available: CGSize(width: 1800, height: 1100))
        #expect(large.width >= 1000 && large.height >= 700)
    }

    @Test func hiddenReplicaProtectsLogicalPrintBeforeHostProjection() {
        let first = host("alpha"), second = host("beta")
        var hidden = GalleryPrint.Mutable(print("shared.png"))
        hidden.collections = ["private"]
        var merged = LibraryEntry(host: first, print: hidden.build())
        merged.copies = [LibraryEntry(host: second, print: print("shared.png"))]
        #expect(LibraryPicker.rows([merged], query: "", machineID: second.id,
            hiddenCollectionIDs: [first.id: ["private"]]).isEmpty)
    }

    @Test func allMachinesPrefersReachableCopyWithoutHidingOfflineInventory() {
        let offline = host("offline"), online = host("online")
        var merged = LibraryEntry(host: offline, print: print("shared.png"))
        let reachable = LibraryEntry(host: online, print: print("shared.png"))
        merged.copies = [reachable]
        let rows = LibraryPicker.rows([merged], query: "", reachableHosts: [online.id])
        #expect(rows.first?.id == reachable.id)
        #expect(rows.first?.copies.map(\.hostID) == [offline.id])
        let scoped = LibraryPicker.rows([merged], query: "", machineID: offline.id,
                                        reachableHosts: [online.id])
        #expect(scoped.first?.hostID == offline.id)
        #expect(scoped.first?.copies.isEmpty == true)
        let unavailable = LibraryPicker.rows([merged], query: "", reachableHosts: [])
        #expect(unavailable.first?.hostID == offline.id)
        #expect(unavailable.count == 1)
    }

    @Test func defaultPickerMinimumIsSpaciousButClampedToScreen() {
        let desktop = LibraryPicker.minimumSheetSize(available: CGSize(width: 1800, height: 1100))
        #expect(desktop.width >= 960)
        #expect(desktop.height >= 640)
        let compact = LibraryPicker.minimumSheetSize(available: CGSize(width: 800, height: 600))
        #expect(compact.width <= 768)
        #expect(compact.height <= 568)
    }

}
