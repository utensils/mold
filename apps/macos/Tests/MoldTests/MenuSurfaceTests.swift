import Foundation
import MoldClient
import Testing

@testable import Mold

/// One menu model for the whole app.
///
/// **Fails today**: there are three -- `RowAction` (Settings, Queue, Models,
/// Machines), `GenerateAction`/`GenerateMenus`/`GenerateMenuItems` (every
/// Generate surface) and `LibraryMenuPlan` (the tile, the Library menu, the
/// sidebar shelf) -- plus a hand-written fourth in `QueueHoldRow` with its own
/// parallel `menuTitles` list. Each carries its own copy of "destructive last,
/// behind a divider" and its own answer for a row with nothing to offer, and
/// eleven views attach `.contextMenu` themselves.
@MainActor
struct MenuSurfaceTests {
    /// Nothing attaches a contextual menu except the one door.
    ///
    /// A source scan rather than a convention, for the same reason
    /// `NativeUATTests.everyHookReadsThroughTheOneGate` is one: the rules that
    /// make a menu right -- the order, the divider, the empty case, the dead
    /// submenu -- live in `RowAction.rendered`, and a view that builds its own
    /// `.contextMenu` gets none of them and nobody notices.
    @Test func everyMenuIsAttachedThroughTheOneModifier() throws {
        var offences: [String] = []
        let files = try sources()
        // A scan that finds no files passes for the wrong reason -- a wrong
        // walk-up, a renamed directory, a sandboxed runner
        // (`RouteEscapingContractTests`' own guard).
        #expect(files.count > 100, "the app's source directory was not found")
        for file in files {
            // `RowActionMenu.swift` IS the door: its two overloads are the
            // only `.contextMenu` in the app.
            guard file.lastPathComponent != "RowActionMenu.swift" else { continue }
            let text = try String(contentsOf: file, encoding: .utf8)
            for (number, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard code.contains(".contextMenu"), !code.hasPrefix("//") else { continue }
                offences.append("\(file.lastPathComponent):\(number + 1)")
            }
        }
        #expect(offences == [], "a contextual menu attached outside `.rowActionMenu`")
    }

    /// Every surface hands the door a declared list, so the same list can be
    /// asked what it offers without a view. These are the ones that were a
    /// `@ViewBuilder` and are now values.
    @Test func theSurfacesThatWereViewsAreListsNow() {
        let hold = QueueHoldRow.offered(
            for: .prose("GPU ran out of memory.", retryable: true), destinations: [])
        #expect(hold.map(\.kind) == [.act(.tryAgain), nil, .cancel])

        #expect(TagEditor.menu(for: "smurf").map(\.kind) == [.filter, nil, .rename, .delete])
        #expect(GenerateMenus.identityPhoto().map(\.kind)
            == [.replacePhoto, .replaceFromLibrary, .removePhoto])
        #expect(DiscoverRow.menuItems(for: FakeFixtures.catalogEntry(id: "cv:1", supported: false))
            .map(\.title) == ["Details…"])
    }

    /// The app AND the design system: `MoldStyle` imports SwiftUI too, so a
    /// `.contextMenu` helper there would be just as invisible.
    private func sources() throws -> [URL] {
        let macos = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
        return ["Sources/Mold", "Packages/MoldStyle/Sources", "Packages/MoldClient/Sources"]
            .flatMap { path -> [URL] in
                let files = FileManager.default.enumerator(at: macos.appending(path: path),
                                                           includingPropertiesForKeys: nil)
                return (files?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
            }
    }
}
