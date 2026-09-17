import MoldClient
import SwiftUI

// What File ▸ Export and File ▸ Save a Copy read. Split from the rest of
// the focused values for size once a mesh's containers joined a clip's.

/// What File ▸ Export… and File ▸ Save a Copy… act on -- the Library's own
/// selection, reached through `LibraryMenu.swift`'s own `actions.save` and
/// `actions.export`, so the menu bar item and the right-click item are the
/// same call (design S6).
struct LibraryFile: Equatable {
    let count: Int
    /// What the SINGLE selected print can also be saved as -- empty when
    /// more than one print is selected or it has no other form
    /// (`LibraryMenu.swift`'s own `exportFormats` gate).
    let exportFormats: [String]
    /// A MESH's containers, from the holding host's own advertised list. The
    /// animated ones collapse into one Turntable… here exactly as they do on
    /// the tile, so the two menus cannot offer different things.
    let meshExports: MeshExport.Split?
    let save: () -> Void
    let export: (String) -> Void

    /// What File ▸ Export draws: `(title, format)` pairs, in the host's order.
    var exportItems: [(title: String, format: String)] {
        guard let meshExports else {
            return exportFormats.map { ($0.uppercased(), $0) }
        }
        var items = meshExports.files.map { ($0.uppercased(), $0) }
        if let turntable = meshExports.animations.first {
            items.append(("Turntable…", turntable))
        }
        return items
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.count == rhs.count && lhs.exportFormats == rhs.exportFormats
            && lhs.meshExports == rhs.meshExports
    }
}
