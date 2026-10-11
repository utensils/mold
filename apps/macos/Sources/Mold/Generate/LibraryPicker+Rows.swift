import Foundation
import MoldClient

/// The grid behind "From Library…" (M8 design, decision 5): the fleet's
/// pictures, never a clip or a mesh, newest first, optionally narrowed by a
/// search query -- pure, so the sheet needs no store to test.
enum LibraryPicker {}

extension LibraryPicker {
    static func rows(
        _ entries: [LibraryEntry], query: String,
        machineID: MoldHost.ID? = nil,
        availableHosts: Set<MoldHost.ID>? = nil,
        reachableHosts: Set<MoldHost.ID> = [],
        hiddenCollectionIDs: [MoldHost.ID: Set<String>] = [:]
    ) -> [LibraryEntry] {
        // Hide a logical print before projecting physical copies: a hidden
        // replica must never leak through a visible copy on another machine.
        let tokens = LibraryEntry.fold(query).split(separator: " ")
        return entries.compactMap { entry in
            guard !entry.everyCopy.contains(where: { copy in
                copy.print.collectionList.contains {
                    hiddenCollectionIDs[copy.hostID]?.contains($0) == true
                }
            }) else { return nil }
            let scope = machineID.map { Set([$0]) } ?? availableHosts
            let allowed = scope.map { selected in availableHosts.map { selected.intersection($0) } ?? selected }
            let candidate: LibraryEntry?
            if let allowed { candidate = entry.presented(onAnyOf: allowed) }
            else { candidate = entry }
            guard let row = candidate, row.print.kind == .picture, row.print.trashedAt == nil,
                  row.everyCopy.contains(where: { copy in tokens.allSatisfy { copy.searchKey.contains($0) } }) else { return nil }
            // Prefer a known reachable origin only in the combined view.
            // Keep every copy for provenance/search, and never change an
            // explicitly selected machine or discard offline-only inventory.
            guard machineID == nil,
                  var preferred = row.everyCopy.first(where: { reachableHosts.contains($0.hostID) })
            else { return row }
            preferred.copies = row.everyCopy.filter { $0.id != preferred.id }.map { copy in
                var physical = copy
                physical.copies = []
                return physical
            }
            return preferred
        }
    }

    static func sheetSize(available: CGSize) -> CGSize {
        CGSize(width: min(1120, max(320, available.width - 32)),
               height: min(800, max(320, available.height - 32)))
    }

    static func minimumSheetSize(available: CGSize) -> CGSize {
        let maximum = sheetSize(available: available)
        return CGSize(width: min(960, maximum.width), height: min(640, maximum.height))
    }

}
