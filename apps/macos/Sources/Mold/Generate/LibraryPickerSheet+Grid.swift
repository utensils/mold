import MoldClient
import SwiftUI

/// The grid and its footer caption, split from `LibraryPickerSheet.swift`
/// purely for size.
extension LibraryPickerSheet {
    /// "1 picture" / "N pictures" -- pure, so the footer's wording is tested
    /// with no view.
    static func caption(count: Int) -> String {
        count == 1 ? "1 picture" : "\(count) pictures"
    }

    @ViewBuilder var grid: some View {
        ScrollView {
            if rows.isEmpty {
                emptyState.padding(40)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 10)], spacing: 10) {
                    ForEach(rows) { entry in tile(entry) }
                }
                .padding(12)
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "No pictures", systemImage: "photo",
            description: Text(query.isEmpty ? "Nothing has been rendered yet." : "Nothing matches.")
        )
    }

    /// A print whose machine has since been removed is skipped -- there is
    /// nothing left to fetch its bytes from.
    @ViewBuilder private func tile(_ entry: LibraryEntry) -> some View {
        if let host = hosts.host(entry.hostID) {
            LibraryCell(entry: entry, host: host, edge: 120,
                        isSelected: selected == entry.id, isLead: selected == entry.id,
                        showsHostBadge: hosts.hosts.count > 1)
                // Double-tap FIRST in the chain so it wins over the plain
                // tap beneath it, the way `PromptWandPopover`'s row learned
                // to (M8 decision 9) -- reversed, a single click would
                // consume the gesture and a double-click could never land.
                .onTapGesture(count: 2) { selected = entry.id; use() }
                .onTapGesture { selected = entry.id }
        }
    }
}
