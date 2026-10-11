import MoldClient
import SwiftUI

/// The grid and its footer caption, split from `LibraryPickerSheet.swift`
/// purely for size.
extension LibraryPickerSheet {
    enum PictureAction: Hashable { case use, preview }

    static func pictureActions(isFetching: Bool) -> [RowAction<PictureAction>] {
        [RowAction(kind: .use, title: "Use Picture", isDisabled: isFetching),
         RowAction(kind: .preview, title: "Preview Picture")]
    }

    /// "1 picture" / "N pictures" -- pure, so the footer's wording is tested
    /// with no view.
    static func caption(count: Int) -> String {
        count == 1 ? "1 picture" : "\(count) pictures"
    }

    @ViewBuilder var grid: some View {
        GeometryReader { geometry in
            let columns = max(1, Int((geometry.size.width - 14) / 170))
            ScrollViewReader { proxy in
                ScrollView {
                    if rows.isEmpty {
                        emptyState.padding(40)
                    } else {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 10)], spacing: 10) {
                            ForEach(rows) { entry in tile(entry).id(entry.id) }
                        }
                        .padding(12)
                    }
                }
                .focusable()
                .onKeyPress(.leftArrow) { moveSelection(-1, proxy: proxy) }
                .onKeyPress(.rightArrow) { moveSelection(1, proxy: proxy) }
                .onKeyPress(.upArrow) { moveSelection(-columns, proxy: proxy) }
                .onKeyPress(.downArrow) { moveSelection(columns, proxy: proxy) }
                .onKeyPress(.space) {
                    guard let entry = rows.first(where: { $0.id == selected }) else { return .ignored }
                    showPreview(entry)
                    return .handled
                }
            }
        }
    }

    private func moveSelection(_ offset: Int, proxy: ScrollViewProxy) -> KeyPress.Result {
        let entries = rows
        guard !entries.isEmpty else { return .ignored }
        let current = entries.firstIndex(where: { $0.id == selected })
        let next = current.map { min(entries.count - 1, max(0, $0 + offset)) } ?? 0
        selected = entries[next].id
        proxy.scrollTo(entries[next].id)
        return .handled
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
            VStack {
            LibraryCell(entry: entry, host: host, edge: 160,
                        isSelected: selected == entry.id, isLead: selected == entry.id,
                        showsHostBadge: hosts.hosts.count > 1)
                // Double-tap FIRST in the chain so it wins over the plain
                // tap beneath it, the way `PromptWandPopover`'s row learned
                // to (M8 decision 9) -- reversed, a single click would
                // consume the gesture and a double-click could never land.
                .onTapGesture(count: 2) { selected = entry.id; use() }
                .onTapGesture { selected = entry.id }
                .rowActionMenu(Self.pictureActions(isFetching: isFetching)) { action in
                    switch action {
                    case .use: selected = entry.id; use()
                    case .preview: showPreview(entry)
                    }
                }
                .accessibilityAction(named: "Use Picture") { selected = entry.id; use() }
                .help("Click to select, double-click to use, or press Space to preview.")
                Button("View") { showPreview(entry) }.font(.caption)
                    .accessibilityLabel("View \(entry.print.displayName)")
            }
        }
    }
}
