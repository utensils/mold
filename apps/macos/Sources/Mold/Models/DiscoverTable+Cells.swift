import AppKit
import MoldClient
import SwiftUI

// One catalog row: its two custom cells and the contextual menu that
// offers the same things they do. Split from `DiscoverTable.swift` past
// the file-size advisory -- nothing here is `private`, because `private`
// does not cross a file boundary even within one type.
extension DiscoverTable {
    /// The row's own controls a second way. Open Page is a `Button` that
    /// hands the URL to the workspace rather than a `Link`: a menu is one
    /// list, and a `Link` is the one row `RowActionMenu` could not draw.
    func perform(_ item: DiscoverRow.Item, on entry: CatalogEntry) {
        switch item {
        case .details:
            detailEntry = entry
        case .install:
            Task { await downloads.install(entry.id, on: host) }
        case let .openPage(url):
            NSWorkspace.shared.open(url)
        }
    }

    func nameCell(_ entry: CatalogEntry) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 4) {
                Text(entry.name).lineLimit(1)
                if let badge = Self.nsfwBadge(entry) {
                    Text(badge).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                }
            }
            if let author = entry.author {
                Text(author).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .contentShape(Rectangle())
        // `.onTapGesture(count: 2)` eats the single click the `Table`'s own
        // `selection:` needs, so double-clicking a row opened its details
        // while never selecting it -- and the Details… item below reads the
        // entry it was built for, not the selection, precisely because of
        // that. `.simultaneousGesture` leaves the single click alone (M8's
        // own `List`-row finding).
        .simultaneousGesture(TapGesture(count: 2).onEnded { detailEntry = entry })
    }

    @ViewBuilder func stateCell(_ entry: CatalogEntry) -> some View {
        switch DiscoverRow.resolve(entry) {
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.secondary)
        case .install:
            Button("Install") { Task { await downloads.install(entry.id, on: host) } }
                .buttonStyle(.bordered).controlSize(.small)
        case let .unsupported(pageURL):
            HStack(spacing: 6) {
                Text("Not supported").font(.caption).foregroundStyle(.secondary)
                if let pageURL { Link("Open Page", destination: pageURL).font(.caption) }
            }
        }
    }
}
