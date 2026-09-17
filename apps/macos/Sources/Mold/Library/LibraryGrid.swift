import MoldClient
import MoldStyle
import SwiftUI

/// The day-sectioned grid, with selection and the keyboard.
struct LibraryGrid: View {
    let sections: [LibrarySection]
    let hosts: [MoldHost]
    let edge: CGFloat
    let showsHostBadges: Bool
    let scope: LibraryScope
    let actions: LibraryActions
    let entries: [LibraryEntry]
    @Binding var selection: LibraryCursor.Selection
    let onOpen: (PrintID) -> Void

    /// Not `private`: the cursor the keyboard drives is built in
    /// `+Selection`, and `private` does not cross a file boundary.
    @State var columns = 1
    /// The grid must HOLD key focus, or its arrows, Return and Space never
    /// reach it -- including when the viewer closes and hands the cursor back.
    @FocusState private var focused: Bool

    var body: some View {
        ScrollViewReader { scroller in
            ScrollView {
                LazyVGrid(columns: gridColumns, alignment: .leading, spacing: 16) {
                    ForEach(sections) { section in
                        Section {
                            ForEach(section.items) { cell($0) }
                        } header: {
                            // A section with no day is the whole list in one
                            // piece, under an order days cannot describe.
                            if section.day != nil { header(section) }
                        }
                    }
                }
                .padding(16)
            }
            .onChange(of: selection.lead) { _, lead in
                guard let lead else { return }
                withAnimation(.snappy) { scroller.scrollTo(lead, anchor: .center) }
            }
        }
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width in
            // The cursor needs the real column count for arrow keys to land
            // where the eye expects.
            columns = max(Int((width - 32 + 12) / (edge + 12)), 1)
        }
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        // ⌘A here, not in `MoldCommands`: Edit already carries the system's
        // Select All, which SwiftUI never routes to a focusable grid
        // (`onCommand` read back disabled, M7 S6), and a second item would
        // be a duplicate -- so the grid answers the key. Proven by PID.
        .onKeyPress(keys: ["a"]) { press in
            guard press.modifiers.contains(.command) else { return .ignored }
            let ids = entries.map(\.id)
            selection = LibraryCursor.Selection(items: Set(ids), anchor: ids.first, lead: selection.lead ?? ids.first)
            return .handled
        }
        // Claimed after a yield rather than in `onAppear`: a `@FocusState`
        // written in the pass that inserts the view is dropped.
        .task { await Task.yield(); focused = true }
        // One handler, because what these keys mean depends on what is HELD
        // with them -- and `onKeyPress(_ key:)` matches its key whatever that
        // is. Space is Quick Look everywhere else on the Mac; Return opens in
        // place, which is the app's own viewer and the only one that plays a
        // clip with the machine's media ticket. See `LibraryGridKeys`.
        .onKeyPress(keys: LibraryGridKeys.keys) { press in
            perform(LibraryGridKeys.action(for: press.key, modifiers: press.modifiers))
        }
    }

    private var gridColumns: [GridItem] {
        [GridItem(.adaptive(minimum: edge, maximum: .infinity), spacing: 12)]
    }

    /// Not `private`: `+Selection` is what drives it.
    var cursor: LibraryCursor {
        LibraryCursor(sections: sections, columns: columns)
    }

    @ViewBuilder private func cell(_ entry: LibraryEntry) -> some View {
        if let host = hosts.first(where: { $0.id == entry.hostID }) {
            LibraryCell(
                entry: entry, host: host, edge: edge,
                isSelected: selection.items.contains(entry.id),
                isLead: selection.lead == entry.id,
                showsHostBadge: showsHostBadges
            )
            .id(entry.id)
            .onTapGesture(count: 2) { onOpen(entry.id) }
            .onTapGesture { click(entry) }
            .draggable(actions.draggable(entry))
            .contextMenu {
                LibraryMenu(targets: targets(for: entry), scope: scope, actions: actions,
                            open: { onOpen(entry.id) })
            }
        }
    }

    private func header(_ section: LibrarySection) -> some View {
        HStack {
            Text(section.day.map { LibraryGrouping.title(for: $0) } ?? "").font(.headline)
            Text(section.items.count.formatted())
                .font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.top, 8)
        .background(.bar)
    }
}
