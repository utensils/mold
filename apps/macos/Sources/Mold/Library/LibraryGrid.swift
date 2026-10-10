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
    /// What a tile's menu needs that a tile does not know: the shelves it
    /// could be filed into, the one being shown, and how much is in the trash.
    let shelves: [CollectionShelf]
    let enclosingShelf: CollectionShelf?
    let trashCount: Int
    @Binding var selection: LibraryCursor.Selection
    let viewport: LibraryViewport
    let returnToPrint: PrintID?
    let onReturnRestored: () -> Void
    let onOpen: (PrintID) -> Void
    var newMediaVisit: LibraryNewMedia.Visit?

    /// Not `private`: the cursor the keyboard drives is built in
    /// `+Selection`, and `private` does not cross a file boundary.
    @State private var width: CGFloat = 0
    @State private var layout = JustifiedLibraryLayout()
    @State private var nativePosition = ScrollPosition()
    @State var keyboardReveal: PrintID?
    @State private var visibleIDs: Set<PrintID> = []
    /// The grid must HOLD key focus, or its arrows, Return and Space never
    /// reach it -- including when the viewer closes and hands the cursor back.
    @FocusState private var focused: Bool

    var body: some View {
        // Context menus are built while the grid redraws. Resolve the bulk
        // selection once, not once per selected cell (quadratic for Select All).
        let selectedTargets = selection.items.isEmpty ? []
            : entries.filter { selection.items.contains($0.id) }
        // One selection-wide snapshot, shared by every selected tile. Menu
        // rows and share objects still materialize only upon opening a menu.
        let selectedPlan = selectedTargets.isEmpty ? nil : LibraryMenu(
            targets: selectedTargets, scope: scope, actions: actions, shelves: shelves,
            enclosingShelf: enclosingShelf, trashCount: trashCount, open: {}).plan
        let laidSections = layout.resolve(sections, width: width, targetHeight: edge)
        return ScrollViewReader { scroller in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: JustifiedLayout.gap) {
                    ForEach(laidSections) { laid in
                        let section = laid.source
                        Section {
                            ForEach(laid.rows) { row in
                                HStack(spacing: JustifiedLayout.gap) {
                                    ForEach(row.items, id: \.index) { item in
                                        cell(section.items[item.index], selectedTargets: selectedTargets, selectedPlan: selectedPlan)
                                            .frame(width: item.width, height: row.height)
                                    }
                                }
                                .frame(height: row.height)
                                .id(row.id)
                            }
                        } header: {
                            // A section with no day is the whole list in one
                            // piece, under an order days cannot describe.
                            if let day = section.day { header(day, count: section.items.count) }
                        }
                    }
                }
                .scrollTargetLayout()
            }
            .scrollPosition($nativePosition)
            .onScrollGeometryChange(for: CGFloat.self) { $0.contentOffset.y } action: { _, offset in
                viewport.report(offset: offset)
            }
            .task(id: width > 0) {
                if width > 0, returnToPrint != nil {
                    nativePosition.scrollTo(y: viewport.uncover())
                    onReturnRestored()
                }
            }
            .onScrollTargetVisibilityChange(idType: PrintID.self, threshold: 0.1) { ids in
                visibleIDs = Set(ids)
            }
            .onChange(of: keyboardReveal) { _, lead in
                guard let lead, !visibleIDs.contains(rowAnchor(lead)) else { return }
                withAnimation(.snappy) { scroller.scrollTo(rowAnchor(lead), anchor: .center) }
            }
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { newWidth in
                let anchor = firstVisible
                width = newWidth
                if let anchor { Task { @MainActor in
                    await Task.yield()
                    scroller.scrollTo(rowAnchor(anchor), anchor: .top)
                } }
            }
            .onChange(of: edge) { _, _ in
                if let anchor = firstVisible { Task { @MainActor in
                    await Task.yield()
                    scroller.scrollTo(rowAnchor(anchor), anchor: .top)
                } }
            }
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

    private var firstVisible: PrintID? {
        entries.first { visibleIDs.contains($0.id) }?.id
    }

    private func rowAnchor(_ id: PrintID) -> PrintID {
        for section in layout.resolve(sections, width: width, targetHeight: edge) {
            for row in section.rows where row.items.contains(where: { section.source.items[$0.index].id == id }) {
                return row.id
            }
        }
        return id
    }

    /// Geometry, rather than a guessed column count, owns vertical arrows.
    var cursor: LibraryCursor {
        LibraryCursor(rows: layout.resolve(sections, width: width, targetHeight: edge).flatMap { section in
            section.rows.map { row in
                row.items.map { (section.source.items[$0.index].id, $0.x + $0.width / 2) }
            }
        })
    }

    @ViewBuilder private func cell(_ entry: LibraryEntry,
                                   selectedTargets: [LibraryEntry],
                                   selectedPlan: LibraryMenuPlan?) -> some View {
        if let host = hosts.first(where: { $0.id == entry.hostID }) {
            LibraryCell(
                entry: entry, host: host, edge: edge,
                isSelected: selection.items.contains(entry.id),
                isLead: selection.lead == entry.id,
                showsHostBadge: showsHostBadges,
                fresh: !scope.isTrash && (newMediaVisit?.contains(entry.print.filename) ?? false)
            )
            .onTapGesture(count: 2) { onOpen(entry.id) }
            .onTapGesture { click(entry) }
            .draggable(actions.draggable(entry))
            .libraryMenu(
                LibraryMenu(targets: selection.items.contains(entry.id) ? selectedTargets : [entry],
                            scope: scope, actions: actions,
                            shelves: shelves, enclosingShelf: enclosingShelf,
                            trashCount: trashCount, open: { onOpen(entry.id) }),
                snapshot: selection.items.contains(entry.id) ? selectedPlan : nil)
        }
    }

    private func header(_ day: Date, count: Int) -> some View {
        HStack {
            Text(LibraryGrouping.title(for: day)).font(.headline)
            Text(count.formatted())
                .font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
