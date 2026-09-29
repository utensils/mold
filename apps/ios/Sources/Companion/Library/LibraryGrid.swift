import MoldClient
import SwiftUI

/// The day-sectioned grid. Columns come from a scaled minimum tile width, so
/// larger text means larger, fewer tiles -- never clipped labels. A pinch
/// walks the five sizes live (`TileSize.pinched`), keeping the print you were
/// looking at in place.
struct LibraryGrid: View {
    let sections: [LibrarySection]
    @Binding var tile: TileSize
    @Binding var position: LibraryScrollPosition
    let returnToPrint: PrintID?
    let selecting: Bool
    @Binding var selection: Set<PrintID>
    let trashed: Bool
    let zoom: Namespace.ID
    let visible: [LibraryEntry]

    /// The accessibility audit's lazy-grid exemption keys on this.
    static let dayHeader = "day-header"

    @Environment(HostStore.self) private var hosts
    @ScaledMetric(relativeTo: .body) private var scale: CGFloat = 1
    /// The size a pinch began at; `nil` between pinches.
    @State private var pinchStart: TileSize?
    /// `anchor` as the pinch began: the print to bring back into place.
    @State private var pinchAnchor: PrintID?

    var body: some View {
        let minimum = tile.basePoints * scale
        let showsHost = Set(visible.flatMap(\.hostNames)).count > 1
        ScrollViewReader { reader in
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: minimum, maximum: minimum * 2), spacing: 3)],
                          spacing: 3) {
                    ForEach(sections) { section in
                        Section {
                            ForEach(section.items) { entry in
                                cell(entry, points: minimum * 1.25, showsHost: showsHost)
                                    .id(entry.id)
                            }
                        } header: {
                            if let day = section.day {
                                Text(LibraryGrouping.title(for: day))
                                    .font(.headline)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    // Opaque: prints scroll under a pinned header,
                                    // and a translucent bar let them show through
                                    // the text.
                                    .background(Color(uiColor: .systemBackground))
                                    .accessibilityAddTraits(.isHeader)
                                    .accessibilityIdentifier(Self.dayHeader)
                            }
                        }
                    }
                }
                .scrollTargetLayout()
            }
            .scrollPosition(id: Binding(get: { position.id }, set: { position.report($0) }), anchor: .top)
            .gesture(PinchRecognizer(changed: pinched, ended: { pinchStart = nil; pinchAnchor = nil }))
            .sensoryFeedback(.selection, trigger: tile)
            .accessibilityRotor("Days") {
                ForEach(sections.filter { $0.day != nil }) { section in
                    AccessibilityRotorEntry(Text(LibraryGrouping.title(for: section.day ?? .now)), id: section.id)
                }
            }
            .accessibilityRotor("Favourites") {
                ForEach(visible.filter(\.print.isFavorite)) { entry in
                    AccessibilityRotorEntry(Text(entry.spokenName), id: entry.id)
                }
            }
            .onChange(of: returnToPrint) { _, id in
                guard let id else { return }
                // The navigation transition can recreate the grid at an earlier
                // offset even while its scroll binding still holds a valid print.
                // Ask the live scroll view to reveal the tile once the viewer ends.
                Task { @MainActor in
                    reader.scrollTo(id, anchor: .top)
                    position.report(id)
                }
            }
        }
    }

    @ViewBuilder private func cell(_ entry: LibraryEntry, points: CGFloat, showsHost: Bool) -> some View {
        let tile = PrintTile(entry: entry, points: points, trashed: trashed,
                             selecting: selecting, selected: selection.contains(entry.id),
                             showsHost: showsHost, drawsBadges: false)
        let tileView = tile.matchedTransitionSource(id: entry.id, in: zoom)
        if selecting {
            Button { toggle(entry.id) } label: { tileView }
                .buttonStyle(.plain)
                .overlay { tile.badges }
        } else {
            NavigationLink(value: entry.id) { tileView }
                .buttonStyle(.plain)
                .overlay { tile.badges }
                // iPad: drag the print itself out -- to Files, Photos, another
                // app, or a picture well -- fetched only when dropped.
                .draggable(DraggedPrint(entry, backend: hosts.backend(for: entry.hostID))) {
                    PrintThumbnail(entry: entry, points: 120, trashed: trashed).frame(width: 120, height: 120)
                }
                .contextMenu { PrintMenu(entries: [entry], trashed: trashed) } preview: {
                    PrintThumbnail(entry: entry, points: 360, trashed: trashed)
                        .frame(width: 360, height: 360)
                }
        }
    }

    /// Walks the sizes live as the fingers move, keeping the top print put.
    private func pinched(_ scale: CGFloat) {
        let start = pinchStart ?? tile
        if pinchStart == nil {
            pinchStart = start
            pinchAnchor = position.id
        }
        let next = TileSize.pinched(from: start, magnification: scale, current: tile)
        guard next != tile else { return }
        withAnimation(.snappy(duration: 0.25)) { tile = next }
        // Bring the print that was on top back to the top once the new
        // layout exists (writing the same id in the same pass does nothing).
        if let keep = pinchAnchor {
            Task { @MainActor in
                position.reset()
                position.report(keep)
            }
        }
    }

    private func toggle(_ id: PrintID) {
        if selection.contains(id) { selection.remove(id) } else { selection.insert(id) }
    }
}

/// A navigation push may make SwiftUI report no visible scroll target while
/// the grid is covered. Keep the last real print so popping the viewer restores
/// it; an explicit shelf or search change starts from the top instead.
struct LibraryScrollPosition {
    private(set) var id: PrintID?

    mutating func report(_ visible: PrintID?) {
        if let visible { id = visible }
    }

    mutating func reset() { id = nil }
}
