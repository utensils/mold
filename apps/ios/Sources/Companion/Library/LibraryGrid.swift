import MoldClient
import SwiftUI

/// The day-sectioned grid. Columns come from a scaled minimum tile width, so
/// larger text means larger, fewer tiles -- never clipped labels. A pinch
/// walks the five sizes live (`TileSize.pinched`), keeping the print you were
/// looking at in place.
struct LibraryGrid: View {
    let sections: [LibrarySection]
    @Binding var tile: TileSize
    let selecting: Bool
    @Binding var selection: Set<PrintID>
    let trashed: Bool
    let zoom: Namespace.ID
    let visible: [LibraryEntry]

    @Environment(HostStore.self) private var hosts
    @ScaledMetric(relativeTo: .body) private var scale: CGFloat = 1
    /// The size a pinch began at; `nil` between pinches.
    @State private var pinchStart: TileSize?
    /// The print at the top of the screen, kept there across size changes.
    @State private var anchor: PrintID?

    var body: some View {
        let minimum = tile.basePoints * scale
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: minimum, maximum: minimum * 2), spacing: 3)],
                      spacing: 3, pinnedViews: [.sectionHeaders]) {
                ForEach(sections) { section in
                    Section {
                        ForEach(section.items) { entry in
                            cell(entry, points: minimum * 1.25)
                                .id(entry.id)
                        }
                    } header: {
                        if let day = section.day {
                            Text(LibraryGrouping.title(for: day))
                                .font(.headline)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(.bar)
                                .accessibilityAddTraits(.isHeader)
                        }
                    }
                }
            }
            .scrollTargetLayout()
        }
        .scrollPosition(id: $anchor, anchor: .top)
        .gesture(PinchRecognizer(changed: pinched, ended: { pinchStart = nil }))
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
    }

    @ViewBuilder private func cell(_ entry: LibraryEntry, points: CGFloat) -> some View {
        let tileView = PrintTile(entry: entry, points: points, trashed: trashed,
                                 selecting: selecting, selected: selection.contains(entry.id),
                                 showsHost: Set(visible.flatMap(\.hostNames)).count > 1)
            .matchedTransitionSource(id: entry.id, in: zoom)
        if selecting {
            Button { toggle(entry.id) } label: { tileView }
                .buttonStyle(.plain)
        } else {
            NavigationLink(value: entry.id) { tileView }
                .buttonStyle(.plain)
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
        pinchStart = start
        let next = TileSize.pinched(from: start, magnification: scale)
        guard next != tile else { return }
        let keep = anchor
        withAnimation(.snappy(duration: 0.25)) { tile = next }
        anchor = keep
    }

    private func toggle(_ id: PrintID) {
        if selection.contains(id) { selection.remove(id) } else { selection.insert(id) }
    }
}
