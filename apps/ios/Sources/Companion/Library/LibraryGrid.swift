import MoldClient
import SwiftUI

/// The day-sectioned, aspect-preserving grid. Row height scales with text. A pinch
/// walks the five sizes live (`TileSize.pinched`), keeping the print you were
/// looking at in place.
struct LibraryGrid: View {
    let sections: [LibrarySection]
    @Binding var tile: TileSize
    @Binding var position: LibraryScrollPosition
    let projection: LibraryGridProjection
    let viewport: LibraryViewport
    let returnGeneration: Int
    let selecting: Bool
    @Binding var selection: Set<PrintID>
    let trashed: Bool
    let zoom: Namespace.ID
    let visible: [LibraryEntry]

    /// The accessibility audit's lazy-grid exemption keys on this.
    static let dayHeader = "day-header"

    @Environment(HostStore.self) private var hosts
    @Environment(ThumbnailLoader.self) private var thumbnails
    @ScaledMetric(relativeTo: .body) private var scale: CGFloat = 1
    /// The size a pinch began at; `nil` between pinches.
    @State private var pinchStart: TileSize?
    /// `anchor` as the pinch began: the print to bring back into place.
    @State private var pinchAnchor: PrintID?
    @State private var layout = JustifiedLibraryLayout()
    @State private var width: CGFloat = 0
    @State private var nativePosition = ScrollPosition()
    @Namespace private var rotor
    @State private var frames: [PrintID: CGRect] = [:]
    @State private var dragSelection: LibraryDragSelection?
    @State private var deleting: [LibraryEntry]?
    @Environment(LibraryStore.self) private var library

    var body: some View {
        let minimum = tile.basePoints * scale
        let laidSections = layout.resolve(sections, width: width, targetHeight: minimum)
        ScrollViewReader { reader in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: JustifiedLayout.gap) {
                    ForEach(laidSections) { laid in
                        let section = laid.source
                        Section {
                            ForEach(laid.rows) { row in
                                HStack(spacing: JustifiedLayout.gap) {
                                    ForEach(row.items) { item in
                                        let entry = section.items[item.index]
                                        cell(entry, points: max(item.width, row.height), showsHost: projection.showsHost)
                                        .frame(width: item.width, height: row.height)
                                        .clipped()
                                        .accessibilityRotorEntry(id: entry.id, in: rotor)
                                        .background {
                                            if selecting {
                                                GeometryReader { geometry in
                                                    Color.clear.preference(key: LibraryTileFrames.self,
                                                                       value: [entry.id: geometry.frame(in: .global)])
                                                }
                                            }
                                        }
                                    }
                                }
                                .frame(height: row.height)
                                .id(row.id)
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
            .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { newWidth in
                let keep = position.id
                width = newWidth
                if let keep { Task { @MainActor in
                    await Task.yield()
                    reader.scrollTo(rowAnchor(for: keep, in: layout.resolve(sections, width: newWidth, targetHeight: minimum)), anchor: .top)
                } }
            }
            // Visibility is an observation, not a request to re-anchor every redraw.
            .onScrollTargetVisibilityChange(idType: PrintID.self, threshold: 0.1) { ids in
                let first = projection.firstVisible(ids)
                if first != position.id { position.report(first) }
            }
            .scrollPosition($nativePosition)
            .onScrollGeometryChange(for: CGFloat.self) { geometry in
                geometry.contentOffset.y
            } action: { _, offset in
                viewport.report(offset: offset)
            }
            .onPreferenceChange(LibraryTileFrames.self) { if selecting { frames = $0 } }
            .gesture(LibrarySelectionRecognizer(enabled: selecting,
                                                canStart: { hit($0) != nil },
                                                changed: sweep,
                                                ended: { dragSelection = nil }))
            .onChange(of: selecting) { _, enabled in
                dragSelection = nil
                if !enabled { frames = [:] }
            }
            .onDisappear { dragSelection = nil }
            .gesture(PinchRecognizer(changed: { pinched($0) }, ended: { pinchStart = nil; pinchAnchor = nil }))
            .onChange(of: tile) { _, next in
                let prepared = position.takeReflowAnchor()
                if let keep = pinchAnchor ?? prepared ?? position.id { Task { @MainActor in
                    await Task.yield()
                    reader.scrollTo(rowAnchor(for: keep, in: layout.resolve(sections, width: width,
                        targetHeight: next.basePoints * scale)), anchor: .top)
                } }
            }
            .sensoryFeedback(.selection, trigger: tile)
            .accessibilityRotor("Days") {
                ForEach(sections.filter { $0.day != nil }) { section in
                    AccessibilityRotorEntry(Text(LibraryGrouping.title(for: section.day ?? .now)), id: section.id)
                }
            }
            .accessibilityRotor("Favourites") {
                ForEach(projection.favorites) { entry in
                    AccessibilityRotorEntry(Text(entry.spokenName), id: entry.id, in: rotor) {
                        reader.scrollTo(rowAnchor(for: entry.id, in: laidSections), anchor: .center)
                    }
                }
            }
            .onChange(of: returnGeneration) { _, _ in
                // Restore the viewport itself, never move the opened tile to
                // the top: it may have been at the bottom of the screen.
                nativePosition.scrollTo(y: viewport.uncover())
            }
        }
        .alert("Delete Immediately?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            if let deleting {
                Button("Delete Immediately", role: .destructive) { Task { await library.deleteImmediately(deleting); self.deleting = nil } }
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: {
            Text("Copies on \(deletingMachines) will be removed for good. This can't be undone.")
        }
    }

    private var deletingMachines: String {
        Set((deleting ?? []).flatMap(\.everyCopy).compactMap { hosts.host($0.hostID)?.name }).sorted().joined(separator: ", ")
    }

    @ViewBuilder private func cell(_ entry: LibraryEntry, points: CGFloat, showsHost: Bool) -> some View {
        let tile = PrintTile(entry: entry, points: points, trashed: trashed,
                             selecting: selecting, selected: selection.contains(entry.id),
                             showsHost: showsHost, drawsBadges: false,
                             fresh: !trashed && library.unreadMedia.isUnread(entry))
        let tileView = tile.matchedTransitionSource(id: entry.id, in: zoom)
        if selecting {
            Button { toggle(entry.id) } label: { tileView }
                .buttonStyle(.plain)
                .overlay { tile.badges }
        } else {
            NavigationLink(value: entry.id) { tileView }
                .buttonStyle(.plain)
                .overlay { tile.badges }
                .simultaneousGesture(TapGesture().onEnded { viewport.cover() })
                // iPad: drag the print itself out -- to Files, Photos, another
                // app, or a picture well -- fetched only when dropped.
                .draggable(DraggedPrint(entry, backend: hosts.backend(for: entry.hostID))) {
                    PrintThumbnail(entry: entry, points: 120, trashed: trashed)
                        .environment(thumbnails)
                        .frame(width: 120, height: 120)
                }
                .contextMenu { PrintMenu(entries: [entry], trashed: trashed, requestPermanentDelete: { deleting = $0 }) } preview: {
                    // UIKit hosts this preview outside the grid's environment.
                    // The drag preview above crosses the same boundary.
                    PrintThumbnail(entry: entry, points: 360, trashed: trashed)
                        .environment(thumbnails)
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
    }

    private func rowAnchor(for id: PrintID, in sections: [JustifiedLibraryLayout.Section]) -> PrintID {
        for section in sections {
            for row in section.rows where row.items.contains(where: { section.source.items[$0.index].id == id }) {
                return row.id
            }
        }
        return id
    }

    private func hit(_ point: CGPoint) -> PrintID? {
        frames.first { $0.value.contains(point) }?.key
    }

    private func sweep(_ point: CGPoint, start: CGPoint) {
        if dragSelection == nil, let id = hit(start) {
            dragSelection = LibraryDragSelection(ids: visible.map(\.id), start: id, selection: selection)
        }
        guard let dragSelection, let id = hit(point) else { return }
        let next = dragSelection.selection(through: id)
        if next != selection { selection = next }
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
    private var reflowAnchor: PrintID?

    mutating func prepareReflow() { reflowAnchor = reflowAnchor ?? id }

    mutating func takeReflowAnchor() -> PrintID? {
        defer { reflowAnchor = nil }
        return reflowAnchor
    }

    mutating func report(_ visible: PrintID?) {
        if let visible { id = visible }
    }

    mutating func reset() { id = nil; reflowAnchor = nil }
}

private struct LibraryTileFrames: PreferenceKey {
    static let defaultValue: [PrintID: CGRect] = [:]
    static func reduce(value: inout [PrintID: CGRect], nextValue: () -> [PrintID: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}
