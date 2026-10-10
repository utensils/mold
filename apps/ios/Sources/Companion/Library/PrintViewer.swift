import MoldClient
import SwiftUI

/// A print, full screen (DESIGN.md §5.2): paging through the grid's own order,
/// tap to hide the chrome, Share / Favourite / Info / Delete along the bottom,
/// and everything else in the ⋯ menu -- the grid's menu, word for word.
struct PrintViewer: View {
    @Environment(AppRouter.self) private var router
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    @Environment(\.dismiss) private var dismiss
    @Environment(HostStore.self) private var hosts
    let start: PrintID
    let projection: LibraryGridProjection
    let trashed: Bool
    @State private var current: PrintID?
    @State private var pageAnchor: PrintID?
    @State private var chrome = true
    @State private var viewportSize = CGSize.zero
    @State private var showsInfo = false
    @State private var deleting: LibraryEntry?

    init(start: PrintID, entries: [LibraryEntry], trashed: Bool, projection: LibraryGridProjection? = nil) {
        self.start = start
        self.projection = projection ?? LibraryGridProjection(entries: entries)
        self.trashed = trashed
    }

    var body: some View {
        let entry = projection.entry(current ?? start)
        let anchor = projection.anchor(for: current ?? start, preferred: pageAnchor ?? start)
        let landscapePlayback = Self.usesLandscapePlayback(
            kind: entry?.print.kind, isPhone: UIDevice.current.userInterfaceIdiom == .phone, size: viewportSize)
        let visibleChrome = !landscapePlayback && (UIDevice.current.userInterfaceIdiom == .phone
            || Self.showsChrome(for: entry?.print.kind, requested: chrome))
        return TabView(selection: Binding(get: { current ?? start }, set: { select($0) })) {
            ForEach(projection.pages(around: anchor)) { page in
                Group {
                    if page.print.kind == .clip || page.print.kind == .mesh {
                        // AVKit and the mesh viewer own their gestures.
                        PrintPage(entry: page, trashed: trashed,
                                  isSelected: (current ?? start) == page.id)
                    } else {
                        PrintPage(entry: page, trashed: trashed,
                                  isSelected: (current ?? start) == page.id)
                            .onTapGesture { withAnimation { chrome.toggle() } }
                    }
                }
                .accessibilityIdentifier("viewer-print-\(page.print.filename)")
                .tag(page.id)
            }
        }
        // UIPageViewController retains numeric page indices. Keep its window
        // stable during ordinary swipes and recreate it only at a boundary;
        // changing the leading item every swipe otherwise skips prints.
        .id(anchor)
        .tabViewStyle(.page(indexDisplayMode: .never))
        .background(.black)
        .onGeometryChange(for: CGSize.self) { $0.size } action: { viewportSize = $0 }
        .ignoresSafeArea(edges: visibleChrome ? [] : .all)
        .toolbar(visibleChrome ? .visible : .hidden, for: .navigationBar, .bottomBar)
        .toolbarVisibility(.hidden, for: .tabBar)
        .statusBarHidden(landscapePlayback)
        .overlay(alignment: .top) {
            if landscapePlayback {
                Button { dismiss() } label: {
                    Label("Close", systemImage: "xmark")
                        .labelStyle(.iconOnly)
                        .padding(12)
                        .frame(minWidth: 44, minHeight: 44)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white)
                .background(.black.opacity(0.7), in: .circle)
                .accessibilityIdentifier("landscape-playback-close")
                .padding()
            }
        }
        .navigationTitle(entry.map(title) ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let entry {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu { PrintMenu(entries: [entry], trashed: trashed, requestPermanentDelete: { deleting = $0.first }) } label: {
                        Label("More", systemImage: "ellipsis")
                    }
                }
                ToolbarItemGroup(placement: .bottomBar) { bottomBar(entry) }
            }
        }
        .sheet(isPresented: $showsInfo) {
            if let entry { PrintInfoSheet(entry: entry, trashed: trashed) }
        }
        .alert("Delete Immediately?", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } })) {
            if let deleting {
                Button("Delete Immediately", role: .destructive) { Task { await library.deleteImmediately([deleting]); self.deleting = nil } }
            }
            Button("Cancel", role: .cancel) { deleting = nil }
        } message: {
            Text("Copies on \(deletingMachines) will be removed for good. This can't be undone.")
        }
        .overlay(alignment: .top) {
            if let status = actions.status {
                Text(status)
                    .padding(.horizontal, 16).padding(.vertical, 8)
                    .background(.regularMaterial, in: .capsule)
                    .padding(.top, 8)
                    .onTapGesture { actions.status = nil }
            }
        }
        .onChange(of: ObjectIdentifier(projection)) { _, _ in
            // The print on screen went away (deleted, moved): back to the grid.
            let now = current ?? start
            guard projection.entry(now) != nil else { dismiss(); return }
            pageAnchor = projection.anchor(for: now, preferred: pageAnchor ?? start)
        }
        .keyboardShortcut(for: projection, current: Binding(get: { current }, set: { if let id = $0 { select(id) } }), start: start, close: { dismiss() })
        // Handoff: the same print, continued in Mold Studio on the Mac.
        .userActivity(PrintHandoff.activityType, element: entry) { entry, activity in
            guard let host = hosts.host(entry.hostID) else { return }
            activity.title = entry.spokenName
            activity.isEligibleForHandoff = true
            activity.addUserInfoEntries(from: PrintHandoff.userInfo(
                filename: entry.print.filename, address: host.baseURL, instanceId: hosts.instanceID(of: host.id)))
        }
    }

    private var deletingMachines: String {
        Set((deleting?.everyCopy ?? []).compactMap { hosts.host($0.hostID)?.name }).sorted().joined(separator: ", ")
    }

    private func select(_ id: PrintID) {
        if projection.shouldRecenter(selected: id, anchor: pageAnchor ?? start) { pageAnchor = id }
        current = id
    }

    /// Use available window geometry so both phone landscape orientations work
    /// without device notifications, orientation forcing or rebuilding the player.
    static func usesLandscapePlayback(kind: PrintKind?, isPhone: Bool, size: CGSize) -> Bool {
        isPhone && kind == .clip && size.width > size.height && size.height > 0
    }

    /// Still-image chrome can be hidden; interactive media must retain the
    /// gallery actions because their own gestures cannot restore our bars.
    static func showsChrome(for kind: PrintKind?, requested: Bool) -> Bool {
        requested || kind == .clip || kind == .mesh
    }

    @ViewBuilder private func bottomBar(_ entry: LibraryEntry) -> some View {
        Button { actions.share([entry]) } label: { Label("Share", systemImage: "square.and.arrow.up") }
        Spacer()
        if !trashed {
            Button { library.apply(.favorite(!entry.isFavorite), to: [entry]) } label: {
                Label(entry.isFavorite ? "Unfavourite" : "Favourite",
                      systemImage: entry.isFavorite ? "star.fill" : "star")
            }
            .keyboardShortcut("f", modifiers: [.command, .option])
            Spacer()
        }
        Button { showsInfo = true } label: { Label("Info", systemImage: "info.circle") }
            .keyboardShortcut("i", modifiers: [.command, .option])
        Spacer()
        Button(role: .destructive) {
            if trashed { deleting = entry } else { Task { await library.trash([entry]) } }
        } label: {
            Label("Delete", systemImage: "trash")
        }
        .keyboardShortcut(.delete, modifiers: .command)
    }

    private func title(_ entry: LibraryEntry) -> String {
        LibraryGrouping.title(for: entry.createdAt)
    }
}

private extension View {
    /// ← → walk the prints on iPad or with a keyboard, and Esc closes, as on
    /// the Mac.
    func keyboardShortcut(for projection: LibraryGridProjection, current: Binding<PrintID?>, start: PrintID,
                          close: @escaping () -> Void) -> some View {
        background {
            Group {
                Button("Close", action: close)
                    .keyboardShortcut(.cancelAction)
                Button("Previous Print") { step(-1, projection, current, start) }
                    .keyboardShortcut(.leftArrow, modifiers: [])
                Button("Next Print") { step(1, projection, current, start) }
                    .keyboardShortcut(.rightArrow, modifiers: [])
            }
            .hidden()
        }
    }
}

@MainActor private func step(_ by: Int, _ projection: LibraryGridProjection,
                            _ current: Binding<PrintID?>, _ start: PrintID) {
    guard let next = projection.step(by, from: current.wrappedValue ?? start) else { return }
    current.wrappedValue = next
}
