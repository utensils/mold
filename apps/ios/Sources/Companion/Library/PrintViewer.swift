import MoldClient
import SwiftUI

/// A print, full screen (DESIGN.md §5.2): paging through the grid's own order,
/// tap to hide the chrome, Share / Favourite / Info / Delete along the bottom,
/// and everything else in the ⋯ menu -- the grid's menu, word for word.
struct PrintViewer: View {
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
    @State private var showsInfo = false

    init(start: PrintID, entries: [LibraryEntry], trashed: Bool, projection: LibraryGridProjection? = nil) {
        self.start = start
        self.projection = projection ?? LibraryGridProjection(entries: entries)
        self.trashed = trashed
    }

    var body: some View {
        let entry = projection.entry(current ?? start)
        let visibleChrome = UIDevice.current.userInterfaceIdiom == .phone
            || Self.showsChrome(for: entry?.print.kind, requested: chrome)
        return TabView(selection: Binding(get: { current ?? start }, set: { select($0) })) {
            ForEach(projection.pages(around: pageAnchor ?? start)) { page in
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
        .id(pageAnchor ?? start)
        .tabViewStyle(.page(indexDisplayMode: .never))
        .background(.black)
        .ignoresSafeArea(edges: visibleChrome ? [] : .all)
        .toolbar(visibleChrome ? .visible : .hidden, for: .navigationBar, .bottomBar)
        .toolbarVisibility(.hidden, for: .tabBar)
        .navigationTitle(entry.map(title) ?? "")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let entry {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu { PrintMenu(entries: [entry], trashed: trashed) } label: {
                        Label("More", systemImage: "ellipsis")
                    }
                }
                ToolbarItemGroup(placement: .bottomBar) { bottomBar(entry) }
            }
        }
        .sheet(isPresented: $showsInfo) {
            if let entry { PrintInfoSheet(entry: entry, trashed: trashed) }
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
            if let now = current ?? Optional(start), projection.entry(now) == nil { dismiss() }
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

    private func select(_ id: PrintID) {
        if projection.shouldRecenter(selected: id, anchor: pageAnchor ?? start) { pageAnchor = id }
        current = id
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
            Button { library.apply(.favorite(!entry.print.isFavorite), to: [entry]) } label: {
                Label(entry.print.isFavorite ? "Unfavourite" : "Favourite",
                      systemImage: entry.print.isFavorite ? "star.fill" : "star")
            }
            .keyboardShortcut("f", modifiers: [.command, .option])
            Spacer()
        }
        Button { showsInfo = true } label: { Label("Info", systemImage: "info.circle") }
            .keyboardShortcut("i", modifiers: [.command, .option])
        Spacer()
        Button(role: .destructive) {
            Task { trashed ? await library.deleteImmediately([entry]) : await library.trash([entry]) }
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
