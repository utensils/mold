import MoldClient
import SwiftUI

/// A print, full screen (DESIGN.md §5.2): paging through the grid's own order,
/// tap to hide the chrome, Share / Favourite / Info / Delete along the bottom,
/// and everything else in the ⋯ menu -- the grid's menu, word for word.
struct PrintViewer: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    @Environment(\.dismiss) private var dismiss
    let start: PrintID
    let entries: [LibraryEntry]
    let trashed: Bool
    @State private var current: PrintID?
    @State private var chrome = true
    @State private var showsInfo = false

    var body: some View {
        let entry = entries.first { $0.id == (current ?? start) }
        TabView(selection: Binding(get: { current ?? start }, set: { current = $0 })) {
            ForEach(entries) { page in
                PrintPage(entry: page, trashed: trashed)
                    .tag(page.id)
                    .onTapGesture { withAnimation { chrome.toggle() } }
            }
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        .background(.black)
        .ignoresSafeArea(edges: chrome ? [] : .all)
        .toolbar(chrome ? .visible : .hidden, for: .navigationBar, .bottomBar)
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
        .onChange(of: entries.map(\.id)) { _, ids in
            // The print on screen went away (deleted, moved): back to the grid.
            if let now = current ?? Optional(start), !ids.contains(now) { dismiss() }
        }
        .keyboardShortcut(for: entries, current: $current, start: start)
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
    /// ← → walk the prints on iPad or with a keyboard, as on the Mac.
    func keyboardShortcut(for entries: [LibraryEntry], current: Binding<PrintID?>, start: PrintID) -> some View {
        background {
            Group {
                Button("Previous Print") { step(-1, entries, current, start) }
                    .keyboardShortcut(.leftArrow, modifiers: [])
                Button("Next Print") { step(1, entries, current, start) }
                    .keyboardShortcut(.rightArrow, modifiers: [])
            }
            .hidden()
        }
    }
}

@MainActor private func step(_ by: Int, _ entries: [LibraryEntry], _ current: Binding<PrintID?>, _ start: PrintID) {
    guard let index = entries.firstIndex(where: { $0.id == (current.wrappedValue ?? start) }) else { return }
    let next = index + by
    guard entries.indices.contains(next) else { return }
    current.wrappedValue = entries[next].id
}
