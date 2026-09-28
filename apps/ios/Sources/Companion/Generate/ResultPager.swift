import MoldClient
import SwiftUI

/// What a render made (DESIGN.md §5.1): a batch pages sideways; each result is
/// the Library's own print -- zoom, clip, turnable mesh -- with Save to
/// Photos, Share, Copy, Favourite and Show in Library along the bottom, and
/// Use These Settings in the menu. A result the Library has not listed yet
/// waits for it (the machine's gallery event arrives within a moment).
struct ResultPager: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    @Environment(AppRouter.self) private var router
    let outcome: BatchOutcome
    let host: MoldHost.ID
    @State private var page = 0

    var body: some View {
        let filenames = outcome.results.compactMap(\.filename)
        let entries = filenames.map { name in
            library.pool.first { $0.everyCopy.contains { $0.hostID == host && $0.print.filename == name } }
        }
        VStack(spacing: 8) {
            TabView(selection: $page) {
                ForEach(Array(entries.enumerated()), id: \.offset) { index, entry in
                    Group {
                        if let entry { PrintPage(entry: entry, trashed: false) } else { ProgressView() }
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: entries.count > 1 ? .always : .never))
            .clipShape(.rect(cornerRadius: 12))
            .task(id: filenames) {
                // Until the grid has it, ask again -- briefly.
                for _ in 0..<10 where entries.contains(where: { $0 == nil }) {
                    await library.reload(host)
                    try? await Task.sleep(for: .milliseconds(400))
                }
            }
            if let summary = outcome.failureSummary {
                Text(summary).font(.footnote).foregroundStyle(.secondaryText)
            }
            if entries.indices.contains(page), let entry = entries[page] {
                ResultBar(entry: entry)
            }
        }
        .padding(.horizontal, 16)
        .sensoryFeedback(.success, trigger: outcome)
    }
}

private struct ResultBar: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    @Environment(AppRouter.self) private var router
    let entry: LibraryEntry

    var body: some View {
        HStack(spacing: 4) {
            if entry.print.kind != .mesh {
                bar("Save to Photos", "square.and.arrow.down") { actions.saveToPhotos([entry]) }
            }
            bar("Share", "square.and.arrow.up") { actions.share([entry]) }
            if entry.print.kind == .picture {
                bar("Copy", "doc.on.doc") { actions.copy(entry) }
            }
            bar(entry.print.isFavorite ? "Unfavourite" : "Favourite", entry.print.isFavorite ? "star.fill" : "star") {
                library.apply(.favorite(!entry.print.isFavorite), to: [entry])
            }
            bar("Show in Library", "photo.on.rectangle.angled") { router.selection = .go(.library) }
            Menu {
                PrintMenu(entries: [entry], trashed: false)
            } label: {
                Label("More", systemImage: "ellipsis").labelStyle(.iconOnly).frame(minWidth: 44, minHeight: 44)
            }
        }
        .padding(4)
        .glassEffect(.regular, in: .capsule)
    }

    private func bar(_ title: LocalizedStringKey, _ symbol: String, _ action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: symbol).labelStyle(.iconOnly).frame(minWidth: 44, minHeight: 44)
        }
        .accessibilityShowsLargeContentViewer()
    }
}
