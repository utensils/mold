import MoldClient
import SwiftUI

/// The same collection action serves the selection bar and print menus.
struct SelectionCollectionMenu: View {
    @Environment(LibraryStore.self) private var library
    @Environment(PrintActions.self) private var actions
    let entries: [LibraryEntry]

    var body: some View {
        Menu {
            ForEach(library.shelves) { shelf in
                let filed = entries.flatMap(\.everyCopy).allSatisfy { shelf.count(in: [$0]) > 0 }
                Button {
                    guard library.canOrganize(entries) else { return }
                    library.apply(.collection(name: shelf.name, slug: shelf.slug, filing: true), to: entries)
                } label: {
                    Label(shelf.name, systemImage: filed ? "checkmark" : "rectangle.stack")
                }
                .disabled(filed)
            }
            if !library.shelves.isEmpty { Divider() }
            Button {
                guard library.canOrganize(entries) else { return }
                actions.sheet = .newCollection(entries)
            } label: {
                Label("New Collection…", systemImage: "plus")
            }
            let removable = library.shelves.filter { shelf in
                entries.flatMap(\.everyCopy).contains { shelf.count(in: [$0]) > 0 }
            }
            if !removable.isEmpty {
                Divider()
                Menu("Remove from Collection") {
                    ForEach(removable) { shelf in
                        Button(shelf.name) {
                            guard library.canOrganize(entries) else { return }
                            library.apply(.collection(name: shelf.name, slug: shelf.slug, filing: false), to: entries)
                        }
                    }
                }
            }
        } label: {
            Label("Add to Collection", systemImage: "rectangle.stack.badge.plus")
        }
        .disabled(!library.canOrganize(entries))
    }
}
