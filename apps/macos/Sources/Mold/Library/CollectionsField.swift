import MoldClient
import MoldStyle
import SwiftUI

/// Which shelves the selection is on, and a way onto another.
///
/// Shows only the shelves EVERY selected print is on, for the same reason the
/// tag chips do: a remove button that acts on all of them must not be offered
/// against something only some of them have.
struct CollectionsField: View {
    let entries: [LibraryEntry]
    let actions: LibraryActions

    @Environment(LibraryStore.self) private var library

    var body: some View {
        WrappingHStack(horizontalSpacing: 4, verticalSpacing: 4, alignment: .center) {
            ForEach(shared) { shelf in
                chip(shelf)
            }
            addMenu
        }
    }

    /// The shelves every selected print is on.
    ///
    /// Membership is spelled in each machine's own ids, so this asks the
    /// question per machine and keeps only the shelves that answered yes
    /// everywhere -- which is what makes one chip mean one thing across a fleet.
    private var shared: [CollectionShelf] {
        library.shelves.filter { shelf in
            entries.allSatisfy { entry in
                guard let id = shelf.hosts[entry.hostID] else { return false }
                return entry.print.collectionList.contains(id)
            }
        }
    }

    private var available: [CollectionShelf] {
        let on = Set(shared.map(\.slug))
        return library.shelves.filter { !on.contains($0.slug) }
    }

    private func chip(_ shelf: CollectionShelf) -> some View {
        HStack(spacing: 3) {
            Text(shelf.name).lineLimit(1).truncationMode(.middle).help(shelf.name)
            Button { actions.unfile(entries, from: shelf) } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
            .help("Take these out of \(shelf.name)")
        }
        .font(.caption)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Chrome.wellFill, in: Capsule())
    }

    @ViewBuilder private var addMenu: some View {
        if !available.isEmpty {
            Menu {
                ForEach(available) { shelf in
                    Button(shelf.name) { actions.file(entries, into: shelf) }
                }
            } label: {
                Image(systemName: "plus")
                    .font(.caption)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Add these to a collection")
            .accessibilityLabel("Add to a collection")
        }
    }
}
