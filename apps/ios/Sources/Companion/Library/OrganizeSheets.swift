import MoldClient
import SwiftUI

/// Tags for one print or several: every tag any of them carries, each a
/// toggle, plus a field for a new one. A change applies to every copy.
struct TagsSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    let entries: [LibraryEntry]
    @State private var newTag = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("New tag", text: $newTag)
                        .textInputAutocapitalization(.never)
                        .onSubmit(add)
                    Button("Add Tag", action: add).disabled(clean(newTag).isEmpty)
                }
                Section {
                    ForEach(known, id: \.self) { tag in
                        let on = entries.allSatisfy { entry in
                            entry.tags.contains { $0.caseInsensitiveCompare(tag) == .orderedSame }
                        }
                        Toggle(tag, isOn: Binding(get: { on }, set: { adding in
                            library.apply(.tag(tag, adding: adding), to: entries)
                        }))
                    }
                } header: {
                    SectionHeader(String(localized: "Tags"))
                }
            }
            .navigationTitle(entries.count == 1 ? String(localized: "Tags") : String(localized: "Tags for \(entries.count) Prints"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .presentationDetents([.medium, .large])
    }

    /// Every tag in the Library, so a tag used elsewhere is one tap away.
    private var known: [String] {
        library.knownTags
    }

    /// A leading `#` is typing, not part of the tag (the Mac's rule).
    private func clean(_ text: String) -> String {
        var tag = text.trimmingCharacters(in: .whitespacesAndNewlines)
        while tag.hasPrefix("#") { tag.removeFirst() }
        return tag
    }

    private func add() {
        let tag = clean(newTag)
        guard !tag.isEmpty else { return }
        library.apply(.tag(tag, adding: true), to: entries)
        newTag = ""
    }
}

/// A new collection, filed with these prints at once (the machine creates it).
struct NewCollectionSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    let entries: [LibraryEntry]
    @State private var name = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Name", text: $name).onSubmit(create)
                } footer: {
                    if !trimmed.isEmpty, slug == nil {
                        Text("A collection's name needs at least one letter or number.")
                            .foregroundStyle(.secondaryText)
                    }
                }
            }
            .navigationTitle("New Collection")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create", action: create).disabled(slug == nil)
                }
            }
        }
        .presentationDetents([.medium])
    }

    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }
    /// The slug every machine will give this name; nil means no machine would
    /// accept it (no letter or number to merge on).
    private var slug: String? { CollectionShelf.slug(for: trimmed) }

    private func create() {
        guard let slug else { return }
        library.apply(.collection(name: trimmed, slug: slug, filing: true), to: entries)
        Task { await library.reload() }
        dismiss()
    }
}

/// A print's title. Blank clears it.
struct RenameSheet: View {
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    let entry: LibraryEntry
    @State private var title = ""

    var body: some View {
        NavigationStack {
            Form {
                TextField("Title", text: $title, prompt: Text(entry.print.displayName)).onSubmit(save)
            }
            .navigationTitle("Rename")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Save", action: save) }
            }
            .onAppear { title = entry.print.title ?? "" }
        }
        .presentationDetents([.medium])
    }

    private func save() {
        let new = title.trimmingCharacters(in: .whitespacesAndNewlines)
        library.apply(.title(from: entry.print.title ?? "", to: new), to: [entry])
        dismiss()
    }
}
