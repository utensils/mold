import MoldClient
import MoldStyle
import SwiftUI

/// Tags on the selection.
///
/// Shows the tags EVERY selected print has. A tag on only some of them would
/// be a lie in a control whose remove button acts on all of them.
/// A tag a sheet is about.
///
/// A wrapper rather than a retroactive `Identifiable` on `String`: conforming
/// a stdlib type in an app target is a conflict waiting for whichever library
/// does it next, and it would make every string in the app look presentable in
/// a `sheet(item:)`.
struct TagName: Identifiable, Hashable {
    let name: String
    var id: String { name }
    init(_ name: String) { self.name = name }
}

struct TagEditor: View {
    let entries: [LibraryEntry]
    let actions: LibraryActions
    let filterBy: (String) -> Void

    @State private var adding = ""
    @State private var renaming: TagName?
    @FocusState private var typing: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !shared.isEmpty {
                WrappingHStack(horizontalSpacing: 4, verticalSpacing: 4, alignment: .center) {
                    ForEach(shared, id: \.self) { tag in
                        chip(tag)
                    }
                }
            }
            TextField("Add a tag", text: $adding)
                .textFieldStyle(.roundedBorder)
                .controlSize(.small)
                .focused($typing)
                .focusedValue(\.editingText, typing ? true : nil)
                .onSubmit {
                    actions.setTag(adding, adding: true, on: entries)
                    adding = ""
                }
        }
        .sheet(item: $renaming) { subject in
            TagNameSheet(tag: subject.name) { actions.renameTag(subject.name, to: $0) }
        }
    }

    private var shared: [String] {
        guard let first = entries.first else { return [] }
        var common = Set(first.print.tagList)
        for entry in entries.dropFirst() {
            common.formIntersection(entry.print.tagList)
        }
        return common.sorted()
    }

    private func chip(_ tag: String) -> some View {
        HStack(spacing: 3) {
            Button { filterBy(tag) } label: { Text(tag) }
                .buttonStyle(.plain)
                .help("Show everything tagged \(tag)")
            Button { actions.setTag(tag, adding: false, on: entries) } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
            .help("Remove this tag")
        }
        .font(.caption)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Chrome.wellFill, in: Capsule())
        // Renaming and deleting reach EVERY print on EVERY machine, which is a
        // different act from taking the tag off this one -- so it lives on the
        // contextual menu and says so, rather than sitting next to the x.
        .contextMenu {
            Button("Show Everything Tagged \u{201C}\(tag)\u{201D}") { filterBy(tag) }
            Divider()
            Button("Rename Tag Everywhere…") { renaming = TagName(tag) }
            Button("Delete Tag Everywhere…", role: .destructive) { actions.deleteTag(tag) }
        }
    }
}
