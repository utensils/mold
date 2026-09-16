import MoldClient
import MoldStyle
import SwiftUI

/// Tags on the selection.
///
/// Shows the tags EVERY selected print has. A tag on only some of them would
/// be a lie in a control whose remove button acts on all of them.
struct TagEditor: View {
    let entries: [LibraryEntry]
    let actions: LibraryActions
    let filterBy: (String) -> Void

    @State private var adding = ""
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
                .onSubmit {
                    actions.setTag(adding, adding: true, on: entries)
                    adding = ""
                }
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
    }
}
