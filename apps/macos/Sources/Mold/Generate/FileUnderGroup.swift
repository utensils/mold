import MoldClient
import MoldStyle
import SwiftUI

/// Title, tags and the collection a finished print files into.
///
/// Absent entirely when the host cannot organize -- an older server has no
/// tag or collection tables to write these into, so offering the fields
/// would promise an edit that never lands.
struct FileUnderGroup: View {
    let shelves: [CollectionShelf]
    @Binding var draft: RenderDraft

    @State private var showingNewCollectionSheet = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            LabeledSection("Title") {
                TextField("Title", text: $draft.title)
                    .textFieldStyle(.roundedBorder)
                if titleOverLimit {
                    Text("Titles are at most \(ClientTags.titleMaxChars) characters.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            LabeledSection("Tags") {
                FileUnderTagsRow(tags: $draft.tags, title: draft.title,
                                autoTagTitle: $draft.autoTagTitle)
            }
            LabeledSection("Collection") {
                Picker("Collection", selection: collectionSelection) {
                    Text("None").tag(CollectionOption.none)
                    ForEach(shelves) { shelf in
                        Text(shelf.name).tag(CollectionOption.named(shelf.name))
                    }
                    Divider()
                    Text("New Collection…").tag(CollectionOption.new)
                }
                .labelsHidden()
            }
            Toggle("Tag new prints with their title", isOn: $draft.autoTagTitle)
        }
        .sheet(isPresented: $showingNewCollectionSheet) {
            NewCollectionNameSheet { draft.collectionName = $0 }
        }
    }

    private var titleOverLimit: Bool {
        draft.title.trimmingCharacters(in: .whitespacesAndNewlines).count > ClientTags.titleMaxChars
    }

    private enum CollectionOption: Hashable {
        case none
        case named(String)
        case new
    }

    /// The picker's selection never carries an ACTION: picking "New
    /// Collection…" opens the sheet and leaves `draft.collectionName`
    /// untouched, so the next redraw reflects whatever is really chosen
    /// rather than the menu item itself.
    private var collectionSelection: Binding<CollectionOption> {
        Binding(
            get: {
                guard let name = draft.collectionName, !name.isEmpty else { return .none }
                return .named(name)
            },
            set: { option in
                switch option {
                case .none: draft.collectionName = nil
                case let .named(name): draft.collectionName = name
                case .new: showingNewCollectionSheet = true
                }
            }
        )
    }
}

extension FileUnderGroup {
    /// Whether this group has anything to show -- the same gate the
    /// Library's own organize controls read.
    static func isShown(capabilities: Capabilities?) -> Bool {
        capabilities?.canOrganize ?? false
    }
}

/// Names a collection for a draft to file into -- never creates one.
///
/// Unlike `ShelfNameSheet` (the Library's own, which creates on commit), the
/// collection this names comes into being only when a print carrying it
/// lands, on whichever machine renders it (`CollectionRef.named` resolves or
/// creates by slug). That difference is why this is its own small sheet
/// rather than a reuse of the Library's.
struct NewCollectionNameSheet: View {
    let onCommit: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var name = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("New Collection").font(.headline)
            Text("Named now; it comes into being once a print lands in it.")
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField("Name", text: $name, prompt: Text("Smurf Village"))
                .textFieldStyle(.roundedBorder)
                .onSubmit(commit)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Name Collection", action: commit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmed.isEmpty)
            }
        }
        .padding(20)
        .frame(width: 360)
    }

    private var trimmed: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    private func commit() {
        guard !trimmed.isEmpty else { return }
        onCommit(trimmed)
        dismiss()
    }
}
