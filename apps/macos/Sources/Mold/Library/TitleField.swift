import MoldClient
import SwiftUI

/// What this print is called.
///
/// Commits on Return and on losing focus, and never on every keystroke: a
/// title typed a letter at a time would be one request per letter and one undo
/// entry per letter, and ⌘Z would walk back through the spelling.
struct TitleField: View {
    let entry: LibraryEntry
    let actions: LibraryActions

    @State private var draft = ""
    @FocusState private var editing: Bool

    var body: some View {
        TextField("Title", text: $draft, prompt: Text(entry.print.filename))
            .textFieldStyle(.roundedBorder)
            .font(.headline)
            .focused($editing)
            .onSubmit(commit)
            .onChange(of: editing) { wasEditing, _ in if wasEditing { commit() } }
            // A different print selected is a different field, not a rename of
            // this one -- so the draft follows the selection rather than
            // being pushed onto it.
            .onChange(of: entry.id) { _, _ in draft = entry.print.title ?? "" }
            .onAppear { draft = entry.print.title ?? "" }
            .accessibilityLabel("Title")
    }

    private func commit() {
        actions.setTitle(draft, on: entry)
    }
}
