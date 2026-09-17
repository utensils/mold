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
            // The viewer's Escape and arrows are key equivalents, which beat
            // a caret. This is how they know to leave it alone.
            .focusedValue(\.editingText, editing ? true : nil)
            .onSubmit(commit)
            .onChange(of: editing) { wasEditing, _ in if wasEditing { commit() } }
            // A different print selected is a different field, not a rename of
            // this one -- so the draft follows the selection rather than
            // being pushed onto it.
            .onChange(of: entry.id) { _, _ in draft = entry.print.title ?? "" }
            .onAppear { draft = entry.print.title ?? "" }
            .accessibilityLabel("Title")
            // A different print is a different field, and SwiftUI defines no
            // order between the `onChange(of: editing)` that commits on focus
            // loss and the `onChange(of: entry.id)` that resets the draft --
            // so a selection changing while the field is focused could commit
            // one print's draft onto the next. A fresh instance per print
            // removes the question rather than reasoning about the ordering.
            .id(entry.id)
    }

    private func commit() {
        actions.setTitle(draft, on: entry)
    }
}
