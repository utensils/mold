import SwiftUI

extension View {
    /// A plain confirm with a danger button, over a `Destruction` waiting on
    /// an answer.
    ///
    /// Three places asked this question their own way -- the Library pane,
    /// the sidebar's Empty Trash, and a collection's own delete -- and all
    /// three meant the same thing: never a typed phrase. Making somebody
    /// retype a word does not make them read the sentence.
    func destructionDialog(_ pending: Binding<LibraryActions.Destruction?>) -> some View {
        confirmationDialog(
            pending.wrappedValue?.title ?? "",
            isPresented: Binding(get: { pending.wrappedValue != nil },
                                 set: { if !$0 { pending.wrappedValue = nil } }),
            presenting: pending.wrappedValue
        ) { destruction in
            Button(destruction.verb, role: .destructive, action: destruction.perform)
            Button("Cancel", role: .cancel) {}
        } message: { destruction in
            Text(destruction.message)
        }
    }
}
