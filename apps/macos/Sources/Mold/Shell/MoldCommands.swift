import SwiftUI

/// Menu bar commands.
///
/// Every shortcut in the app is declared HERE and only printed elsewhere. The
/// Generate button shows ⌘↩ in its label but does not bind it a second time --
/// two bindings for one action queue the work twice.
struct MoldCommands: Commands {
    @Binding var destination: Destination
    @FocusedValue(\.refreshAction) private var refresh

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Image") { destination = .generate }
                .keyboardShortcut("n")
        }

        CommandGroup(after: .toolbar) {
            ForEach(Array(Destination.allCases.enumerated()), id: \.element) { index, item in
                Button(item.title) { destination = item }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
            Divider()
            Button("Refresh") { refresh?() }
                .keyboardShortcut("r")
                .disabled(refresh == nil)
        }
    }
}

/// Lets whichever pane is showing say how it refreshes, so ⌘R means the right
/// thing in each without the menu knowing about any of them.
struct RefreshActionKey: FocusedValueKey {
    typealias Value = () -> Void
}

extension FocusedValues {
    var refreshAction: RefreshActionKey.Value? {
        get { self[RefreshActionKey.self] }
        set { self[RefreshActionKey.self] = newValue }
    }
}
