import SwiftUI

/// Menu bar commands.
///
/// Every shortcut in the app is declared HERE and only printed elsewhere. The
/// Generate button shows ⌘↩ in its label but does not bind it a second time --
/// two bindings for one action queue the work twice.
struct MoldCommands: Commands {
    @Binding var destination: Destination
    @FocusedValue(\.refreshAction) private var refresh
    @FocusedValue(\.promptTuck) private var promptTuck

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
            // Clicking the picture does this too, but a click is not
            // discoverable and is not available from the keyboard.
            Button(promptTuck?.isTucked == true ? "Show Prompt" : "Hide Prompt") {
                promptTuck?.toggle()
            }
            .keyboardShortcut("p", modifiers: [.command, .option])
            .disabled(promptTuck == nil)
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

/// Whether the Generate pane's prompt capsule is tucked away, and how to
/// change that. Equatable on the state alone -- a closure never is, and the
/// menu only needs to redraw when the word on the item changes.
struct PromptTuckAction: Equatable {
    let isTucked: Bool
    let toggle: () -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.isTucked == rhs.isTucked }
}

struct PromptTuckKey: FocusedValueKey {
    typealias Value = PromptTuckAction
}

extension FocusedValues {
    var refreshAction: RefreshActionKey.Value? {
        get { self[RefreshActionKey.self] }
        set { self[RefreshActionKey.self] = newValue }
    }

    var promptTuck: PromptTuckAction? {
        get { self[PromptTuckKey.self] }
        set { self[PromptTuckKey.self] = newValue }
    }
}
