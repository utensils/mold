import SwiftUI

/// The Queue menu.
///
/// `LibraryCommands.swift`'s own reason: the menu bar is what Help ▸ Search
/// searches and VoiceOver reads, and an action that exists only in a row's
/// contextual menu is unreachable from the keyboard. Every job item here is
/// the same call the selected row's own controls make; Empty Queue… is the
/// same call the toolbar's own button makes (`QueuePane+Commands.swift`).
struct QueueCommands: Commands {
    @Binding var destination: Destination
    @FocusedValue(\.queueSelection) private var selection
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandMenu("Queue") {
            // The Queue menu must remain useful from every destination and
            // after the main window has been closed. Row actions are scoped
            // to the Queue pane, but the doorway to that pane is global.
            Button("Show Queue") {
                destination = .queue
                openWindow(id: "main")
            }
            if let selection {
                Divider()
                RowActionMenu(actions: selection.offered, perform: selection.perform) { item in
                    // ⌘⌫, the Library's own Move to Trash chord: the selected
                    // row leaves the queue from the keyboard, and Help ▸
                    // Search finds it. The menu bar is the only surface that
                    // carries chords -- a contextual menu shows none.
                    item == .act(.cancel) ? KeyboardShortcut(.delete, modifiers: .command) : nil
                }
            }
        }
    }
}
