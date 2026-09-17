import SwiftUI

/// Menu bar commands.
///
/// Every shortcut in the app is declared HERE and only printed elsewhere. The
/// Generate button shows ⌘↩ in its label but does not bind it a second time --
/// two bindings for one action queue the work twice.
///
/// Select All is NOT declared here: the system's own Edit ▸ Select All
/// already exists (read from the live menu: `Select All [off] A`), a second
/// item with the same key would be a duplicate, and SwiftUI does not route
/// the system item to a focusable grid -- so the Library grid answers ⌘A
/// itself (`LibraryGrid`), the one shortcut declared outside this file.
/// Find is added after `.textEditing` because this template carries no
/// Find submenu of its own.
/// Items that ride a focused value are present and disabled when nothing
/// publishes them, the rule `Refresh` set, so Help ▸ Search can find every
/// item from every pane.
struct MoldCommands: Commands {
    @Binding var destination: Destination
    @FocusedValue(\.refreshAction) private var refresh
    @FocusedValue(\.promptTuck) private var promptTuck
    @FocusedValue(\.inspectorToggle) private var inspector
    @FocusedValue(\.libraryFile) private var libraryFile
    @FocusedValue(\.librarySelection) private var librarySelection
    @FocusedValue(\.findAction) private var findAction
    @FocusedValue(\.thumbnailScale) private var thumbnailScale
    @Environment(\.openURL) private var openURL

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Image") { destination = .generate }
                .keyboardShortcut("n")
        }

        CommandGroup(after: .saveItem) {
            Menu("Export…") {
                ForEach(libraryFile?.exportFormats ?? [], id: \.self) { format in
                    Button(format.uppercased()) { libraryFile?.export(format) }
                }
            }
            .disabled(libraryFile?.exportFormats.isEmpty ?? true)
            .keyboardShortcut("e", modifiers: [.command, .shift])
            Button("Save a Copy…") { libraryFile?.save() }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                .disabled((libraryFile?.count ?? 0) == 0)
            // A `ShareLink` in a menu already draws as "Share ▸" with the
            // system's own sharing services one level in -- the same control
            // `LibraryMenu.swift`'s own right-click menu uses, just moved
            // (design decision 24).
            if let share = librarySelection?.share, !share.isEmpty {
                ShareLink(items: share) { SharePreview($0.filename) }
            }
        }

        CommandGroup(after: .textEditing) {
            Button("Find") { findAction?() }
                .keyboardShortcut("f")
                .disabled(findAction == nil)
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
            Button(inspector?.isShowing == true ? "Hide Inspector" : "Show Inspector") {
                inspector?.toggle()
            }
            .keyboardShortcut("i", modifiers: [.command, .option])
            .disabled(inspector == nil)
            Button("Refresh") { refresh?() }
                .keyboardShortcut("r")
                .disabled(refresh == nil)
            Divider()
            Button("Larger Thumbnails") { thumbnailScale?.step(ThumbnailStep.delta) }
                .keyboardShortcut("+", modifiers: .command)
                .disabled(thumbnailScale == nil)
            Button("Smaller Thumbnails") { thumbnailScale?.step(-ThumbnailStep.delta) }
                .keyboardShortcut("-", modifiers: .command)
                .disabled(thumbnailScale == nil)
        }

        // The default item points at a help book this app does not ship.
        CommandGroup(replacing: .help) {
            Button("Mold on the Web") {
                openURL(URL(string: "https://utensils.io/mold/")!)
            }
        }
    }
}
