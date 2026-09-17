import AppKit

/// Whether anything in this app has a caret in it.
///
/// One question, asked once, because the per-field answer could only ever be
/// as complete as the list of fields that remembered to publish it -- and two
/// of the four did not. `ShelfNameSheet`'s Name and `TagNameSheet`'s field
/// published no `editingText`, so a bare space typed into either fired the
/// Library menu's Quick Look and never reached the caret: "Smurf Village"
/// became "SmurfVillage".
///
/// AppKit's field editor is one `NSText` shared by every `NSTextField` in a
/// window, and it posts these two notifications whichever field it is serving
/// -- in a sheet, in the inspector, in the toolbar's search field. So a view
/// that adds a text field tomorrow is covered without knowing this type
/// exists, which is the property a focused value could not have.
@MainActor
@Observable
final class TextEditingFocus {
    static let shared = TextEditingFocus()

    private(set) var isEditing = false
    @ObservationIgnored private var observers: [any NSObjectProtocol] = []

    /// Starts watching. Called once from the composition root, beside
    /// `ClickModifiers.startObserving()`.
    func startObserving(center: NotificationCenter = .default) {
        guard observers.isEmpty else { return }
        observers = [
            center.addObserver(forName: NSText.didBeginEditingNotification,
                               object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.isEditing = true }
            },
            center.addObserver(forName: NSText.didEndEditingNotification,
                               object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.isEditing = false }
            },
            // A window losing key takes its field editor with it, and no
            // "ended" arrives for a sheet dismissed mid-edit.
            center.addObserver(forName: NSWindow.didResignKeyNotification,
                               object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.isEditing = false }
            },
        ]
    }
}
