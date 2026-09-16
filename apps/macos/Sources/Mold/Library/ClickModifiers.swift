import AppKit
import MoldClient

/// Which modifier keys were held for the click being handled.
///
/// SwiftUI's tap gestures carry no modifiers on macOS, so command-click and
/// shift-click are otherwise indistinguishable from a plain click. Reading
/// `NSEvent.modifierFlags` inside the tap action is NOT good enough: a cell
/// also has a double-tap gesture, so the single tap is deferred until the
/// double-click interval elapses, and by then the key is usually released --
/// every command-click arrives looking like a plain one.
///
/// So the flags are recorded at mouse DOWN, which is the moment the intent was
/// expressed, and read back when the deferred action finally runs.
@MainActor
enum ClickModifiers {
    private static var lastDown: NSEvent.ModifierFlags = []
    private static var monitor: Any?

    /// Starts watching. Called once from the composition root.
    static func startObserving() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown]) { event in
            lastDown = event.modifierFlags
            return event
        }
    }

    static var current: LibraryCursor.Modifier {
        // Shift wins over command, matching the Finder.
        if lastDown.contains(.shift) { return .extend }
        if lastDown.contains(.command) { return .toggle }
        return .none
    }
}
