import AppKit

/// One string, on the pasteboard.
///
/// `CopyButton` already does this for the ⧉ beside a value; a menu item has no
/// button to hang it off, and three menus each clearing and setting the
/// general pasteboard by hand is how one of them ends up forgetting to clear.
enum Clipboard {
    static func put(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}
