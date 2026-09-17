import AppKit

/// One string, on the pasteboard.
///
/// Every copy of a STRING goes through it -- the ⧉ beside a value, a menu
/// item that has no button to hang one off -- because several surfaces each
/// clearing and setting the general pasteboard by hand is how one of them
/// ends up forgetting to clear. (A picture is `writeObjects`, and stays
/// where it is written.)
enum Clipboard {
    static func put(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}
