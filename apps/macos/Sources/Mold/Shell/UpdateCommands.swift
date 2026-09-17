import SwiftUI

/// Mold ▸ Check for Updates…
///
/// Declared HERE and nowhere else. The README's rule about two bindings for
/// one action queueing the work twice applies to a menu item as much as to a
/// shortcut: `checkForUpdates()` shows a window, and two of them is two
/// windows racing the same session.
///
/// `CommandGroup(after: .appInfo)` is the macOS convention -- directly under
/// "About Mold", above the Settings item -- and is the placement Sparkle's own
/// SwiftUI setup uses (https://sparkle-project.org/documentation/programmatic-setup).
/// It carries no keyboard shortcut, like every other app's.
///
/// In a build with no updater the item is ABSENT rather than greyed out: a
/// disabled "Check for Updates…" in a dev build would be a promise about a
/// feed that build must never read.
struct UpdateCommands: Commands {
    var body: some Commands {
        CommandGroup(after: .appInfo) {
            if let updates = SoftwareUpdates.shared {
                CheckForUpdatesItem(updates: updates)
            }
        }
    }
}

/// A view rather than a bare `Button` because the enabled state has to
/// re-evaluate: `canCheckForUpdates` is false while a check is already
/// running, and a menu built once would keep offering it.
private struct CheckForUpdatesItem: View {
    let updates: SoftwareUpdates

    var body: some View {
        Button("Check for Updates…") { updates.checkForUpdates() }
            .disabled(!updates.canCheckForUpdates)
    }
}
