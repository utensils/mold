import MoldClient
import SwiftUI

/// `LibraryMenuPlan`'s items, as menu rows.
///
/// The one renderer, so the tile's right-click menu and the menu bar's Library
/// menu cannot drift again: they differ in WHERE they appear and in what they
/// are given, never in what is offered or what it is called. Share is the one
/// thing outside the plan -- it is a `ShareLink`, a system control rather than
/// something this app performs -- and each surface adds it in its own place.
struct LibraryMenuItems: View {
    let items: [LibraryMenuItem]
    let perform: (LibraryAction) -> Void

    var body: some View {
        ForEach(items) { item in
            if item.isDivider {
                Divider()
            } else if item.isSubmenu {
                Menu(item.title) {
                    LibraryMenuItems(items: item.children, perform: perform)
                }
            } else if let action = item.action {
                Button(item.title, role: item.isDestructive ? .destructive : nil) {
                    perform(action)
                }
            }
        }
    }
}
