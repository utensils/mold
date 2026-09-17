import SwiftUI

/// A fixed column pinned to the trailing edge of the content, inside it.
///
/// Deliberately NOT SwiftUI's `.inspector`: on macOS that splits the WINDOW's
/// toolbar at the column's leading edge, so `.searchable`'s field -- which
/// sits at the trailing end -- is drawn across the divider and the controls
/// before it are crammed into what is left. Drawn here, inside the detail, the
/// toolbar stays one undivided row over the whole pane, the way a Finder
/// window's toolbar sits over its whole width.
///
/// A modifier rather than an inline `HStack` because Generate takes the same
/// column, and two copies of a rule are two rules.
extension View {
    func trailingColumn(
        isShowing: Bool, @ViewBuilder _ column: () -> some View
    ) -> some View {
        HStack(spacing: 0) {
            // The content is the flexible side and must SAY so: an `HStack`
            // hands a child its ideal width unless told otherwise, and the
            // grid's ideal is one tile, which left it adrift in the middle
            // with the column pushed off the trailing edge.
            self.frame(maxWidth: .infinity, maxHeight: .infinity)
            if isShowing {
                Divider()
                column().frame(width: 320)
            }
        }
    }
}
