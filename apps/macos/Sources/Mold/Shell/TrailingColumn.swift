import MoldStyle
import SwiftUI

/// The inspector column pinned to the trailing edge of a pane.
///
/// SwiftUI's own `.inspector`, because it is the only thing that carries the
/// divider UP THROUGH the toolbar: everything before it belongs to the pane
/// and everything after it to the column, in the toolbar exactly as in the
/// content. Drawn as a plain `HStack` inside the detail -- which is what this
/// was -- the toolbar stayed one undivided row over the whole pane while the
/// column took a slice out of the content under it, so the sort, thumbnail
/// size and inspector controls were crammed into what was left and the search
/// field was drawn across a divider the toolbar knew nothing about (the
/// owner's screenshot, 2026-09-17).
///
/// A modifier rather than an inline call because Generate takes the same
/// column, and two copies of a rule are two rules.
extension View {
    /// - Parameter searchFillsTheColumn: whether this pane's own `.searchable`
    ///   field already occupies the toolbar over the column. macOS pins that
    ///   field to the toolbar's trailing end at a fixed width -- which is what
    ///   `TrailingColumn.width` IS -- so a pane that has one needs nothing
    ///   else there, and a pane that has not stretches the switch to reserve
    ///   the same width instead. Either way what the PANE puts in the toolbar
    ///   stops at the divider rather than being drawn across the column.
    func trailingColumn(
        isShowing: Binding<Bool>, searchFillsTheColumn: Bool = false,
        @ViewBuilder _ column: () -> some View
    ) -> some View {
        inspector(isPresented: isShowing) {
            column().inspectorColumnWidth(TrailingColumn.width)
        }
        .toolbar {
            // Hidden, there is no column and no divider, so the switch is an
            // ordinary trailing button whatever the pane does with search.
            let reservesColumn = isShowing.wrappedValue && !searchFillsTheColumn
            // The reservation is a spacer BESIDE the button, never a frame
            // on it: a frame widens the button's hit region too, and a click
            // anywhere in the empty band over the column toggled the column.
            ToolbarItem {
                HStack(spacing: 0) {
                    if reservesColumn { Spacer(minLength: 0) }
                    Button { isShowing.wrappedValue.toggle() } label: {
                        Label("Inspector", systemImage: "sidebar.trailing")
                    }
                    .help(isShowing.wrappedValue ? "Hide the inspector" : "Show the inspector")
                }
                .frame(width: reservesColumn ? TrailingColumn.toolbarRegion : nil)
            }
        }
    }
}

enum TrailingColumn {
    /// The column is exactly as wide as the search field above it.
    ///
    /// `.searchable` on macOS puts its field in the WINDOW's toolbar, at a
    /// fixed width, hard against the toolbar's trailing inset -- there is no
    /// supported placement that moves it, and none that makes it follow a
    /// column. So the column follows IT: at any other width the field hangs
    /// over the divider by the difference, which is the straddle in the
    /// owner's screenshot; at this one it sits flush against the column's
    /// leading edge and the two toolbar regions read as one clean split.
    ///
    /// Fixed rather than resizable for the same reason, and because the
    /// `HStack` this replaced was a hard `.frame(width: 320)` anyway.
    static let width: CGFloat = toolbarRegion + Chrome.toolbarEdgeInset

    /// The column's share of the toolbar: its width, less the inset every
    /// toolbar keeps at the window's edge. It is AppKit's own search-field
    /// width because that field is the one thing in the toolbar whose size
    /// nothing here can choose.
    static let toolbarRegion: CGFloat = 320
}
