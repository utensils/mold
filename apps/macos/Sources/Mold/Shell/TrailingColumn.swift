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
    /// A search field already reserves the trailing toolbar region; other
    /// panes reserve that space beside the inspector switch themselves.
    func trailingColumn(
        isShowing: Binding<Bool>, searchFillsTheColumn: Bool = false, resizable: Bool = false,
        @ViewBuilder _ column: () -> some View
    ) -> some View {
        modifier(TrailingColumnModifier(isShowing: isShowing, searchFillsTheColumn: searchFillsTheColumn,
                                        resizable: resizable, column: column()))
    }
}

private struct TrailingColumnModifier<Column: View>: ViewModifier {
    @Binding var isShowing: Bool
    let searchFillsTheColumn: Bool
    let resizable: Bool
    let column: Column
    @State private var measuredWidth = TrailingColumn.width

    func body(content: Content) -> some View {
        content.inspector(isPresented: $isShowing) {
            column
                // Keep the inspector width trait outermost. Wrapping it in the
                // geometry observer can leave the native divider fixed.
                .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { measuredWidth = $0 }
                .inspectorColumnWidth(min: resizable ? 260 : TrailingColumn.width,
                                      ideal: resizable ? 320 : TrailingColumn.width,
                                      max: resizable ? 480 : TrailingColumn.width)
        }
        .toolbar {
            // Keep the pane's controls and the inspector switch in separate
            // glass groups. Otherwise macOS paints one capsule across the
            // column divider even when their hit regions fit individually.
            ToolbarSpacer(.fixed)
            // Hidden, there is no column and no divider, so the switch is an
            // ordinary trailing button whatever the pane does with search.
            let reservesColumn = isShowing && !searchFillsTheColumn
            // The reservation is a spacer BESIDE the button, never a frame
            // on it: a frame widens the button's hit region too, and a click
            // anywhere in the empty band over the column toggled the column.
            ToolbarItem {
                HStack(spacing: 0) {
                    if reservesColumn { Spacer(minLength: 0) }
                    Button { isShowing.toggle() } label: {
                        Label("Inspector", systemImage: "sidebar.trailing")
                    }
                    .help(isShowing ? "Hide the inspector" : "Show the inspector")
                }
                .frame(width: reservesColumn ? max(0, measuredWidth - Chrome.toolbarEdgeInset) : nil)
            }
            .sharedBackgroundVisibility(.hidden)
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
    /// Search-backed columns keep this fixed width. Generate opts into a
    /// resizable column and measures it to keep its toolbar reservation aligned.
    static let width: CGFloat = toolbarRegion + Chrome.toolbarEdgeInset

    /// The column's share of the toolbar: its width, less the inset every
    /// toolbar keeps at the window's edge. It is AppKit's own search-field
    /// width because that field is the one thing in the toolbar whose size
    /// nothing here can choose.
    static let toolbarRegion: CGFloat = 320
}
