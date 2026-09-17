import MoldClient
import SwiftUI

// The viewer's own toolbar: back to the grid, the two steps, the print's
// name, and the two things you do to it. Split from the viewer for size.
extension LibraryViewer {
    var bar: some View {
        HStack(spacing: 12) {
            // A grid, not a third chevron: the icon says WHERE back goes, and
            // the bar no longer reads as three arrows in a row.
            Button { onClose() } label: { Label("Library", systemImage: "square.grid.2x2") }
                // Every key this viewer answers is bound to the control that
                // performs it, never to a focus the viewer holds: SwiftUI
                // hands the grid's focus to the search field the moment the
                // viewer replaces it, so `.onKeyPress` here reached nothing.
                // A key equivalent is window-scoped and needs no focus, and
                // these controls exist only while a print is showing.
                .keyboardShortcut(isEditing ? nil : KeyboardShortcut.cancelAction)
                .help("Back to the library (esc)")
            Divider().frame(height: 14)
            Button { onStep(-1) } label: { Label("Previous", systemImage: "chevron.left") }
                .keyboardShortcut(stepping(.leftArrow))
                .help("Previous print (←)")
            Button { onStep(1) } label: { Label("Next", systemImage: "chevron.right") }
                .keyboardShortcut(stepping(.rightArrow))
                .help("Next print (→)")
            Spacer()
            Text(entry.print.metadata.prompt ?? entry.print.filename)
                .lineLimit(1)
                .foregroundStyle(.secondary)
            Spacer()
            Button { actions.toggleFavorite([entry]) } label: {
                Label("Favourite",
                      systemImage: entry.print.isFavorite ? "star.fill" : "star")
            }
            .help(entry.print.isFavorite ? "Remove from Favourites" : "Add to Favourites")
            Button { actions.save([entry]) } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save a copy")
        }
        .buttonStyle(.accessoryBar)
        .labelStyle(.iconOnly)
        .padding(10)
        .background(.bar)
    }
}
