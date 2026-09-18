import Foundation

public extension LibraryShowing {
    /// The prints the right-hand column -- and the Library menu -- are about.
    ///
    /// ONE rule, derived rather than written: while the viewer is showing a
    /// print, THAT print is what everything downstream is about; with the
    /// viewer closed it is whatever the grid has selected. Opening a print
    /// deliberately does not write the grid's selection -- the selection is
    /// the cursor a ⇧-click extends and ⌘A replaces, and a viewer that
    /// rewrote it on every ← would throw that away -- so the two are read
    /// together here instead of one being made to follow the other.
    ///
    /// Falls back to the selection for a `viewing` the list no longer holds,
    /// because that is exactly when the pane falls back to the grid
    /// (`LibraryPane.content`): the print is not on screen, so nothing should
    /// still be describing it.
    func inspected(viewing: PrintID?) -> [LibraryEntry] {
        guard let viewing, let open = visible.first(where: { $0.id == viewing })
        else { return selected }
        return [open]
    }
}
