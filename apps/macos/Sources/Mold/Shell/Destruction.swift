import Foundation

/// Something permanent, waiting on an answer.
///
/// Moved out of `LibraryActions` (M5 S5) so a models pane can raise the same
/// confirm without reaching into a Library type for it -- the third
/// mechanism S7 of M1.5 already removed once. `LibraryActions` keeps a
/// `typealias` so nothing that already writes `LibraryActions.Destruction`
/// has to change.
struct Destruction: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    let verb: String
    let perform: () -> Void
}
