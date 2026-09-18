import Foundation
import MoldClient

/// What an **Also Running** row offers, declared once.
///
/// The same list feeds the row's inline controls and its contextual menu --
/// `RowAction` is the app's one menu model, and every new row needs its own.
struct AlsoRunningActions {
    enum Kind: Hashable {
        case pause, resume, cancel, forget
    }

    let canPause: Bool
    let canResume: Bool
    let canCancel: Bool
    /// Settled work this app is still holding, so its answer can be read and
    /// then dismissed. Nothing is destroyed: the print is already made or
    /// already not.
    let canForget: Bool
    /// Live work nothing here can stop. The menu still carries Cancel,
    /// INERT -- "present but disabled" is how this app says a row cannot do
    /// this right now, and a right-click that opened nothing said nothing.
    let stopNote: String?

    init(_ row: AlsoRunningRow) {
        canPause = row.canPause
        canResume = row.canResume
        canCancel = row.canCancel
        canForget = row.isSettled
        stopNote = row.stopNote
    }

    /// Destructive last, behind a separator -- `RowAction.rendered`'s rule,
    /// which this declares rather than restates.
    func offered() -> [RowAction<Kind>] {
        var items: [RowAction<Kind>] = []
        if canPause { items.append(Self.item(.pause)) }
        if canResume { items.append(Self.item(.resume)) }
        if canForget { items.append(Self.item(.forget)) }
        if canCancel { items.append(Self.item(.cancel)) }
        if stopNote != nil { items.append(Self.item(.cancel, disabled: true)) }
        return RowAction.rendered(items)
    }

    static func item(_ kind: Kind, disabled: Bool = false) -> RowAction<Kind> {
        switch kind {
        case .pause: RowAction(kind: .pause, title: "Pause")
        case .resume: RowAction(kind: .resume, title: "Resume")
        case .forget: RowAction(kind: .forget, title: "Dismiss")
        case .cancel: RowAction(kind: .cancel, title: "Cancel", isDestructive: true, isDisabled: disabled)
        }
    }
}
