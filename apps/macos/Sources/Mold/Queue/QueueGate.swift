import Foundation
import MoldClient

/// The whole-queue gate, as a menu and a toolbar both need to see it.
///
/// Declared ONCE and shared, so the menu bar item and the pane's own control
/// cannot disagree about the word on them -- which is the same rule the
/// Library's plan follows, and the reason `RowAction` exists.
struct QueueGateOffer: Equatable {
    struct Machine: Equatable, Identifiable {
        let id: MoldHost.ID
        let name: String
        let isPaused: Bool

        /// One machine's word. It says what pressing it DOES, never what the
        /// queue currently is.
        var title: String { isPaused ? "Resume Queue" : "Pause Queue" }
        var titleNamingMachine: String {
            isPaused ? "Resume Queue on \(name)" : "Pause Queue on \(name)"
        }
    }

    /// Only the machines that advertise it. A machine that does not is not
    /// listed -- absent, never present and inert.
    let machines: [Machine]
    let toggle: (MoldHost.ID) -> Void

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.machines == rhs.machines }

    /// The rows to draw. Nothing where no machine offers it; one plain item
    /// where exactly one does; a named item per machine otherwise -- the
    /// idiom Empty Queue… already uses, so a mixed fleet is never ambiguous.
    ///
    /// Deliberately NO keyboard chord. The desktop app binds Space, and this
    /// one cannot: the Library already owns a bare Space for Quick Look, and
    /// the README warns about binding one key twice. Nothing else is both
    /// free and conventional for "pause", so the item carries none rather
    /// than inventing a chord nobody would guess.
    func items() -> [RowAction<MoldHost.ID>] {
        guard machines.count != 1 else {
            let machine = machines[0]
            return [RowAction(kind: machine.id, title: machine.title)]
        }
        return machines.map { RowAction(kind: $0.id, title: $0.titleNamingMachine) }
    }

    /// What the pane says about a paused machine. A toggled label alone is
    /// not visible enough: the word on a control tells you what pressing it
    /// does, not what is true right now.
    static func pausedSentence(machine: String) -> String {
        "\(machine) is not starting anything new — its queue is paused."
    }
}
