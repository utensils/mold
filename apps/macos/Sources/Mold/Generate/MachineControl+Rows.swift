import Foundation
import MoldClient

/// The Machine control's menu contents (M8 design, decision 2): Auto plus
/// every machine that is up and generates, with the CHOSEN machine always
/// listed so the choice stays visible and changeable even while it is down.
/// `MachineControl`'s primary declaration is the view (`MachineControl.swift`,
/// S2) -- a zero-case enum can never be constructed, so it cannot also
/// conform to `View`, and this file only extends the struct declared there.
extension MachineControl {
    struct Row: Equatable, Identifiable {
        let id: MoldHost.ID
        let name: String
        let isUp: Bool

        var caption: String { isUp ? name : "\(name) — can't be reached" }
    }

    struct Rows: Equatable {
        let autoName: String?
        let machines: [Row]
        let chosen: MoldHost.ID?

        var label: String {
            guard let chosen else {
                return autoName.map { "Auto · \($0)" } ?? "Auto"
            }
            return machines.first { $0.id == chosen }?.name ?? "Auto"
        }
    }

    static func rows(
        hosts: [MoldHost], chosen: MoldHost.ID?, preferred: MoldHost?,
        isUp: (MoldHost) -> Bool, generates: (MoldHost) -> Bool
    ) -> Rows {
        var machines: [Row] = []
        for host in hosts {
            let up = isUp(host)
            guard host.id == chosen || (up && generates(host)) else { continue }
            machines.append(Row(id: host.id, name: host.name, isUp: up))
        }
        return Rows(autoName: preferred?.name, machines: machines, chosen: chosen)
    }

    /// The model to run after the machine changes: the same name when it is
    /// ready there, else the first ready model there, else nil.
    static func model(after current: String?, readyThere: [Model]) -> Model? {
        if let current, let kept = readyThere.first(where: { $0.name == current }) {
            return kept
        }
        return readyThere.first
    }
}
