import Foundation
import MoldClient

/// Where the Machines destination opens, and which machine is already open on
/// it.
///
/// `MOLD_NATIVE_DESTINATION=machines` lands on the fleet overview -- ALWAYS,
/// even when a machine was left open last launch, because a UAT run that
/// photographs "the overview" must photograph the overview. `MOLD_NATIVE_MACHINE`
/// names one machine and opens its page instead, so the same run can
/// photograph both without a script driving the mouse.
enum MachineLaunch: Equatable {
    /// The launch said nothing about this destination: whatever was open
    /// stays open.
    case unchanged
    case overview
    case machine(MoldHost.ID)

    /// The destination's raw name and the requested machine's NAME -- the
    /// friendly one `MOLD_NATIVE_HOSTS` seeds and the sidebar prints, matched
    /// without regard to case. A name matching nothing lands on the overview
    /// rather than on some other machine's page.
    static func resolve(destination: String?, machine: String?,
                        in hosts: [MoldHost]) -> MachineLaunch {
        guard destination == Destination.machines.rawValue else { return .unchanged }
        guard let wanted = machine?.trimmingCharacters(in: .whitespacesAndNewlines),
              !wanted.isEmpty
        else { return .overview }
        let match = hosts.first { $0.name.caseInsensitiveCompare(wanted) == .orderedSame }
        return match.map { .machine($0.id) } ?? .overview
    }
}

/// Which machine the Machines destination has OPEN, held in the one preference
/// the sidebar already writes.
///
/// `selectedMachine` is the sidebar's selection and the Models pane's picked
/// machine; making the pushed page a second, parallel piece of state is how
/// Back would leave a machine row highlighted with the overview on screen.
/// So the navigation path IS that preference, mapped both ways.
enum MachineNavigation {
    /// Empty is the overview. One machine deep is its page -- the stack is
    /// never deeper, because a machine's page leads nowhere else.
    static func path(selected stored: String, in hosts: [MoldHost]) -> [MoldHost.ID] {
        guard let id = UUID(uuidString: stored), hosts.contains(where: { $0.id == id })
        else { return [] }
        return [id]
    }

    /// What the preference becomes when the stack changes -- empty for the
    /// overview, which is what deselects the sidebar's machine row.
    static func stored(path: [MoldHost.ID]) -> String {
        path.last?.uuidString ?? ""
    }
}
