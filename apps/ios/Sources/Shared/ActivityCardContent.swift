import Foundation

/// Presentation of an existing ActivityKit payload; keep the wire state stable
/// so activities started by an older installed build still draw correctly.
nonisolated struct ActivityCardContent {
    let title: String
    let machine: String
    let detail: String?
    let progress: Double?
    let canStop: Bool

    init(state: GenerationActivityAttributes.ContentState, machine: String, isStale: Bool) {
        self.machine = machine
        title = isStale ? String(localized: "Open Mold Studio to refresh") : state.sentence
        canStop = state.phase == .running
        progress = canStop && !isStale ? state.fraction : nil
        let suffix = " · \(machine)"
        if !canStop || isStale || state.figure == machine {
            detail = nil
        } else if let figure = state.figure, figure.hasSuffix(suffix) {
            detail = String(figure.dropLast(suffix.count))
        } else {
            detail = state.figure
        }
    }
}
