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

/// The Lock Screen's bounded layout. The unit contract measures the three
/// normal rows with UIKit's actual text metrics, including their padding.
nonisolated enum ActivityCardLayout {
    static let maximumHeight: CGFloat = 160
    static let inset: CGFloat = 16
    static var contentHeight: CGFloat { maximumHeight - 2 * inset }
    static let previewSide: CGFloat = 48
    static let stopSide: CGFloat = 44
    static let rowSpacing: CGFloat = 10
    static let titleSpacing: CGFloat = 3
    static let progressSpacing: CGFloat = 5
    static let progressHeight: CGFloat = 4
}
