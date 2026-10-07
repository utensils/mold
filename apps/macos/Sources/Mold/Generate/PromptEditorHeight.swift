import Foundation

/// The prompt grows upward inside a bottom-aligned panel. Window resizing
/// clamps the rendered height without discarding the person's saved preference.
enum PromptEditorHeight {
    static let initial: CGFloat = 72
    static let minimum: CGFloat = 48
    static let maximum: CGFloat = 560

    static func resolve(_ preferred: CGFloat, available: CGFloat) -> CGFloat {
        let ceiling = max(0, min(maximum, available))
        let floor = min(minimum, ceiling)
        return min(ceiling, max(floor, preferred.isFinite ? preferred : initial))
    }

    static func dragged(from height: CGFloat, translation: CGFloat, available: CGFloat) -> CGFloat {
        resolve(height - translation, available: available)
    }
}
