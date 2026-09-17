import Foundation
import MoldClient

/// The stroke undo funnel.
///
/// `applyAdd`/`applyRemove` are each other's inverse, and each registers the
/// OTHER as what undoing it does -- the same "same funnel" rule
/// `MoldUndo.register` documents, which is what makes redo free rather than
/// just undo.
extension MaskEditorSheet {
    /// Not `private`: the canvas's `paintGesture` calls it on drag end.
    func commitDrag() {
        defer { dragPoints = [] }
        guard !dragPoints.isEmpty else { return }
        applyAdd(MaskStroke(points: dragPoints, radius: brushSize, erases: erasing))
    }

    /// Not `private`: `MaskEditorSheet+Toolbar`'s hidden ⌘Z button calls it.
    func performUndo() {
        MaskUndo.resolve(environment: environmentUndo, own: ownUndo).undo()
    }

    private func applyAdd(_ stroke: MaskStroke) {
        strokes.add(stroke)
        registerStrokeInverse { applyRemove(stroke) }
    }

    private func applyRemove(_ stroke: MaskStroke) {
        strokes.undo()
        registerStrokeInverse { applyAdd(stroke) }
    }

    private func registerStrokeInverse(_ inverse: @escaping () -> Void) {
        let manager = MaskUndo.resolve(environment: environmentUndo, own: ownUndo)
        manager.registerUndo(withTarget: ownUndo) { _ in
            MainActor.assumeIsolated { inverse() }
        }
        if !manager.isUndoing { manager.setActionName("Stroke") }
    }
}
