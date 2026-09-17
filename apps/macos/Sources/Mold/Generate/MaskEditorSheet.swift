import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// A sheet over Generate: paint a repaint mask against the source image.
///
/// Opened from the Refine group's Mask row (`RefineGroup.swift`) through
/// `GenerateController.showsMaskEditor`. The picture is drawn
/// underneath at reduced opacity so what is painted reads against what it
/// covers, and every committed stroke is stored in the SOURCE image's own
/// pixel space (`MaskStroke`) -- the canvas here is just a scaled window onto
/// that space, never a second coordinate system `MaskRender` has to know
/// about. Split by concern: `+Toolbar` draws the controls, `+Undo` is the
/// stroke undo funnel, `+Source` decodes the picture and the display scale.
struct MaskEditorSheet: View {
    @Binding var draft: RenderDraft
    @Environment(\.dismiss) private var dismiss
    /// Not `private`: `+Undo` reads it to resolve which manager ⌘Z goes to.
    @Environment(\.undoManager) var environmentUndo

    /// Not `private`: `+Source`'s `loadSource()` decodes into it.
    @State var source: NSImage?
    /// Not `private`, same reason -- `+Source`'s `canvasSize` and this
    /// file's `finish()` both read it.
    @State var sourcePixelSize = CGSize(width: 1024, height: 1024)
    /// Not `private`: the toolbar draws from it (Clear/Done's enabled state),
    /// the canvas paints it, and `+Undo` mutates it.
    @State var strokes = MaskStrokes()
    /// Not `private`, same reason: the toolbar's brush/erase controls bind
    /// to these two directly, and `+Undo`'s `commitDrag` reads them.
    @State var brushSize = MaskBrush.defaultSize
    @State var erasing = false
    /// Not `private`: `+Undo`'s `commitDrag` clears it.
    @State var dragPoints: [CGPoint] = []
    /// The compiled-in fallback `MaskUndo.resolve` falls back to when
    /// `@Environment(\.undoManager)` has nothing to hand out -- a sheet is
    /// not guaranteed a first responder SwiftUI is willing to give one to.
    /// Not `private`: `+Undo` reads it too.
    @State var ownUndo = UndoManager()
    @State private var pendingDestruction: LibraryActions.Destruction?

    var body: some View {
        VStack(spacing: 0) {
            toolbar
                .padding(10)
            Divider()
            canvas
                .frame(width: canvasSize.width, height: canvasSize.height)
        }
        // Never narrower than the toolbar's own words: a square source gives
        // a 480pt canvas, and at that width Cancel truncated to "Can…".
        .frame(minWidth: Self.maxCanvasSize.width)
        .background(hiddenShortcuts)
        .task { loadSource() }
        .destructionDialog($pendingDestruction)
    }

    // MARK: - Canvas

    private var canvas: some View {
        Canvas { context, size in
            if let source {
                context.opacity = 0.45
                context.draw(Image(nsImage: source), in: CGRect(origin: .zero, size: size))
                context.opacity = 1
            }
            context.drawLayer { layer in
                for stroke in strokes.strokes { draw(stroke, in: &layer) }
                if !dragPoints.isEmpty {
                    draw(MaskStroke(points: dragPoints, radius: brushSize, erases: erasing), in: &layer)
                }
            }
        }
        .contentShape(Rectangle())
        .gesture(paintGesture)
    }

    private func draw(_ stroke: MaskStroke, in layer: inout GraphicsContext) {
        layer.blendMode = stroke.erases ? .clear : .normal
        let radius = stroke.radius * displayScale
        let path = Path { path in
            for point in stroke.points {
                let center = CGPoint(x: point.x * displayScale, y: point.y * displayScale)
                path.addEllipse(in: CGRect(x: center.x - radius, y: center.y - radius,
                                           width: radius * 2, height: radius * 2))
            }
        }
        layer.fill(path, with: .color(Color.accentColor.opacity(0.55)))
    }

    private var paintGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let location = value.location
                dragPoints.append(CGPoint(x: location.x / displayScale, y: location.y / displayScale))
            }
            .onEnded { _ in commitDrag() }
    }

    // MARK: - Actions
    //
    // Not `private`: `MaskEditorSheet+Toolbar` calls every one of these.

    func stepBrush(_ delta: Int) {
        brushSize = MaskBrush.stepped(brushSize, by: delta)
    }

    func finish() {
        draft.media.maskImage = MaskRender.png(strokes, size: sourcePixelSize)?.base64EncodedString()
        dismiss()
    }

    func requestCancel() {
        guard !strokes.isEmpty else {
            dismiss()
            return
        }
        pendingDestruction = LibraryActions.Destruction(
            title: "Discard this mask?",
            message: "The strokes you painted have not been saved.",
            verb: "Discard",
            perform: { dismiss() }
        )
    }
}
