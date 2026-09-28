import MoldClient
import PencilKit
import SwiftUI

/// Paint where to change the picture (DESIGN.md §5.1): the source picture,
/// with a PencilKit canvas over it. White is "change this", like the Mac's
/// mask; saved at the source's own pixel size. Undo is PencilKit's own.
struct MaskEditor: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.dismiss) private var dismiss
    @State private var canvas = PKCanvasView()

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                ZStack {
                    if let data = generate.draft.media.sourceImage.flatMap({ Data(base64Encoded: $0) }),
                       let picture = UIImage(data: data) {
                        Image(uiImage: picture).resizable().scaledToFit()
                            .accessibilityLabel("The picture you are painting over")
                    }
                    MaskCanvas(canvas: canvas)
                }
                .frame(width: geometry.size.width, height: geometry.size.height)
            }
            .background(.black)
            .navigationTitle("Paint Where to Change")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) { Button("Use Mask") { save() } }
                ToolbarItemGroup(placement: .bottomBar) {
                    Button { canvas.undoManager?.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                    Button { canvas.undoManager?.redo() } label: { Label("Redo", systemImage: "arrow.uturn.forward") }
                    Spacer()
                    Button("Clear", role: .destructive) { canvas.drawing = PKDrawing() }
                }
            }
        }
    }

    /// The strokes as a white-on-black mask at the source's pixel size.
    private func save() {
        guard let pixels = generate.draft.media.sourceImagePixels else { dismiss(); return }
        let size = CGSize(width: pixels.width, height: pixels.height)
        let bounds = canvas.bounds
        let scale = min(bounds.width / size.width, bounds.height / size.height)
        let drawn = CGRect(x: (bounds.width - size.width * scale) / 2, y: (bounds.height - size.height * scale) / 2,
                           width: size.width * scale, height: size.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let mask = UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: size))
            let strokes = canvas.drawing.image(from: drawn, scale: 1 / scale)
            strokes.withTintColor(.white).draw(in: CGRect(origin: .zero, size: size))
        }
        generate.draft.media.maskImage = mask.pngData()?.base64EncodedString()
        dismiss()
    }
}

private struct MaskCanvas: UIViewRepresentable {
    let canvas: PKCanvasView

    func makeUIView(context: Context) -> PKCanvasView {
        canvas.backgroundColor = .clear
        canvas.isOpaque = false
        canvas.drawingPolicy = .anyInput
        canvas.tool = PKInkingTool(.marker, color: .white.withAlphaComponent(0.7), width: 40)
        canvas.accessibilityLabel = "Mask canvas. Paint over the parts to change."
        return canvas
    }

    func updateUIView(_ view: PKCanvasView, context: Context) {}
}
