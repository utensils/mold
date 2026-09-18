import MoldClient
import SwiftUI

// The inline controls and what performing one means. They are drawn FROM the
// same `MeshViewMenuPlan` the contextual menu draws, so a control and a menu
// row can never offer different things or call them different names.
extension MeshCanvas {

    @ViewBuilder var controls: some View {
        HStack(spacing: 8) {
            Text(summary)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            ForEach(Array(plan.items.enumerated()), id: \.offset) { _, item in
                if item.isSubmenu {
                    Menu(item.title) {
                        RowActionMenu(actions: item.children, perform: handle)
                    }
                    .menuStyle(.button)
                    .fixedSize()
                } else if let kind = item.kind, Self.inlineKinds.contains(kind) {
                    Button(item.title) { handle(kind) }
                }
            }
        }
        .buttonStyle(.accessoryBar)
        .padding(.horizontal, 4)
    }

    /// The three view controls sit on the bar; Save and Show in Library live
    /// on the surrounding surface's own bar and would be a second copy here.
    static var inlineKinds: Set<MeshViewAction> {
        [.resetView, .toggleWireframe, .toggleAutoRotate]
    }

    /// `12,345 triangles · 6,789 points`, the caption under the mesh. The
    /// numbers are the file's own, so a person can tell a decimated render
    /// from a raw one without exporting it.
    var summary: String {
        guard let scene else { return "" }
        return "\(scene.triangleCount.formatted()) triangles · "
            + "\(scene.vertexCount.formatted()) points"
    }

    func handle(_ action: MeshViewAction) {
        switch action {
        case .resetView:
            resetToken += 1
        case .toggleWireframe:
            guard let renderer else { return }
            wireframe = renderer.toggleWireframe()
            redrawToken += 1
        case .toggleAutoRotate:
            // Stopping is permanent for this mount, exactly as an interaction
            // is; starting again is what the control offers until then.
            wantsTour.toggle()
        case .exportFile, .exportTurntable, .save, .showInLibrary, .reuse:
            perform(action)
        }
    }
}
