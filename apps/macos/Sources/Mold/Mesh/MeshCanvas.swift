import AppKit
import MoldClient
import SwiftUI

/// A mesh print, interactive, with the poster underneath it.
///
/// The one 3-D surface in the app: the Library's viewer and the Generate
/// result canvas both mount this. It owns loading, the failure sentence and
/// the controls; `MeshView` owns the drawing and `MeshRenderer` the camera.
///
/// EVERY failure lands on the poster with one line saying why -- no Metal
/// device, a refused download, a corrupt file, a shader that will not build.
/// A black rectangle is never an answer.
struct MeshCanvas: View {
    /// How to get the bytes, and what identity a reload is keyed on.
    let printID: String
    let fetch: @Sendable () async throws -> Data
    let poster: NSImage?
    let alt: String
    /// The Library's viewer tours a mesh; a Generate result does not.
    let offersAutoRotate: Bool
    let exports: MeshExport.Split
    let canSave: Bool
    let canShowInLibrary: Bool
    /// Only the Library's viewer knows the print behind the mesh.
    var canReuse = false
    let perform: (MeshViewAction) -> Void

    /// Not `private`: the controls live in `+Controls` for size, and `private`
    /// does not cross files for the same type.
    @State var scene: MeshScene?
    @State var renderer: MeshRenderer?
    @State private var note: String?
    @State var wireframe = false
    @State var autoRotating = false
    /// Retired by the first interaction, exactly as the reference retires it:
    /// once a person has touched the viewer it is theirs for this mount.
    @State var wantsTour = true
    @State var redrawToken = 0
    @State var resetToken = 0

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                posterLayer
                if let renderer, let scene {
                    MeshView(renderer: renderer, scene: scene, autoRotate: tours,
                             redrawToken: redrawToken, resetToken: resetToken,
                             onAutoRotatingChange: { autoRotating = $0 },
                             onInteraction: { wantsTour = false })
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let note {
                Text(note)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .accessibilityAddTraits(.isStaticText)
            } else if scene != nil {
                controls
            }
        }
        .rowActionMenu(plan.items, perform: handle)
        .task(id: printID) { await load() }
    }

    var tours: Bool { offersAutoRotate && wantsTour }

    /// Kept under the mesh while it loads and forever on any failure: the
    /// gallery already has this picture, and it IS the view's home frame.
    @ViewBuilder private var posterLayer: some View {
        if let poster {
            Image(nsImage: poster)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
                .accessibilityLabel(alt)
                .opacity(scene == nil ? 1 : 0)
        } else if scene == nil {
            Image(systemName: "cube")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
                .accessibilityLabel(alt)
        }
    }

    var plan: MeshViewMenuPlan {
        MeshViewMenuPlan(exports: exports, hasEdges: scene?.hasEdges ?? false,
                         isWireframe: wireframe, isAutoRotating: autoRotating,
                         offersAutoRotate: offersAutoRotate && scene != nil,
                         canSave: canSave, canShowInLibrary: canShowInLibrary,
                         canReuse: canReuse)
    }

    private func load() async {
        scene = nil
        note = nil
        wireframe = false
        // The tour is per MESH, not per mount: stepping to the next print
        // starts a fresh one, and only an interaction with THAT one ends it.
        wantsTour = true
        do {
            let bytes = try await fetch()
            let payload = try await Task.detached(priority: .userInitiated) {
                try MeshPayload.load(bytes)
            }.value
            let renderer = try self.renderer ?? MeshRenderer(pixelFormat: .bgra8Unorm,
                                                             depthFormat: .depth32Float)
            guard let built = MeshScene(payload, device: renderer.device) else {
                throw MeshViewFailure.upload
            }
            self.renderer = renderer
            scene = built
        } catch is CancellationError {
            // A new print replaced this one; its own load reports for itself.
        } catch let failure as MeshViewFailure {
            note = failure.sentence
        } catch {
            note = MeshViewFailure.reading(error).sentence
        }
    }
}
