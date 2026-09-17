import AppKit
import Metal
import MoldClient
import Testing

@testable import Mold

/// The half of the 3-D view that only the app bundle can answer for: that the
/// Metal shaders are IN the bundle under the names the pipeline asks for, that
/// a focused mesh view claims the arrows, and that the wireframe toggle
/// reflects what the GPU actually accepted.
///
/// **Fails today**: there was no mesh view. A mesh print stopped at its
/// poster, drawn at the opacity this app uses for "still loading", forever.
@MainActor
@Suite struct MeshViewerSuite {

    private func renderer() throws -> MeshRenderer {
        try MeshRenderer(pixelFormat: .bgra8Unorm, depthFormat: .depth32Float)
    }

    /// One triangle, built as the parser would hand it over.
    private func payload(indices: [UInt32] = [0, 1, 2]) -> MeshPayload {
        let positions: [Float] = [0, 0, 0, 2, 0, 0, 0, 4, -1]
        return MeshPayload(
            mesh: ParsedMesh(positions: positions,
                             normals: [Float](repeating: 0, count: 9),
                             uvs: nil, colors: nil, indices: indices,
                             baseColorTexture: nil,
                             bounds: MeshBounds(min: SIMD3(0, 0, -1), max: SIMD3(2, 4, 0)),
                             vertexCount: 3, triangleCount: indices.count / 3),
            texture: nil)
    }

    /// The shaders live in a `.metal` file compiled into the app, so this is
    /// the one test that can say they are THERE and that
    /// `mold_mesh_vertex` / `mold_mesh_fragment` are still their names. A
    /// rename would otherwise surface as a poster with a sentence under it.
    @Test func theShadersAreInTheBundleUnderTheNamesThePipelineAsksFor() throws {
        let renderer = try renderer()
        #expect(renderer.camera == MeshViewerCamera.homeCamera())
        #expect(!renderer.isWireframe)
        #expect(!renderer.sceneHasEdges)
    }

    /// The camera opens on the poster's own, and `0` returns it there from
    /// wherever a drag left it.
    @Test func opensOnThePosterCameraAndReturnsToIt() throws {
        let renderer = try renderer()
        renderer.setCamera(MeshInteraction.orbit(renderer.camera, dx: 2, dy: 0.5))
        #expect(renderer.camera != MeshViewerCamera.homeCamera())
        renderer.setCamera(MeshViewerCamera.homeCamera())
        #expect(renderer.camera == MeshViewerCamera.homeCamera())
    }

    /// The toggle reports what the GPU accepted, so a control can never show
    /// an overlay that will not draw.
    @Test func togglesTheWireframeOnlyForAMeshWithEdges() throws {
        let renderer = try renderer()
        #expect(!renderer.toggleWireframe(), "no mesh installed")

        let scene = try #require(MeshScene(payload(), device: renderer.device))
        renderer.install(scene)
        #expect(renderer.sceneHasEdges)
        #expect(renderer.toggleWireframe())
        #expect(renderer.isWireframe)
        // The edge buffer is built ONCE and reused, so the second turn on is
        // the same answer rather than a second upload.
        #expect(!renderer.toggleWireframe())
        #expect(renderer.toggleWireframe())

        // Installing another mesh clears it: the new one may have no edges.
        renderer.install(scene)
        #expect(!renderer.isWireframe)
    }

    /// Every triangle degenerate: nothing to outline, so the toggle refuses
    /// and the menu drops the row rather than offering an inert one.
    @Test func refusesTheWireframeForAMeshWithNoEdges() throws {
        let renderer = try renderer()
        let scene = try #require(MeshScene(payload(indices: [1, 1, 1]),
                                           device: renderer.device))
        renderer.install(scene)
        #expect(!renderer.sceneHasEdges)
        #expect(!renderer.toggleWireframe())
        #expect(!renderer.isWireframe)
    }

    /// Releasing drops the scene, which is what frees the buffers and the
    /// texture when the view goes away.
    @Test func releasingDropsTheMesh() throws {
        let renderer = try renderer()
        renderer.install(try #require(MeshScene(payload(), device: renderer.device)))
        renderer.release()
        #expect(!renderer.sceneHasEdges)
    }

    /// A focused mesh view ORBITS with the arrows, so the Library viewer's
    /// window-scoped previous/next stand down while it has focus.
    @Test func aFocusedMeshViewClaimsTheArrows() throws {
        let view = MeshMetalView(renderer: try renderer())
        #expect(ArrowKeyClaim.claims(view))
        #expect(view.acceptsFirstResponder)
        // The rule is about THIS view, not about every view.
        #expect(!ArrowKeyClaim.claims(NSButton()))
    }

    /// **Fails today**: only the CLAIM was tested. Giving the arrows BACK --
    /// closing the mesh, stepping to a picture, clicking into the grid -- is
    /// the half that decides whether the Library still works afterwards, and
    /// nothing said a thing about it.
    @Test func givesTheArrowsBackWhenNothingClaimsThem() throws {
        let view = MeshMetalView(renderer: try renderer())
        #expect(ArrowKeyClaim.claims(view))
        // Every responder the viewer falls back to once the mesh is gone.
        #expect(!ArrowKeyClaim.claims(nil))
        #expect(!ArrowKeyClaim.claims(NSView()))
        #expect(!ArrowKeyClaim.claims(NSImageView()))
        // And the responders that claimed them BEFORE this lane still do.
        #expect(ArrowKeyClaim.claims(NSSlider()))
        #expect(ArrowKeyClaim.claims(NSTextField()))
    }

    /// Both edges are announced, so the viewer is told rather than polling a
    /// notification that is not about first responders.
    @Test func announcesTakingAndGivingUpFirstResponder() throws {
        let view = MeshMetalView(renderer: try renderer())
        let counter = Counter()
        let token = NotificationCenter.default.addObserver(
            forName: MeshMetalView.claimChanged, object: view, queue: nil
        ) { _ in counter.bump() }
        defer { NotificationCenter.default.removeObserver(token) }
        _ = view.becomeFirstResponder()
        _ = view.resignFirstResponder()
        #expect(counter.count == 2)
    }

    /// A tiny box, because a notification block is not `@MainActor`.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func bump() { lock.withLock { value += 1 } }
        var count: Int { lock.withLock { value } }
    }

    private func key(_ characters: String, shift: Bool = false) -> MeshInteraction.Key? {
        let event = NSEvent.keyEvent(
            with: .keyDown, location: .zero,
            modifierFlags: shift ? [.shift] : [], timestamp: 0, windowNumber: 0,
            context: nil, characters: characters,
            charactersIgnoringModifiers: characters, isARepeat: false, keyCode: 0)
        return event.flatMap(MeshMetalView.key(for:))
    }

    /// The arrows carry no printable character, so they are matched by their
    /// special-key value while `+`, `-` and `0` come from the shared table.
    @Test func readsEveryKeyTheReferenceBinds() {
        #expect(key(String(UnicodeScalar(NSLeftArrowFunctionKey)!)) == .orbitLeft)
        #expect(key(String(UnicodeScalar(NSRightArrowFunctionKey)!)) == .orbitRight)
        #expect(key(String(UnicodeScalar(NSUpArrowFunctionKey)!)) == .orbitUp)
        #expect(key(String(UnicodeScalar(NSDownArrowFunctionKey)!)) == .orbitDown)
        #expect(key("+") == .zoomIn)
        #expect(key("-") == .zoomOut)
        #expect(key("0") == .home)
        // Anything else falls through to the responder chain untouched.
        #expect(key("k") == nil)
    }

    /// A refusal keeps the READER's sentence, which names what was wrong with
    /// the file; anything else gets the transport one.
    @Test func putsTheReasonOnThePosterRatherThanJustAFailure() {
        let parse = MeshViewFailure.reading(GLBParseError("GLB has no meshes"))
        #expect(parse.sentence.contains("GLB has no meshes"))
        #expect(MeshViewFailure.noDevice.sentence.contains("poster"))
        // Every failure says something; a black rectangle is never an answer.
        for failure in [MeshViewFailure.noDevice, .shaders("x"), .upload,
                        .unreadable("y"), .transport("z")] {
            #expect(failure.sentence.count > 20)
        }
    }
}
