import Metal
import MetalKit
import MoldClient
import os
import simd

/// What the shaders read, once per draw. The field order and types mirror
/// `MeshUniforms` in `MeshShaders.metal` exactly.
struct MeshUniforms {
    var modelView: simd_float4x4
    var projection: simd_float4x4
    var normalMatrix: simd_float3x3
    var hasTexture: Float
    var wireframe: Float
}

/// One draw's worth of state, read out under the renderer's lock in one go.
///
/// The edge buffer is part of it BECAUSE it is the one mutable thing a
/// `MeshScene` carries: copying the pointer out under the lock is what makes
/// `MeshScene`'s own "every reader goes through the lock" true.
struct MeshFrame {
    let scene: MeshScene
    let camera: ViewerCamera
    let extent: Double
    let edges: (any MTLBuffer)?
    let edgeCount: Int
}

/// The mesh view's Metal state and its camera.
///
/// **Deliberately not `@MainActor`.** `MTKViewDelegate`'s callbacks arrive off
/// the main actor under this target's default isolation, exactly as
/// QuickLookUI's do (`Library/QuickLook.swift`), so the mutable state sits
/// behind a lock rather than behind an isolation it cannot honour.
final class MeshRenderer: NSObject, MTKViewDelegate, @unchecked Sendable {
    /// Everything a draw reads, and the only thing an event writes.
    private struct State {
        var scene: MeshScene?
        var camera = MeshViewerCamera.homeCamera()
        var wireframe = false
        /// The last pitch the framing was solved for, and its answer. The
        /// sweep bound is invariant in AZIMUTH, not in elevation, so a tilted
        /// view needs its own extent or the silhouette runs off the frame;
        /// caching on the pitch means a yaw drag, the tour, a zoom and a
        /// resize never pay for it.
        var framedPitch = Double.nan
        var framedExtent = 0.0
    }

    let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLRenderPipelineState
    private let depthState: any MTLDepthStencilState
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// Fails -- rather than drawing a black rectangle -- when this Mac has no
    /// usable Metal device, or the shaders will not build.
    init(pixelFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {
        let stack = try MeshMetalStack.make(pixelFormat: pixelFormat,
                                            depthFormat: depthFormat)
        device = stack.device
        queue = stack.queue
        pipeline = stack.pipeline
        depthState = stack.depthState
        super.init()
    }

    // MARK: - What the view asks of it

    func install(_ scene: MeshScene?) {
        state.withLock {
            $0.scene = scene
            $0.wireframe = false
            $0.framedPitch = .nan
        }
    }

    /// Releases the GPU resources when the view goes away, so opening and
    /// closing the lightbox all session does not accumulate meshes.
    func release() { install(nil) }

    var camera: ViewerCamera { state.withLock { $0.camera } }
    var isWireframe: Bool { state.withLock { $0.wireframe } }
    var sceneHasEdges: Bool { state.withLock { $0.scene?.hasEdges ?? false } }

    func setCamera(_ camera: ViewerCamera) {
        state.withLock { $0.camera = camera }
    }

    /// Returns the wireframe state after the toggle. A GPU that refuses the
    /// edge buffer leaves it where it was.
    @discardableResult func toggleWireframe() -> Bool {
        let device = device
        return state.withLock { state in
            guard let scene = state.scene, scene.hasEdges else { return false }
            if state.wireframe {
                state.wireframe = false
            } else if scene.ensureEdges(device: device) {
                state.wireframe = true
            }
            return state.wireframe
        }
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        // EVERYTHING the draw reads is taken under the lock, the edge buffer
        // included. It used to hand the `MeshScene` reference out and read
        // `edges`/`edgeCount` unsynchronised while `toggleWireframe` could
        // write them from the main actor -- benign only by an invariant
        // nobody had written down, and a race on a GPU resource pointer the
        // moment any of it changed.
        let frame = state.withLock { state -> MeshFrame? in
            guard let scene = state.scene else { return nil }
            if state.camera.pitch != state.framedPitch {
                state.framedPitch = state.camera.pitch
                // Never below the poster's own extent, so the home view is the
                // poster's exact framing and a tilt only ever pulls BACK.
                state.framedExtent = Swift.max(
                    scene.extent,
                    MeshViewerCamera.sweepExtentOfProfile(
                        scene.profile, elevationRad: state.camera.pitch))
            }
            let overlay = state.wireframe && scene.edgeCount > 0
            return MeshFrame(scene: scene, camera: state.camera,
                             extent: state.framedExtent,
                             edges: overlay ? scene.edges : nil,
                             edgeCount: overlay ? scene.edgeCount : 0)
        }
        guard let frame,
              let descriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: descriptor)
        else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        draw(frame, size: view.drawableSize, into: encoder)
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }
}
