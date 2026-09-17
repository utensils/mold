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
        let snapshot = state.withLock { state -> (MeshScene, ViewerCamera, Bool, Double)? in
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
            return (scene, state.camera, state.wireframe, state.framedExtent)
        }
        guard let snapshot,
              let descriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let buffer = queue.makeCommandBuffer(),
              let encoder = buffer.makeRenderCommandEncoder(descriptor: descriptor)
        else { return }
        encode(into: encoder, view: view, scene: snapshot.0, camera: snapshot.1,
               wireframe: snapshot.2, extent: snapshot.3)
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }

    /// Split out so `MeshRenderer+Draw` owns the composition.
    private func encode(into encoder: any MTLRenderCommandEncoder, view: MTKView,
                        scene: MeshScene, camera: ViewerCamera, wireframe: Bool,
                        extent: Double) {
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        draw(scene, camera: camera, wireframe: wireframe, extent: extent,
             size: view.drawableSize, into: encoder)
    }
}
