import Metal
import MetalKit

/// The Metal objects a mesh view needs before it can draw anything.
///
/// Built once per view and handed to `MeshRenderer`. Separate from it so the
/// renderer is the camera and the draw, and so every way this can FAIL -- no
/// device, no queue, a shader that will not build -- is one place with one
/// answer: throw, and let the poster stand.
nonisolated struct MeshMetalStack {
    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let pipeline: any MTLRenderPipelineState
    let depthState: any MTLDepthStencilState

    static func make(pixelFormat: MTLPixelFormat,
                     depthFormat: MTLPixelFormat) throws -> MeshMetalStack {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw MeshViewFailure.noDevice }

        let library: any MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            throw MeshViewFailure.shaders(error.localizedDescription)
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = library.makeFunction(name: "mold_mesh_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "mold_mesh_fragment")
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        descriptor.depthAttachmentPixelFormat = depthFormat
        let pipeline: any MTLRenderPipelineState
        do {
            pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw MeshViewFailure.shaders(error.localizedDescription)
        }

        // Ordinary depth testing, writes on: the reference enables
        // `gl.DEPTH_TEST` and nothing else.
        let depth = MTLDepthStencilDescriptor()
        depth.depthCompareFunction = .less
        depth.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else {
            throw MeshViewFailure.noDevice
        }
        return MeshMetalStack(device: device, queue: queue, pipeline: pipeline,
                              depthState: depthState)
    }
}
