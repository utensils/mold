import Metal
import MetalKit

/// The Metal objects a mesh view needs before it can draw anything.
///
/// Built once per view and handed to `MeshRenderer`. Separate from it so the
/// renderer is the camera and the draw, and so every way this can FAIL -- no
/// device, no queue, a shader that will not build -- is one place with one
/// answer: throw, and let the poster stand.
public nonisolated struct MeshMetalStack {
    public let device: any MTLDevice
    public let queue: any MTLCommandQueue
    public let pipeline: any MTLRenderPipelineState
    public let depthState: any MTLDepthStencilState

    public static func make(pixelFormat: MTLPixelFormat,
                     depthFormat: MTLPixelFormat) throws -> MeshMetalStack {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw MeshViewFailure.noDevice }

        let library: any MTLLibrary
        do {
            library = try Self.shaderLibrary(on: device)
        } catch let failure as MeshViewFailure {
            throw failure
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

    /// The shaders. The two builds ship them differently: Xcode compiles a
    /// package's `.metal` into the resource bundle's `default.metallib` even
    /// when it is declared `.copy` (the apps), while `swift build` copies the
    /// source and never compiles it (`swift test`). The compiled library
    /// first, then the source -- a few milliseconds, once per view.
    static func shaderLibrary(on device: any MTLDevice) throws -> any MTLLibrary {
        if let compiled = try? device.makeDefaultLibrary(bundle: .module),
           compiled.functionNames.contains("mold_mesh_vertex") {
            return compiled
        }
        guard let url = Bundle.module.url(forResource: "MeshShaders", withExtension: "metal") else {
            throw MeshViewFailure.shaders("the mesh shaders are missing from this build")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        return try device.makeLibrary(source: source, options: nil)
    }
}
