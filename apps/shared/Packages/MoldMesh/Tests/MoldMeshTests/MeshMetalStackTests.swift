import Metal
import Testing

@testable import MoldMesh

/// The shaders travel inside the package (`Bundle.module`), not the app: a
/// stack that cannot find them would leave every mesh on its poster.
struct MeshMetalStackTests {
    @Test(.enabled(if: MTLCreateSystemDefaultDevice() != nil, "needs a Metal device"))
    func theShadersBuildFromThePackageBundle() throws {
        let stack = try MeshMetalStack.make(pixelFormat: .bgra8Unorm, depthFormat: .depth32Float)
        #expect(stack.device.name.isEmpty == false)
    }
}
