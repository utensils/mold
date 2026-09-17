import Foundation
import Metal
import MoldClient

/// One mesh on the GPU, plus the framing facts the camera needs.
///
/// Immutable except for the edge buffer, which is built on the FIRST wireframe
/// toggle and never rebuilt -- a mesh nobody outlines never pays for the
/// deduplication (`MeshViewer.vue:347-368`).
///
/// `@unchecked Sendable` because Metal's own buffer and texture types are not
/// Sendable and the one mutable field is the edge buffer: every reader and the
/// single writer go through `MeshRenderer`'s lock, which is what actually
/// serialises them. The same answer `QuickLook` gives for QuickLookUI's
/// off-main getters.
nonisolated final class MeshScene: @unchecked Sendable {
    let positions: any MTLBuffer
    let normals: any MTLBuffer
    let colors: any MTLBuffer
    let uvs: any MTLBuffer
    let indices: any MTLBuffer
    let indexCount: Int
    let texture: (any MTLTexture)?
    /// World-space bounding-box centre: the point every camera orbits. mold
    /// centres a mesh on the query GRID, never on the model.
    let center: SIMD3<Double>
    /// Half the bounding-box diagonal. The DEPTH range only -- never the fit.
    let radius: Double
    /// The half-extent the poster and the turntable frame to, at the poster's
    /// own elevation, so this view's home IS the thumbnail. Never recomputed.
    let extent: Double
    /// `(radial, |dy|)` per vertex, so a tilt can re-frame without the mesh.
    let profile: [Float]
    let bounds: MeshBounds
    let vertexCount: Int
    let triangleCount: Int
    /// False for a mesh whose every triangle is degenerate: nothing to outline.
    let hasEdges: Bool

    private(set) var edges: (any MTLBuffer)?
    private(set) var edgeCount = 0
    private let sourceIndices: [UInt32]

    init?(_ payload: MeshPayload, device: any MTLDevice) {
        let mesh = payload.mesh
        let vertexCount = mesh.vertexCount
        let defaultColour: [Float] = Array(
            repeating: [0.82, 0.82, 0.86], count: vertexCount).flatMap { $0 }
        guard let positions = device.buffer(mesh.positions),
              let normals = device.buffer(mesh.normals),
              // A constant vertex attribute costs no buffer in GL; Metal has
              // no such thing, so an untextured or uncoloured mesh carries the
              // default filled in rather than a second pipeline.
              let colors = device.buffer(mesh.colors ?? defaultColour),
              let uvs = device.buffer(mesh.uvs ?? [Float](repeating: 0, count: vertexCount * 2)),
              let indices = mesh.indices.withUnsafeBytes({ bytes in
                  device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                    options: .storageModeShared)
              })
        else { return nil }

        self.positions = positions
        self.normals = normals
        self.colors = colors
        self.uvs = uvs
        self.indices = indices
        indexCount = mesh.indices.count
        texture = payload.texture.flatMap { device.texture($0) }
        center = mesh.bounds.center
        radius = mesh.bounds.radius
        profile = MeshViewerCamera.sweepProfile(mesh.positions, center: mesh.bounds.center)
        extent = MeshViewerCamera.sweepExtentOfProfile(
            profile, elevationRad: MeshViewerCamera.homeCamera().pitch)
        bounds = mesh.bounds
        self.vertexCount = vertexCount
        triangleCount = mesh.triangleCount
        hasEdges = MeshViewerMath.meshHasEdges(mesh.indices)
        sourceIndices = mesh.indices
    }

    /// Uploads the edge list, ONCE, the first time the overlay is switched on.
    /// False when the GPU refuses the buffer, so the control stays where it
    /// was rather than promising an overlay that will never draw.
    func ensureEdges(device: any MTLDevice) -> Bool {
        if edges != nil { return true }
        guard hasEdges else { return false }
        let list = MeshViewerMath.edgeIndices(sourceIndices)
        guard !list.isEmpty,
              let buffer = list.withUnsafeBytes({ bytes in
                  device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                    options: .storageModeShared)
              })
        else { return false }
        edges = buffer
        edgeCount = list.count
        return true
    }
}

private extension MTLDevice {
    nonisolated func buffer(_ values: [Float]) -> (any MTLBuffer)? {
        values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress, !bytes.isEmpty else { return nil }
            return makeBuffer(bytes: base, length: bytes.count, options: .storageModeShared)
        }
    }

    nonisolated func texture(_ image: MeshTextureImage) -> (any MTLTexture)? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: image.width, height: image.height,
            // No mipmaps: mold's textures are not power-of-two and the
            // reference samples LINEAR/CLAMP_TO_EDGE with none.
            mipmapped: false)
        descriptor.usage = .shaderRead
        guard let texture = makeTexture(descriptor: descriptor) else { return nil }
        image.rgba.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            texture.replace(
                region: MTLRegionMake2D(0, 0, image.width, image.height),
                mipmapLevel: 0, withBytes: base, bytesPerRow: image.width * 4)
        }
        return texture
    }
}
