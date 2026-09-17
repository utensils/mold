import Metal
import MetalKit
import MoldClient
import simd

// The draw itself, composed exactly as `MeshViewer.vue:277-345` composes it.
extension MeshRenderer {

    func draw(_ frame: MeshFrame, size: CGSize,
              into encoder: any MTLRenderCommandEncoder) {
        let scene = frame.scene
        let camera = frame.camera
        let extent = frame.extent
        // Backing-store pixels, not points: the scale factor cancels between
        // the fit and the half-extents, so a retina view frames the mesh
        // exactly as the server's poster does.
        let width = Double(size.width)
        let height = Double(size.height)
        // `zoom` is the pull-back factor the wheel, the pinch and +/- have
        // always spoken -- larger means further away -- so it DIVIDES the fit.
        let scale = MeshViewerCamera.orthographicScale(
            extent: extent, width: width, height: height,
            margin: MeshViewerCamera.POSTER_MARGIN) / camera.zoom
        // A mesh with no extent, or a view with no area, has nothing to frame:
        // the pass has already cleared, so leave it cleared rather than build
        // a projection out of a division by zero.
        guard scale > 0, scale.isFinite else { return }

        let distance = scene.radius * 3
        let modelView = MeshMatrix.modelView(camera: camera, center: scene.center,
                                             distance: distance)
        // The mesh sits within `radius` of the eye axis' centre, so these
        // planes bracket it whatever the orbit angle.
        let projection = MeshMatrix.orthographic(
            halfWidth: width / 2 / scale, halfHeight: height / 2 / scale,
            near: distance - scene.radius * 2, far: distance + scene.radius * 2)

        var uniforms = MeshUniforms(
            modelView: matrix(modelView),
            projection: metalDepth(matrix(projection)),
            normalMatrix: normalMatrix(MeshMatrix.upper3x3(modelView)),
            hasTexture: scene.texture == nil ? 0 : 1,
            wireframe: 0)

        encoder.setVertexBuffer(scene.positions, offset: 0, index: 0)
        encoder.setVertexBuffer(scene.normals, offset: 0, index: 1)
        encoder.setVertexBuffer(scene.colors, offset: 0, index: 2)
        encoder.setVertexBuffer(scene.uvs, offset: 0, index: 3)
        if let texture = scene.texture { encoder.setFragmentTexture(texture, index: 0) }
        // NO culling: mold writes `doubleSided` materials and the fragment
        // shader lights a back face by its flipped normal. glTF winds front
        // faces counter-clockwise, which Metal has to be told.
        encoder.setCullMode(.none)
        encoder.setFrontFacing(.counterClockwise)

        // Decided under the lock, with the buffer it names.
        let overlay = frame.edges != nil
        if overlay {
            // Pushing the filled triangles away from the eye keeps the edges
            // from z-fighting the very surface they outline -- the reference's
            // `polygonOffset(1, 1)`, whose `units` term is expressed here in
            // depth units rather than GL's smallest-resolvable-difference.
            encoder.setDepthBias(1e-4, slopeScale: 1, clamp: 0)
        }
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 4)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle, indexCount: scene.indexCount, indexType: .uint32,
            indexBuffer: scene.indices, indexBufferOffset: 0)

        guard let edges = frame.edges else { return }
        encoder.setDepthBias(0, slopeScale: 0, clamp: 0)
        uniforms.wireframe = 1
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 4)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 0)
        encoder.drawIndexedPrimitives(
            type: .line, indexCount: frame.edgeCount, indexType: .uint32,
            indexBuffer: edges, indexBufferOffset: 0)
    }

    private func matrix(_ m: Mat4) -> simd_float4x4 {
        simd_float4x4(columns: (SIMD4(m[0], m[1], m[2], m[3]),
                                SIMD4(m[4], m[5], m[6], m[7]),
                                SIMD4(m[8], m[9], m[10], m[11]),
                                SIMD4(m[12], m[13], m[14], m[15])))
    }

    private func normalMatrix(_ m: [Float]) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(m[0], m[1], m[2]),
                                SIMD3(m[3], m[4], m[5]),
                                SIMD3(m[6], m[7], m[8])))
    }

    /// `MeshMatrix.orthographic` is the REFERENCE's projection, mapping the
    /// near plane to -1 and the far plane to +1 -- GL's clip range, which a
    /// Rust test pins through the TypeScript it mirrors. Metal's is `[0, 1]`,
    /// so everything in front of the mesh's midpoint would be clipped away.
    /// Remapping here keeps the shared matrix the one definition of the
    /// framing and puts the one Metal-specific fact where it belongs.
    private func metalDepth(_ m: simd_float4x4) -> simd_float4x4 {
        var out = m
        out.columns.2.z = m.columns.2.z * 0.5
        out.columns.3.z = m.columns.3.z * 0.5 + 0.5
        return out
    }
}
