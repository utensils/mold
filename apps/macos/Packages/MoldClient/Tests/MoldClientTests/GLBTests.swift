import Foundation
import Testing

@testable import MoldClient

/// The `.glb` reader, ported from `studio/lib/glb.test.ts` in full.
///
/// **Fails today**: this app had no GLB reader at all — a mesh print stopped
/// at the server's poster, because `NSImage(data:)` returns nil for a GLB and
/// nothing on macOS previews one.
///
/// The malformed-container matrix is the point: the gallery hands these bytes
/// straight off the wire, so a hostile file must fail with a SENTENCE and
/// never read past the buffer.
@Suite struct GLBSuite {

    private func reason(_ body: () throws -> Void) -> String {
        do {
            try body()
            return ""
        } catch let error as GLBParseError {
            return error.description
        } catch {
            return "threw \(type(of: error)) rather than a GLBParseError"
        }
    }

    private func triple(_ data: [Float], _ index: Int) -> SIMD3<Double> {
        SIMD3(Double(data[index * 3]), Double(data[index * 3 + 1]),
              Double(data[index * 3 + 2]))
    }

    // MARK: - A well-formed mold mesh

    @Test func readsPositionsIndicesBoundsAndCountsExactly() throws {
        let mesh = try GLB.parse(GLBFixture.triangleGLB())
        #expect(mesh.positions == GLBFixture.triangle.positions)
        #expect(mesh.indices == [0, 1, 2])
        #expect(mesh.bounds.min == SIMD3(0, 0, -1))
        #expect(mesh.bounds.max == SIMD3(2, 4, 0))
        #expect(mesh.vertexCount == 3)
        #expect(mesh.triangleCount == 1)
    }

    @Test func reportsAbsentOptionalAttributesAsNil() throws {
        let mesh = try GLB.parse(GLBFixture.triangleGLB())
        #expect(mesh.uvs == nil)
        #expect(mesh.colors == nil)
        #expect(mesh.baseColorTexture == nil)
    }

    @Test func readsUVsColorsAndSuppliedNormalsWhenPresent() throws {
        var spec = GLBFixture.triangle
        spec.uvs = [0, 0, 1, 0, 0, 1]
        spec.colors = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        spec.normals = [0, 0, 1, 0, 0, 1, 0, 0, 1]
        let built = GLBFixture.buildDocument(spec)
        let mesh = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        #expect(mesh.uvs == [0, 0, 1, 0, 0, 1])
        #expect(mesh.colors == [1, 0, 0, 0, 1, 0, 0, 0, 1])
        #expect(mesh.normals == [0, 0, 1, 0, 0, 1, 0, 0, 1])
    }

    /// A VEC4 `COLOR_0` is accepted with its alpha DROPPED rather than the
    /// whole file refused — the renderer wants RGB.
    @Test func dropsTheAlphaOfAVEC4VertexColour() throws {
        var spec = GLBFixture.triangle
        spec.colors = [1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, 0.5]
        var built = GLBFixture.buildDocument(spec)
        var accessors = built.json["accessors"] as! [[String: Any]]
        // The fixture writes COLOR_0 as VEC3; widen it in place.
        accessors[2]["type"] = "VEC4"
        built.json["accessors"] = accessors
        let mesh = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        #expect(mesh.colors == [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }

    @Test func surfacesAnEmbeddedBaseColourImageWithItsMediaType() throws {
        var spec = GLBFixture.triangle
        spec.uvs = [0, 0, 1, 0, 0, 1]
        spec.png = [137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3]
        let built = GLBFixture.buildDocument(spec)
        let texture = try GLB.parse(GLBFixture.assemble(built.json, built.bin)).baseColorTexture
        #expect(texture?.mimeType == "image/png")
        #expect(texture.map { [UInt8]($0.data) } == spec.png)
    }

    /// A `uri` image is a second network fetch, which is not something a mesh
    /// file gets to ask for: the mesh still shades, with its vertex colours.
    @Test func neverFollowsAUriImage() throws {
        var spec = GLBFixture.triangle
        spec.uvs = [0, 0, 1, 0, 0, 1]
        spec.png = [1, 2, 3, 4]
        var built = GLBFixture.buildDocument(spec)
        built.json["images"] = [["uri": "https://example.invalid/skin.png"]]
        #expect(reason {
            _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        }.contains("image is missing \"bufferView\""))
    }

    // MARK: - Malformed containers

    @Test func rejectsABufferTooSmallToHoldAHeader() {
        #expect(reason { _ = try GLB.parse([UInt8](repeating: 0, count: 8)) }
            .contains("12-byte header"))
    }

    @Test func rejectsBadMagicNamingTheMagic() {
        #expect(reason {
            _ = try GLB.parse(GLBFixture.triangleGLB(.init(magic: "GLTF")))
        }.contains("bad magic"))
    }

    @Test func rejectsGLTF1NamingTheVersion() {
        #expect(reason {
            _ = try GLB.parse(GLBFixture.triangleGLB(.init(version: 1)))
        }.contains("unsupported GLB version 1"))
    }

    @Test func rejectsADeclaredLengthThatDisagreesWithTheBuffer() {
        let buffer = GLBFixture.triangleGLB()
        #expect(reason {
            _ = try GLB.parse(GLBFixture.triangleGLB(.init(totalLength: buffer.count + 4)))
        }.contains("length mismatch"))
    }

    @Test func rejectsABINChunkThatRunsPastTheFile() {
        #expect(reason {
            _ = try GLB.parse(GLBFixture.triangleGLB(.init(binChunkLength: 4096)))
        }.contains("truncated GLB BIN chunk"))
    }

    @Test func rejectsAnAccessorThatReadsPastTheEndOfItsBufferView() {
        var built = GLBFixture.buildDocument(GLBFixture.triangle)
        var accessors = built.json["accessors"] as! [[String: Any]]
        accessors[0]["count"] = 999
        built.json["accessors"] = accessors
        let message = reason { _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin)) }
        #expect(message.contains("POSITION accessor reads"))
        #expect(message.contains("past the end of the buffer"))
    }

    @Test func rejectsAnIndexThatPointsPastTheVertices() {
        var spec = GLBFixture.triangle
        spec.indices = [0, 1, 7]
        let built = GLBFixture.buildDocument(spec)
        #expect(reason {
            _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        }.contains("index 7 at position 2 is past the 3-vertex"))
    }

    @Test func rejectsAJSONChunkThatIsNotJSON() {
        var buffer = GLBFixture.assemble(["asset": ["version": "2.0"]], [])
        // Corrupt the first byte of the JSON payload, which starts at byte 20.
        buffer[20] = 0x7B + 1
        #expect(reason { _ = try GLB.parse(buffer) }.contains("not valid JSON"))
    }

    @Test func throwsAParseErrorRatherThanACrashOnAGuttedDocument() {
        let buffer = GLBFixture.assemble(["asset": ["version": "2.0"]], [])
        #expect(reason { _ = try GLB.parse(buffer) }.contains("\"meshes\" array"))
    }

    /// A bufferView pointing at a buffer that is not the embedded BIN chunk is
    /// an EXTERNAL file, which this reader will not go and get.
    @Test func rejectsABufferViewThatIsNotTheEmbeddedBIN() {
        var built = GLBFixture.buildDocument(GLBFixture.triangle)
        var views = built.json["bufferViews"] as! [[String: Any]]
        views[0]["buffer"] = 1
        built.json["bufferViews"] = views
        #expect(reason {
            _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        }.contains("points at buffer 1"))
    }

    @Test func rejectsAPrimitiveThatIsNotTriangles() {
        var built = GLBFixture.buildDocument(GLBFixture.triangle)
        var meshes = built.json["meshes"] as! [[String: Any]]
        var primitives = meshes[0]["primitives"] as! [[String: Any]]
        primitives[0]["mode"] = 1
        meshes[0]["primitives"] = primitives
        built.json["meshes"] = meshes
        #expect(reason {
            _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        }.contains("mode 1 is not triangles"))
    }

    @Test func rejectsASparseAccessor() {
        var built = GLBFixture.buildDocument(GLBFixture.triangle)
        var accessors = built.json["accessors"] as! [[String: Any]]
        accessors[0]["sparse"] = ["count": 1]
        built.json["accessors"] = accessors
        #expect(reason {
            _ = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        }.contains("sparse"))
    }

    /// A file bigger than the cap is refused BEFORE it is walked, so a
    /// hostile 4 GB "mesh" cannot wedge the app. The cap the app ships with
    /// is the viewer's own (`MeshViewer.vue:225`).
    @Test func refusesAFileOverTheCapBeforeReadingIt() throws {
        let valid = Data(GLBFixture.triangleGLB())
        #expect(GLB.maximumBytes == 256 * 1024 * 1024)
        #expect(throws: GLBParseError.self) {
            _ = try GLB.parse(valid, maximumBytes: valid.count - 1)
        }
        // One byte more room and the same bytes parse, so the refusal is the
        // cap and not something else about the file.
        #expect(try GLB.parse(valid, maximumBytes: valid.count).triangleCount == 1)
    }

    // MARK: - Index component types

    @Test(arguments: [GLBAccessor.unsignedByte, GLBAccessor.unsignedShort,
                      GLBAccessor.unsignedInt])
    func readsEveryIndexWidthAsTheSameTriangleList(_ componentType: Int) throws {
        var spec = GLBFixture.MeshSpec(positions: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                                       indices: [0, 1, 2, 2, 1, 3])
        spec.indexComponentType = componentType
        let built = GLBFixture.buildDocument(spec)
        let mesh = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        #expect(mesh.indices == [0, 1, 2, 2, 1, 3])
        #expect(mesh.triangleCount == 2)
    }

    // MARK: - Generated normals

    /// A regular tetrahedron centred on the origin: every vertex normal must
    /// end up pointing straight away from the centre, so "outward" is
    /// checkable without knowing anything about the smoothing weights.
    private var tetrahedron: GLBFixture.MeshSpec {
        GLBFixture.MeshSpec(positions: [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1],
                            // Counter-clockwise seen from outside, the winding
                            // glTF calls front-facing.
                            indices: [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])
    }

    @Test func generatesUnitLengthNormalsThatFaceOutward() throws {
        let built = GLBFixture.buildDocument(tetrahedron)
        let mesh = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        #expect(mesh.normals.count == mesh.positions.count)
        for vertex in 0..<mesh.vertexCount {
            let normal = triple(mesh.normals, vertex)
            let position = triple(mesh.positions, vertex)
            let length = (normal.x * normal.x + normal.y * normal.y
                + normal.z * normal.z).squareRoot()
            #expect(abs(length - 1) < 1e-5)
            // The outward direction at a tetrahedron corner IS the corner
            // direction.
            let outward = (position.x * position.x + position.y * position.y
                + position.z * position.z).squareRoot()
            let dot = (normal.x * position.x + normal.y * position.y
                + normal.z * position.z) / outward
            #expect(abs(dot - 1) < 1e-5)
        }
    }

    @Test func prefersTheFilesOwnNormalsOverGeneratedOnes() throws {
        var spec = tetrahedron
        spec.normals = Array(repeating: [Float(0), 1, 0], count: 4).flatMap { $0 }
        let built = GLBFixture.buildDocument(spec)
        let mesh = try GLB.parse(GLBFixture.assemble(built.json, built.bin))
        #expect(triple(mesh.normals, 0) == SIMD3(0, 1, 0))
    }
}
