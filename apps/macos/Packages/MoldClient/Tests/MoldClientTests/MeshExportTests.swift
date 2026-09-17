import Foundation
import Testing

@testable import MoldClient

/// Ported from `studio/lib/meshExport.test.ts`.
///
/// **Fails today**: `ExportOptions` carried a hard-coded `{obj, stl, ply, zip}`
/// set and posted `{"format": …}` and nothing else, so a host that added a
/// container was invisible, and the geometry knobs a slicer needs could not be
/// sent at all (review 03-L1).
@Suite struct MeshExportSuite {

    private let capabilities = MeshExportGeometryCapabilities(
        sizeMm: MeshSizeControl(min: 1, max: 1000, default: 100),
        upAxes: [.y, .z],
        origins: [.center, .floor],
        defaults: [
            "obj": MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor),
            "stl": MeshExportGeometry(sizeMm: 100, upAxis: .z, origin: .floor),
            "ply": MeshExportGeometry(sizeMm: 100, upAxis: .z, origin: .floor),
        ])

    /// A unit-ish box in the stored Y-up frame: 1 wide, 0.4286 tall, 0.6857 deep.
    private let bounds = MeshBounds(min: SIMD3(-0.5, -0.2143, -0.3429),
                                    max: SIMD3(0.5, 0.2143, 0.3428))

    private func body(_ request: MeshExportRequest) throws -> [String: Any] {
        let data = try MoldJSON.encoder.encode(request)
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    // MARK: - The split

    @Test func namesTheTurntableContainersWhateverTheirCase() {
        #expect(["gif", "APNG", " webp "].map(MeshExport.isAnimated) == [true, true, true])
        #expect(["obj", "stl", "ply", "usdz"].map(MeshExport.isAnimated)
            == [false, false, false, false])
    }

    @Test func splitsTheHostsListIntoGeometryFilesAndTurntablesInItsOrder() {
        let split = MeshExport.split(["obj", "stl", "ply", "gif", "apng", "webp"])
        #expect(split.files == ["obj", "stl", "ply"])
        #expect(split.animations == ["gif", "apng", "webp"])
    }

    /// The server lists the stored container first so a client can see what it
    /// holds; no menu should offer "Export as GLB" beside Save.
    @Test func dropsGLBTheStoredFormFromBothHalves() {
        let split = MeshExport.split(["glb", "obj", "GLB", "gif"])
        #expect(split.files == ["obj"])
        #expect(split.animations == ["gif"])
    }

    /// The server has already dropped any name IT does not know
    /// (`known_mesh_export_formats`, `types.rs:11999-12009`), so anything that
    /// reaches this client is a container the HOST will really write — and the
    /// app saves bytes to a file, which needs no per-format knowledge.
    @Test func keepsAContainerThisClientHasNeverHeardOfAsADirectTranscode() {
        #expect(MeshExport.split(["obj", "usdz"]).files == ["obj", "usdz"])
    }

    @Test func answersTwoEmptyListsForAHostThatAdvertisesNothing() {
        for advertised in [nil, []] as [[String]?] {
            #expect(MeshExport.split(advertised).files.isEmpty)
            #expect(MeshExport.split(advertised).animations.isEmpty)
        }
    }

    @Test func normalisesTheAdvertisedSpellingToLowerCase() {
        let split = MeshExport.split(["OBJ", "Gif"])
        #expect(split.files == ["obj"])
        #expect(split.animations == ["gif"])
    }

    /// An absent `mesh` block is a host with no mesh family at all, so there
    /// is nothing to convert — and never a list this app invented.
    @Test func aHostWithNoMeshBlockOffersNoExports() throws {
        let empty = try MoldJSON.decoder.decode(Capabilities.self,
                                                from: Data("{}".utf8))
        #expect(empty.mesh == nil)
        #expect(empty.meshExports.files.isEmpty)
        #expect(empty.meshExports.animations.isEmpty)
        #expect(empty.meshGeometryDefaults(for: "stl") == nil)
    }

    /// The whole block, decoded from the server's own spelling.
    @Test func readsTheHostsAdvertisedMeshBlockOffTheWire() throws {
        let json = """
        {"mesh": {"generation": true, "formats": ["glb"],
                  "export_formats": ["glb", "obj", "stl", "ply", "gif"],
                  "textures": true,
                  "export_geometry": {"size_mm": {"min": 1, "max": 1000, "default": 100},
                                      "up_axes": ["y", "z"],
                                      "origins": ["center", "floor"],
                                      "defaults": {"obj": {"size_mm": null, "up_axis": "y",
                                                           "origin": "floor"},
                                                   "stl": {"size_mm": 100, "up_axis": "z",
                                                           "origin": "floor"}}}}}
        """
        let capabilities = try MoldJSON.decoder.decode(Capabilities.self,
                                                       from: Data(json.utf8))
        #expect(capabilities.meshExports.files == ["obj", "stl", "ply"])
        #expect(capabilities.meshExports.animations == ["gif"])
        #expect(capabilities.meshGeometryDefaults(for: "stl")?.upAxis == .z)
        #expect(capabilities.meshGeometryDefaults(for: "obj")?.sizeMm == nil)
        // Advertised but not listed in `defaults`: no knobs, bare format.
        #expect(capabilities.meshGeometryDefaults(for: "ply") == nil)
    }

    // MARK: - Filenames

    @Test func keepsThePrintsOwnStemAndTakesTheRequestedExtension() {
        #expect(MeshExport.filename("armchair 01.glb", format: "obj") == "armchair 01.obj")
        #expect(MeshExport.filename("armchair__hunyuan3d__s7.glb", format: "stl")
            == "armchair__hunyuan3d__s7.stl")
        #expect(MeshExport.filename("armchair 01.glb", format: "GIF") == "armchair 01.gif")
    }

    @Test func fallsBackToAStemWhenTheNameHasNone() {
        #expect(MeshExport.filename("", format: "stl") == "mold-mesh.stl")
        #expect(MeshExport.filename(".glb", format: "stl") == "mold-mesh.stl")
    }

    // MARK: - Geometry options

    @Test func namesTheGeometryContainersAndExcludesTheStoredFormAndTurntables() {
        #expect(["obj", "STL", " ply "].map(MeshExport.takesGeometryOptions)
            == [true, true, true])
        #expect(["glb", "GLB", "gif", "apng", "webp"].map(MeshExport.takesGeometryOptions)
            == [false, false, false, false, false])
        // The host's own defaults table decides what it will actually scale;
        // this structural rule must not hide a container a future host adds.
        #expect(MeshExport.takesGeometryOptions("usdz"))
    }

    @Test func readsTheHostsPerFormatDefaults() {
        #expect(MeshExportGeometry.defaults(capabilities, format: "stl")
            == MeshExportGeometry(sizeMm: 100, upAxis: .z, origin: .floor))
        #expect(MeshExportGeometry.defaults(capabilities, format: "OBJ")
            == MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor))
    }

    /// The presence of the block is the ONLY gate. An older server DROPS the
    /// three keys instead of refusing them, so a client that guessed defaults
    /// would promise a resize the host never performed.
    @Test func answersNilForAHostThatAdvertisesNoGeometryBlock() {
        #expect(MeshExportGeometry.defaults(nil, format: "stl") == nil)
    }

    @Test func answersNilForTheStoredFormTheTurntablesAndAnUnlistedContainer() {
        #expect(MeshExportGeometry.defaults(capabilities, format: "glb") == nil)
        #expect(MeshExportGeometry.defaults(capabilities, format: "gif") == nil)
        #expect(MeshExportGeometry.defaults(capabilities, format: "usdz") == nil)
    }

    @Test func clampsADefaultSizeIntoTheHostsOwnBounds() {
        let tight = MeshExportGeometryCapabilities(
            sizeMm: MeshSizeControl(min: 10, max: 50, default: 50),
            upAxes: capabilities.upAxes, origins: capabilities.origins,
            defaults: capabilities.defaults)
        #expect(MeshExportGeometry.defaults(tight, format: "stl")?.sizeMm == 50)
    }

    @Test func fallsBackToTheFirstAdvertisedAxisAndOrigin() {
        let narrow = MeshExportGeometryCapabilities(
            sizeMm: capabilities.sizeMm, upAxes: [.y], origins: [.center],
            defaults: capabilities.defaults)
        #expect(MeshExportGeometry.defaults(narrow, format: "stl")
            == MeshExportGeometry(sizeMm: 100, upAxis: .y, origin: .center))
    }

    // MARK: - Dimensions and the sentence

    @Test func scalesTheLongestStoredExtentToTheRequestedSize() throws {
        let dimensions = try #require(
            MeshExportGeometry.dimensionsMm(bounds: bounds, sizeMm: 100, upAxis: .y))
        #expect(abs(dimensions.x - 100) < 1e-4)
        #expect(abs(dimensions.y - 42.86) < 0.05)
        #expect(abs(dimensions.z - 68.57) < 0.05)
    }

    /// A Z-up file rotates (x, y, z) into (x, -z, y), so its own axes read
    /// width, then depth, then height.
    @Test func reordersTheExtentsForAZUpExport() throws {
        let yUp = try #require(
            MeshExportGeometry.dimensionsMm(bounds: bounds, sizeMm: 100, upAxis: .y))
        let zUp = try #require(
            MeshExportGeometry.dimensionsMm(bounds: bounds, sizeMm: 100, upAxis: .z))
        #expect(abs(zUp.x - yUp.x) < 1e-6)
        #expect(abs(zUp.y - yUp.z) < 1e-6)
        #expect(abs(zUp.z - yUp.y) < 1e-6)
    }

    @Test func writesModelUnitsVerbatimWhenNoSizeIsAskedFor() throws {
        let dimensions = try #require(
            MeshExportGeometry.dimensionsMm(bounds: bounds, sizeMm: nil, upAxis: .y))
        #expect(abs(dimensions.x - 1) < 1e-6)
        #expect(abs(dimensions.y - 0.4286) < 1e-3)
    }

    @Test func answersNilRatherThanDividingByZeroOnNoBoxOrADegenerateOne() {
        #expect(MeshExportGeometry.dimensionsMm(bounds: nil, sizeMm: 100, upAxis: .z) == nil)
        let point = MeshBounds(min: SIMD3(1, 1, 1), max: SIMD3(1, 1, 1))
        #expect(MeshExportGeometry.dimensionsMm(bounds: point, sizeMm: 100, upAxis: .z) == nil)
    }

    @Test func namesWhatTheExportedFileWillMeasure() {
        #expect(MeshExportGeometry.sizeLabel(
            bounds: bounds,
            options: MeshExportGeometry(sizeMm: 100, upAxis: .z, origin: .floor))
            == "100.0 × 68.6 × 42.9 mm")
        #expect(MeshExportGeometry.sizeLabel(
            bounds: bounds,
            options: MeshExportGeometry(sizeMm: nil, upAxis: .z, origin: .floor))
            == "as stored (1.00 × 0.69 × 0.43)")
        #expect(MeshExportGeometry.sizeLabel(
            bounds: nil,
            options: MeshExportGeometry(sizeMm: 120, upAxis: .z, origin: .floor))
            == "longest side 120 mm")
        #expect(MeshExportGeometry.sizeLabel(
            bounds: nil,
            options: MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor))
            == "as stored")
    }

    // MARK: - The request body

    @Test func sendsTheThreeKeysTheHostAdvertised() throws {
        let body = try body(.geometry(
            format: "stl",
            MeshExportGeometry(sizeMm: 120, upAxis: .y, origin: .center)))
        #expect(body["format"] as? String == "stl")
        #expect(body["size_mm"] as? Double == 120)
        #expect(body["up_axis"] as? String == "y")
        #expect(body["origin"] as? String == "center")
    }

    /// The wire has no way to ask a size-defaulting format to skip scaling, so
    /// "as stored" is simply the ABSENT key, and it is only ever offered for a
    /// format whose own default is already nil.
    @Test func omitsSizeMmForAnAsStoredExport() throws {
        let body = try body(.geometry(
            format: "obj",
            MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor)))
        #expect(Set(body.keys) == ["format", "up_axis", "origin"])
    }

    /// The old-server shape, key for key: an older host DROPS unknown keys.
    @Test func sendsTheBareFormatWhenTheHostAdvertisedNoGeometryBlock() throws {
        #expect(try Set(body(.geometry(format: "stl", nil)).keys) == ["format"])
    }

    /// The server REFUSES geometry keys on the stored form and on a turntable,
    /// so they can never be built into a request for one.
    @Test func neverSendsGeometryKeysForTheStoredFormOrATurntable() throws {
        let knobs = MeshExportGeometry(sizeMm: 120, upAxis: .y, origin: .center)
        #expect(try Set(body(.geometry(format: "glb", knobs)).keys) == ["format"])
        #expect(try Set(body(.geometry(format: "gif", knobs)).keys) == ["format"])
    }

    @Test func sendsTheTurntableKnobsForAnAnimatedContainer() throws {
        let body = try body(.turntable(
            format: "GIF", MeshTurntableOptions(frames: 72, fps: 24, maxDimension: 1024,
                                                transparent: true)))
        #expect(body["format"] as? String == "gif")
        #expect(body["frames"] as? Int == 72)
        #expect(body["fps"] as? Int == 24)
        #expect(body["max_dimension"] as? Int == 1024)
        #expect(body["transparent"] as? Bool == true)
        #expect(body["size_mm"] == nil)
    }

    /// Turntable keys on a geometry container are refused by the server, so
    /// asking for one builds the bare format instead.
    @Test func neverSendsTurntableKeysForAGeometryContainer() throws {
        #expect(try Set(body(.turntable(format: "stl", MeshTurntableOptions())).keys)
            == ["format"])
    }

    /// A stepper can only ever post a value the server accepts.
    @Test func clampsEveryTurntableValueIntoTheServersBounds() throws {
        let large = try body(.turntable(
            format: "gif",
            MeshTurntableOptions(frames: 900, fps: 120, maxDimension: 8192)))
        #expect(large["frames"] as? Int == MeshTurntableOptions.frameBounds.upperBound)
        #expect(large["fps"] as? Int == MeshTurntableOptions.maximumFPS)
        #expect(large["max_dimension"] as? Int == MeshTurntableOptions.maximumDimension)

        let small = try body(.turntable(
            format: "gif", MeshTurntableOptions(frames: 1, fps: 0, maxDimension: 1)))
        #expect(small["frames"] as? Int == MeshTurntableOptions.frameBounds.lowerBound)
        #expect(small["fps"] as? Int == 1)
    }
}
