import Foundation
import Testing

@testable import MoldClient

// `GET /api/gallery/export-options` answers ONE flat list covering both
// kinds, and the two views of it deliberately OVERLAP: a mesh can be exported
// as an animated turntable, so `forMesh` is a union and `forVideo` is
// animated-only. Nothing pinned that, and the two sets are one word apart.

private func options(_ formats: [String]) throws -> ExportOptions {
    let json = try JSONSerialization.data(withJSONObject: ["formats": formats])
    return try MoldJSON.decoder.decode(ExportOptions.self, from: json)
}

/// One mixed list, read both ways.
@Test func theTwoViewsSplitOneListAndOverlapOnTheAnimatedFormats() throws {
    let all = try options(["gif", "apng", "webp", "obj", "stl", "ply", "zip"])
    #expect(all.forVideo == ["gif", "apng", "webp"])
    #expect(all.forMesh == ["gif", "apng", "webp", "obj", "stl", "ply", "zip"])
}

/// A clip is never offered a geometry file, and the order the host listed
/// them in is kept -- the menu reads the way the machine answered.
@Test func aClipIsNeverOfferedGeometry() throws {
    let host = try options(["stl", "webp", "obj", "gif"])
    #expect(host.forVideo == ["webp", "gif"])
    #expect(host.forMesh == ["stl", "webp", "obj", "gif"])
}

/// `glb` is the STORED form, not something a print is converted INTO, so it
/// is in neither view even when the host lists it first
/// (`capabilities.mesh.export_formats` puts it there so a client can see what
/// it holds).
@Test func theStoredFormIsNotAnExport() throws {
    let host = try options(["glb", "obj", "gif"])
    #expect(!host.forMesh.contains("glb"))
    #expect(!host.forVideo.contains("glb"))
}

/// A format this build has never heard of is not offered, because the app has
/// nothing to do with it -- but it does not disturb the ones it knows.
@Test func aFormatThisBuildDoesNotKnowIsSimplyNotOffered() throws {
    let host = try options(["gif", "avif", "obj"])
    #expect(host.forVideo == ["gif"])
    #expect(host.forMesh == ["gif", "obj"])
}

/// An older host that answers nothing offers nothing, rather than a default
/// list this app invented.
@Test func aHostWithNoExportsOffersNone() throws {
    let host = try options([])
    #expect(host.forVideo.isEmpty)
    #expect(host.forMesh.isEmpty)
}
