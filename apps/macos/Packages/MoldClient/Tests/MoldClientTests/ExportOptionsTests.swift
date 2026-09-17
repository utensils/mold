import Foundation
import Testing

@testable import MoldClient

// `GET /api/gallery/export-options` answers ONE flat list covering both
// kinds, and the CLIP's half of it is the animated containers. The mesh view
// this type used to carry is gone: a mesh's containers are
// `capabilities.mesh.export_formats`, the host's own advertised list, so a
// client set could no longer hide a container a host added (review 03-L1).

private func options(_ formats: [String]) throws -> ExportOptions {
    let json = try JSONSerialization.data(withJSONObject: ["formats": formats])
    return try MoldJSON.decoder.decode(ExportOptions.self, from: json)
}

/// One mixed list; a clip takes the animated half of it.
@Test func aClipTakesTheAnimatedHalfOfTheHostsList() throws {
    let all = try options(["gif", "apng", "webp", "obj", "stl", "ply", "zip"])
    #expect(all.forVideo == ["gif", "apng", "webp"])
}

/// A clip is never offered a geometry file, and the order the host listed
/// them in is kept -- the menu reads the way the machine answered.
@Test func aClipIsNeverOfferedGeometry() throws {
    let host = try options(["stl", "webp", "obj", "gif"])
    #expect(host.forVideo == ["webp", "gif"])
}

/// `glb` is the STORED form, not something a print is converted INTO, so a
/// clip is never offered it either.
@Test func theStoredFormIsNotAnExport() throws {
    let host = try options(["glb", "obj", "gif"])
    #expect(!host.forVideo.contains("glb"))
}

/// A format this build has never heard of is not offered, because the app has
/// nothing to do with it -- but it does not disturb the ones it knows.
@Test func aFormatThisBuildDoesNotKnowIsSimplyNotOffered() throws {
    let host = try options(["gif", "avif", "obj"])
    #expect(host.forVideo == ["gif"])
}

/// An older host that answers nothing offers nothing, rather than a default
/// list this app invented.
@Test func aHostWithNoExportsOffersNone() throws {
    let host = try options([])
    #expect(host.forVideo.isEmpty)
}
