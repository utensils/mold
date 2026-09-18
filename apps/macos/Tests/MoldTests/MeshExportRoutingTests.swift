import Foundation
import MoldClient
import Testing

@testable import Mold

/// Which door an Export ▸ row takes: straight to the machine, or through the
/// sheet that carries the controls the host advertised.
@MainActor
struct MeshExportRoutingTests {
    private func machine() -> MoldHost {
        MoldHost(name: "workstation", baseURL: URL(string: "http://workstation")!)
    }

    /// `isMesh` reads the print's own `format`, not its extension, so these
    /// fixtures carry the one the host would report.
    private func entry(_ filename: String, format: String,
                       host: MoldHost) -> LibraryEntry {
        let json = """
        {"filename": "\(filename)", "format": "\(format)", "metadata": {}, "timestamp": 1000}
        """
        let print = try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
        return LibraryEntry(host: host, print: print)
    }

    /// A host that converts meshes AND clips, with geometry knobs.
    private func capabilities() throws -> Capabilities {
        let json = """
        {"mesh": {"generation": true, "formats": ["glb"],
                  "export_formats": ["glb", "obj", "stl", "gif"],
                  "export_geometry": {"size_mm": {"min": 1, "max": 1000, "default": 100},
                                      "up_axes": ["y", "z"], "origins": ["center", "floor"],
                                      "defaults": {"obj": {"size_mm": null, "up_axis": "y",
                                                           "origin": "floor"},
                                                   "stl": {"size_mm": 100, "up_axis": "z",
                                                           "origin": "floor"}}}}}
        """
        return try MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    private func actions(_ host: MoldHost, prompt: @escaping (MeshExportPrompt) -> Void)
        throws -> LibraryActions {
        let hosts = HostStore(hosts: [host]) { _ in FakeBackend(host: host) }
        hosts.capabilities[host.id] = try capabilities()
        var actions = LibraryActions(hosts: hosts, library: LibraryStore(hosts: hosts))
        actions.meshExport = prompt
        return actions
    }

    /// **Fails today**: `requestExport` branches on `MeshExport.isAnimated`
    /// alone, and a CLIP's only containers are `gif`/`apng`/`webp` -- so every
    /// video export opened a turntable sheet titled "views around the mesh"
    /// and posted `transparent`, which the server refuses outright
    /// (`routes.rs:9965`). Right-click an MP4 ▸ Export ▸ GIF always 422'd.
    /// Fallout of deleting `ExportOptions.forMesh`.
    @Test func aClipExportNeverTakesTheMeshDoor() throws {
        let host = machine()
        var prompted: [MeshExportPrompt] = []
        let actions = try actions(host) { prompted.append($0) }
        for format in ["gif", "apng", "webp"] {
            actions.requestExport(entry("clip.mp4", format: "mp4", host: host), as: format)
        }
        #expect(prompted.isEmpty, "a clip has no turntable and no geometry")
    }

    /// A MESH's animated container is the one that opens the sheet, because a
    /// turntable is a RENDER with frames, a rate and a size of its own.
    @Test func aMeshTurntableAsksForItsControls() throws {
        let host = machine()
        var prompted: [MeshExportPrompt] = []
        let actions = try actions(host) { prompted.append($0) }
        actions.requestExport(entry("chair.glb", format: "glb", host: host), as: "gif")
        #expect(prompted.count == 1)
        #expect(prompted.first?.format == "gif")
        // A turntable carries no geometry knobs; the server refuses them.
        #expect(prompted.first?.geometry == nil)
    }

    /// A geometry container asks only where the host advertised knobs.
    @Test func aGeometryContainerAsksOnlyWhereTheHostAdvertisedKnobs() throws {
        let host = machine()
        var prompted: [MeshExportPrompt] = []
        let actions = try actions(host) { prompted.append($0) }
        actions.requestExport(entry("chair.glb", format: "glb", host: host), as: "stl")
        #expect(prompted.count == 1)
        #expect(prompted.first?.geometry?.sizeMm == 100)
        // `ply` is advertised as a container but carries no defaults entry, so
        // there is nothing to ask about and it converts straight away.
        actions.requestExport(entry("chair.glb", format: "glb", host: host), as: "ply")
        #expect(prompted.count == 1)
    }

    /// **Fails today**: the sheet offered "Resize for printing" for EVERY
    /// geometry container, and unticking it OMITS `size_mm` -- which the
    /// server reads as its OWN default, 100 mm for STL and PLY
    /// (`validation.rs:2666`). So "as stored" silently wrote a 100 mm model.
    /// The reference gates it on the format's own default being null
    /// (`ui/components/MeshGeometryFields.vue:64`).
    @Test func asStoredIsOfferedOnlyWhereTheHostsDefaultIsAlreadyUnscaled() throws {
        let host = machine()
        var prompted: [MeshExportPrompt] = []
        let actions = try actions(host) { prompted.append($0) }
        actions.requestExport(entry("chair.glb", format: "glb", host: host), as: "obj")
        actions.requestExport(entry("chair.glb", format: "glb", host: host), as: "stl")
        #expect(prompted.count == 2)
        // OBJ's own default IS unscaled, so "as stored" is a real choice.
        #expect(prompted[0].offersAsStored)
        // STL's is 100 mm. There is no way on the wire to ask for unscaled,
        // so the sheet must not pretend there is.
        #expect(!prompted[1].offersAsStored)
    }
}
