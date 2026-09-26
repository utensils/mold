import AppKit
import MoldClient
import SwiftUI

// Getting a print out of Mold: what the machine that holds it will convert
// it into, and where the result lands. Split for size.
@MainActor
extension LibraryActions {
    /// What this CLIP can be converted into on the machine that holds it.
    ///
    /// A mesh is answered by `meshExports` instead, because its containers
    /// come from `capabilities.mesh.export_formats` -- the host's own list --
    /// and a turntable's options are not a transcode's.
    func exportFormats(for entry: LibraryEntry) -> [String] {
        guard entry.print.isVideo else { return [] }
        return hosts.exportOptions[entry.hostID]?.forVideo ?? []
    }

    /// The host's advertised mesh containers, split into one-click transcodes
    /// and the animated turntables that share a sheet. Empty for a host with
    /// no mesh block -- such a host has no mesh family and nothing to convert.
    func meshExports(for entry: LibraryEntry) -> MeshExport.Split {
        guard entry.print.isMesh else { return MeshExport.Split(files: [], animations: []) }
        return hosts.capabilities[entry.hostID]?.meshExports
            ?? MeshExport.Split(files: [], animations: [])
    }

    /// The geometry knobs to OFFER for one container, or nil to post the bare
    /// format. Nil is an older host, or a container it does not scale.
    func meshGeometry(for entry: LibraryEntry, format: String) -> MeshExportGeometry? {
        hosts.capabilities[entry.hostID]?.meshGeometryDefaults(for: format)
    }

    /// The host's own geometry block, or nil on one that predates it.
    func meshGeometryCapabilities(
        for entry: LibraryEntry
    ) -> MeshExportGeometryCapabilities? {
        hosts.capabilities[entry.hostID]?.mesh?.exportGeometry
    }

    /// Converts a print and saves the result.
    func export(_ entry: LibraryEntry, as format: String) {
        export(entry, request: .geometry(format: format, nil))
    }

    /// The door every Export ▸ row goes through.
    ///
    /// A turntable ALWAYS asks -- its frames, rate and size are the point of
    /// the entry's ellipsis. A geometry container asks only where the host
    /// advertised knobs to ask about. Everything else converts straight away,
    /// which is what a clip's containers have always done.
    func requestExport(_ entry: LibraryEntry, as format: String, bounds: MeshBounds? = nil) {
        // A CLIP's only containers are `gif`/`apng`/`webp`, so the animated
        // test ALONE sent every video export through the turntable sheet --
        // which posts `transparent`, and the server refuses that outright for
        // anything but a mesh turntable. The kind is half the question.
        guard entry.print.isMesh else {
            export(entry, as: format)
            return
        }
        let animated = MeshExport.isAnimated(format)
        let geometry = animated ? nil : meshGeometry(for: entry, format: format)
        guard let meshExport, animated || geometry != nil else {
            export(entry, as: format)
            return
        }
        meshExport(MeshExportPrompt(entry: entry, format: format, geometry: geometry,
                                    capabilities: meshGeometryCapabilities(for: entry),
                                    bounds: bounds))
    }

    /// A mesh's animated containers share one entry, so this picks the first
    /// the host advertised and opens the sheet on it.
    func requestTurntable(_ entry: LibraryEntry, bounds: MeshBounds? = nil) {
        guard let format = meshExports(for: entry).animations.first else { return }
        requestExport(entry, as: format, bounds: bounds)
    }

    /// The same, with whatever optional controls the caller resolved.
    func export(_ entry: LibraryEntry, request: MeshExportRequest) {
        Task {
            guard let client = hosts.backend(for: entry.hostID) else { return }
            let data: Data
            do {
                // Bounded like every other buffered body -- a conversion the
                // machine performs is still an answer this app holds whole.
                data = try ResponseCeiling.checked(
                    await client.export(entry.print.filename, request: request),
                    ceiling: ResponseCeiling.media, what: "that export")
                hosts.succeeded(on: entry.hostID)
            } catch {
                hosts.report(error, on: entry.hostID, doing: "export that print")
                return
            }

            let panel = NSSavePanel()
            panel.nameFieldStringValue = MeshExport.filename(entry.print.filename,
                                                             format: request.format)
            guard await panel.begin() == .OK, let url = panel.url else { return }
            do {
                try await Task.detached(priority: .utility) {
                    try data.write(to: url)
                }.value
            } catch {
                // A disk full, a read-only folder: the person chose Export…,
                // waited for the machine to convert, picked a destination --
                // and got no file and no message. `saveAll` already reports
                // this exact failure; nobody fixed the export half.
                hosts.report(error, on: entry.hostID, doing: "save that export")
            }
        }
    }
}
