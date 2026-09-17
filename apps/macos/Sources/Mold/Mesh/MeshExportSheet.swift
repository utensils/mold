import MoldClient
import SwiftUI

/// One mesh export, waiting for the controls the HOST said it accepts.
///
/// A geometry container carries the size, the up axis and the origin; an
/// animated one carries the turntable's frames, rate and size. Never both --
/// the server refuses the wrong group rather than ignoring it.
struct MeshExportPrompt: Identifiable {
    let id: String
    let entry: LibraryEntry
    let format: String
    /// The host's defaults for this container, or nil for a turntable.
    let geometry: MeshExportGeometry?
    /// The host's bounds and the axes it accepts. Nil on an older host, which
    /// is the one gate: its exports post the bare format.
    let capabilities: MeshExportGeometryCapabilities?
    /// The mesh's own box, when a viewer has reported one, so the sentence can
    /// name what the file will measure.
    let bounds: MeshBounds?

    init(entry: LibraryEntry, format: String, geometry: MeshExportGeometry?,
         capabilities: MeshExportGeometryCapabilities?, bounds: MeshBounds? = nil) {
        id = "\(entry.id.host)#\(entry.id.filename)#\(format)"
        self.entry = entry
        self.format = format
        self.geometry = geometry
        self.capabilities = capabilities
        self.bounds = bounds
    }
}

struct MeshExportSheet: View {
    let prompt: MeshExportPrompt
    let onExport: (MeshExportRequest) -> Void
    @Environment(\.dismiss) private var dismiss

    @State var geometry: MeshExportGeometry
    @State var turntable = MeshTurntableOptions()
    @State var scaled: Bool

    init(prompt: MeshExportPrompt, onExport: @escaping (MeshExportRequest) -> Void) {
        self.prompt = prompt
        self.onExport = onExport
        let resolved = prompt.geometry
            ?? MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor)
        _geometry = State(initialValue: resolved)
        _scaled = State(initialValue: resolved.sizeMm != nil)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.headline)
            if prompt.geometry == nil {
                turntableBody
            } else {
                geometryBody
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Export…") {
                    onExport(request)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private var title: String {
        prompt.geometry == nil
            ? "Export a turntable as \(prompt.format.uppercased())"
            : "Export as \(prompt.format.uppercased())"
    }

    var request: MeshExportRequest {
        guard prompt.geometry != nil else {
            return .turntable(format: prompt.format, turntable)
        }
        var resolved = geometry
        // "As stored" is the ABSENT key, and it is only ever offered where the
        // host's own default for this container is already unscaled.
        if !scaled { resolved.sizeMm = nil }
        return .geometry(format: prompt.format,
                         prompt.capabilities == nil ? nil : resolved)
    }
}
