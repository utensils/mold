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
    let exportOptions: ExportOptions?
    /// The mesh's own box, when a viewer has reported one, so the sentence can
    /// name what the file will measure.
    let bounds: MeshBounds?

    /// Whether "as stored" is a choice AT ALL for this container.
    ///
    /// The wire has no way to ask a size-defaulting format to skip scaling --
    /// an absent `size_mm` is read as the host's OWN default, which is 100 mm
    /// for STL and PLY (`validation.rs:2666`). So it is offered exactly where
    /// that default is already null, which is the reference's own rule
    /// (`ui/components/MeshGeometryFields.vue:64`). Offering it anywhere else
    /// labels a 100 mm file "as stored".
    var offersAsStored: Bool { geometry?.sizeMm == nil }

    init(entry: LibraryEntry, format: String, geometry: MeshExportGeometry?,
         capabilities: MeshExportGeometryCapabilities?, exportOptions: ExportOptions? = nil,
         bounds: MeshBounds? = nil) {
        id = "\(entry.id.host)#\(entry.id.filename)#\(format)"
        self.entry = entry
        self.format = format
        self.geometry = geometry
        self.capabilities = capabilities
        self.exportOptions = exportOptions
        self.bounds = bounds
    }
}

struct MeshExportSheet: View {
    let prompt: MeshExportPrompt
    let onExport: (MeshExportRequest) async throws -> Bool
    @Environment(\.dismiss) private var dismiss

    @State var geometry: MeshExportGeometry
    @State var turntable = MeshTurntableOptions()
    @State var scaled: Bool
    @State var gif: GifExportSelection
    @State private var converting = false
    @State private var error: String?
    @State private var conversion: Task<Void, Never>?

    init(prompt: MeshExportPrompt, onExport: @escaping (MeshExportRequest) async throws -> Bool) {
        self.prompt = prompt
        self.onExport = onExport
        let resolved = prompt.geometry
            ?? MeshExportGeometry(sizeMm: nil, upAxis: .y, origin: .floor)
        _geometry = State(initialValue: resolved)
        // Forced on wherever "as stored" is not a choice, so the toggle can
        // never leave `size_mm` absent on a format whose default is a size.
        _scaled = State(initialValue: resolved.sizeMm != nil)
        _gif = State(initialValue: GifExportSelection(options: prompt.exportOptions))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title).font(.headline)
            Group {
                if prompt.geometry == nil { turntableBody } else { geometryBody }
            }.disabled(converting)
            if converting { ProgressView("Converting mesh…") }
            if let error { Text(error).foregroundStyle(.secondary) }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { conversion?.cancel(); dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Export…") {
                    submit()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(converting || !gif.valid(format: prompt.format, options: prompt.exportOptions))
            }
        }
        .padding(20)
        .frame(width: 380)
        // A bigger frame or a transparent backdrop takes views off the table;
        // the held value follows rather than waiting to be refused.
        .onChange(of: turntable.maxDimension) { _, _ in turntable = turntable.clamped }
        .onChange(of: turntable.transparent) { _, _ in turntable = turntable.clamped }
        .onDisappear { conversion?.cancel() }
    }

    private var title: String {
        prompt.geometry == nil
            ? "Export a turntable as \(prompt.format.uppercased())"
            : "Export as \(prompt.format.uppercased())"
    }

    var request: MeshExportRequest {
        guard prompt.geometry != nil else {
            var options = turntable
            options.playback = gif.playback
            options.repeatMode = gif.repeatMode
            options.pauseMs = gif.pause(format: prompt.format, options: prompt.exportOptions)
            return .turntable(format: prompt.format, options)
        }
        var resolved = geometry
        // "As stored" is the ABSENT key, and it is only ever offered where the
        // host's own default for this container is already unscaled.
        if !scaled, prompt.offersAsStored { resolved.sizeMm = nil }
        return .geometry(format: prompt.format,
                         prompt.capabilities == nil ? nil : resolved)
    }

    private func submit() {
        let sending = request
        converting = true
        error = nil
        conversion = Task {
            defer { converting = false }
            do {
                let saved = try await onExport(sending)
                try Task.checkCancellation()
                if saved { dismiss() }
            } catch is CancellationError { return } catch { self.error = error.sentence }
        }
    }
}
