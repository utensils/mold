import MoldClient
import SwiftUI

struct VideoExportPrompt: Identifiable {
    let entry: LibraryEntry
    let format: String
    let options: ExportOptions?
    var id: String { "\(entry.hostID)|\(entry.print.filename)|\(format)" }
}

/// Conversion stays cancellable until the save panel opens; the prompt owns the displayed print.
struct VideoExportSheet: View {
    let prompt: VideoExportPrompt
    let onExport: (VideoExportRequest) async throws -> Bool
    @Environment(HostStore.self) private var hosts
    @Environment(\.dismiss) private var dismiss
    @State private var options: ExportOptions?
    @State private var gif: GifExportSelection
    @State private var maxDimension = 720
    @State private var fps = 12
    @State private var loading = false
    @State private var converting = false
    @State private var error: String?
    @State private var conversion: Task<Void, Never>?

    init(prompt: VideoExportPrompt, onExport: @escaping (VideoExportRequest) async throws -> Bool) {
        self.prompt = prompt
        self.onExport = onExport
        _options = State(initialValue: prompt.options)
        _gif = State(initialValue: GifExportSelection(options: prompt.options))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export clip as \(prompt.format.uppercased())").font(.headline)
            Text(prompt.entry.print.displayName).foregroundStyle(.secondary)
            Form {
                if prompt.format == "gif" { GifExportControls(selection: $gif, options: options) }
                Picker("Longest side", selection: $maxDimension) {
                    Text("Original size").tag(0)
                    ForEach([240, 480, 720, 1080], id: \.self) { Text("\($0) px").tag($0) }
                }
                Picker("Frame rate", selection: $fps) {
                    Text("Original frame rate").tag(0)
                    ForEach([8, 12, 15, 24, 30], id: \.self) { Text("\($0) fps").tag($0) }
                }
            }.disabled(loading || converting)
            if loading { ProgressView("Reading export options…") }
            if converting { ProgressView("Converting clip…") }
            if let error {
                Text(error).foregroundStyle(.secondary)
                if options == nil { Button("Try Again") { Task { await load() } } }
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { conversion?.cancel(); dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button("Export…") { submit() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(loading || converting || options?.forVideo.contains(prompt.format) != true
                              || !gif.valid(format: prompt.format, options: options))
            }
        }
        .padding(20).frame(width: 400)
        .task { if options == nil { await load() } }
        .onDisappear { conversion?.cancel() }
    }

    private func load() async {
        loading = true
        defer { loading = false }
        do {
            guard let backend = hosts.backend(for: prompt.entry.hostID) else { throw MoldClientError.malformedResponse }
            let result = try await backend.exportOptions()
            try Task.checkCancellation()
            options = result
            gif = GifExportSelection(options: result)
            error = result.forVideo.contains(prompt.format) ? nil : "This machine no longer offers this format."
        } catch is CancellationError { return } catch { self.error = error.localizedDescription }
    }

    private func submit() {
        let request = VideoExportRequest(format: prompt.format, playback: gif.playback, repeatMode: gif.repeatMode,
                                         maxDimension: maxDimension == 0 ? nil : maxDimension,
                                         fps: fps == 0 ? nil : fps,
                                         pauseMs: gif.pause(format: prompt.format, options: options))
        converting = true
        error = nil
        conversion = Task {
            defer { converting = false }
            do {
                let saved = try await onExport(request)
                try Task.checkCancellation()
                if saved { dismiss() }
            } catch is CancellationError { return } catch { self.error = error.localizedDescription }
        }
    }
}
