import AppKit
import MoldClient
import SwiftUI

/// What the Generate pane shows while and after a render.
struct RunCanvas: View {
    let state: RunState
    let host: MoldHost?
    let showInLibrary: () -> Void

    @State private var preview: NSImage?
    @State private var result: NSImage?

    var body: some View {
        ZStack {
            switch state {
            case .idle, .submitting:
                idle
            case .running:
                running
            case let .finished(finished, _):
                finishedView(finished)
            case let .failed(message):
                ContentUnavailableView("That didn't finish", systemImage: "exclamationmark.triangle",
                                       description: Text(message))
            }
        }
        // Decoded on change, never in `body`: `body` re-runs on every progress
        // tick and decoding a PNG there would burn a decode per tick.
        .onChange(of: state.previewData, initial: true) { _, data in
            preview = data.flatMap(NSImage.init(data:))
        }
        .task(id: resultFilename) { await loadResult() }
    }

    @ViewBuilder private var idle: some View {
        ContentUnavailableView {
            Label("Nothing rendered yet", systemImage: "wand.and.sparkles")
        } description: {
            Text(state.isBusy ? "Sending…" : "Your picture appears here.")
        }
    }

    @ViewBuilder private var running: some View {
        VStack(spacing: 14) {
            if let preview {
                Image(nsImage: preview)
                    .resizable()
                    .interpolation(.medium)
                    .aspectRatio(contentMode: .fit)
                    .padding(40)
            } else {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(state.stage ?? "Getting ready…")
                        .foregroundStyle(.secondary)
                    Text("A preview appears after the first step.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    @ViewBuilder private func finishedView(_ finished: BatchResult) -> some View {
        if let result {
            VStack(spacing: 12) {
                Image(nsImage: result)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                ResultBar(result: finished, host: host, showInLibrary: showInLibrary)
            }
            .padding(24)
        } else {
            ProgressView("Fetching your picture…")
        }
    }

    private var resultFilename: String? {
        guard case let .finished(finished, _) = state else { return nil }
        return finished.filename
    }

    private func loadResult() async {
        guard let filename = resultFilename, let host else { result = nil; return }
        var request = URLRequest(url: MediaURL(baseURL: host.baseURL).media(filename))
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return }
        result = NSImage(data: data)
    }
}
