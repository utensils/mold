import AppKit
import MoldClient
import SwiftUI

/// What you can do with the picture that just came out.
///
/// The print is already in the machine's gallery by the time it appears here,
/// so these act on the stored file rather than on anything held in memory --
/// which is also why "Show in Library" can simply go and find it.
struct ResultBar: View {
    let result: BatchResult
    let host: MoldHost?
    let showInLibrary: () -> Void

    @State private var saving = false

    var body: some View {
        HStack(spacing: 8) {
            Button { Task { await save() } } label: {
                Label("Save a Copy", systemImage: "square.and.arrow.down")
            }
            .disabled(saving || host == nil)

            Button { Task { await copy() } } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .disabled(host == nil)

            Button(action: showInLibrary) {
                Label("Show in Library", systemImage: "photo.on.rectangle.angled")
            }

            if let seed = result.seed {
                Spacer(minLength: 12)
                Text("seed \(String(seed))")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
    }

    private func bytes() async -> Data? {
        guard let host, let filename = result.filename else { return nil }
        var request = URLRequest(url: MediaURL(baseURL: host.baseURL).media(filename))
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        return try? await URLSession.shared.data(for: request).0
    }

    private func save() async {
        saving = true
        defer { saving = false }
        guard let data = await bytes(), let filename = result.filename else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = filename
        guard await panel.begin() == .OK, let url = panel.url else { return }
        try? data.write(to: url)
    }

    private func copy() async {
        guard let data = await bytes(), let image = NSImage(data: data) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([image])
    }
}
