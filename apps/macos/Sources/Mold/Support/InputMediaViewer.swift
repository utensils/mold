import AVKit
import MoldClient
import SwiftUI

/// Selected input only: no library acknowledgment or attachment side effects.
struct InputMediaViewer: View {
    let title: String
    let identity: String
    let load: () async throws -> SourceMediaFile
    let isCurrent: () -> Bool
    @Environment(\.dismiss) private var dismiss
    @State private var file: SourceMediaFile?
    @State private var image: NSImage?
    @State private var player: AVPlayer?
    @State private var kind: SourceMediaOriginal.Kind?
    @State private var failure: String?
    @State private var retry = 0
    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                Group {
                    if let image { ZoomableImage(image: image, identity: identity, alphaBed: true) }
                    else if let player {
                        if kind == .audio { SourceAudioPlayback(player: player) }
                        else { VideoPlaybackView(player: player) }
                    } else if let failure {
                        ContentUnavailableView("Input unavailable", systemImage: "exclamationmark.triangle", description: Text(failure))
                        Button("Retry") { retry += 1 }
                    } else { ProgressView("Loading input…") }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }.padding()
            .navigationTitle(title)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() }.keyboardShortcut(.cancelAction) } }
        }
        .frame(minWidth: 480, idealWidth: 760, minHeight: 420, idealHeight: 620)
        .task(id: "\(identity)|\(retry)") {
            clear(); failure = nil
            do {
                guard isCurrent() else { throw CancellationError() }
                let original = try await load()
                let inspected = try await SourceMediaOriginal.inspect(original)
                try Task.checkCancellation()
                guard isCurrent() else { return }
                file = original; kind = inspected.kind
                if let decoded = inspected.image { image = NSImage(cgImage: decoded.image, size: .zero) }
                else { player = AVPlayer(url: inspected.url) }
            } catch is CancellationError {} catch {
                if !Task.isCancelled, isCurrent() { failure = error.sentence }
            }
        }
        .onChange(of: isCurrent()) { _, current in if !current { clear(); dismiss() } }
        .onDisappear(perform: clear)
    }
    private func clear() { player?.pause(); player?.replaceCurrentItem(with: nil); player = nil; image = nil; file = nil }
}

struct InputMediaViewButton: View {
    let title: String
    let identity: String
    let load: () async throws -> SourceMediaFile
    var isCurrent: () -> Bool = { true }
    @State private var showing = false
    @State private var openedIdentity = ""
    var body: some View {
        Button { openedIdentity = identity; showing = true } label: {
            Label("View", systemImage: "arrow.up.left.and.arrow.down.right")
        }
        .environment(\.isEnabled, true)
        .accessibilityLabel("View \(title)")
        .help("Inspect this input without changing it")
        .onChange(of: identity) { _, _ in showing = false }
        .onChange(of: isCurrent()) { _, current in if !current { showing = false } }
        .sheet(isPresented: $showing) {
            InputMediaViewer(title: title, identity: identity, load: load, isCurrent: { openedIdentity == identity && isCurrent() })
        }
    }
}
