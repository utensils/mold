import AVKit
import MoldClient
import SwiftUI

/// Opens one exact original; all member usages remain separate, even in one print.
struct SourceMediaDetails: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @Environment(\.dismiss) private var dismiss
    let source: SourceMediaDisplay
    let initial: String
    let gallery: Bool
    let openOwner: (PrintID) -> Void
    @State private var selected: String?
    @State private var file: SourceMediaFile?
    @State private var image: NSImage?
    @State private var player: AVPlayer?
    @State private var error: String?
    @State private var pixels: String?
    @State private var inspectedKind: SourceMediaOriginal.Kind?
    @State private var inspectedKey: SourceMediaOriginalKey?
    @State private var retry = 0
    @State private var limit = 48
    private var usages: [SourceMediaCatalog.Row] {
        source.visibleUsages(access: library.sourceAccess, catalog: library.sourceMediaCatalog, gallery: gallery)
    }
    private var current: SourceMediaCatalog.Row? { usages.first { $0.id == (selected ?? initial) } }

    var body: some View {
        let usages = usages
        let current = usages.first { $0.id == (selected ?? initial) }
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let row = current {
                        Group {
                            if let image { ZoomableImage(image: image, identity: row.id) }
                            else if let player {
                                if (inspectedKind?.rawValue ?? row.member.contentKind) == "audio" { SourceAudioPlayback(player: player) }
                                else { VideoPlaybackView(player: player) }
                            }
                            else if let error {
                                ContentUnavailableView("Original unavailable", systemImage: "exclamationmark.triangle",
                                    description: Text(error))
                            } else { ProgressView("Loading original…") }
                        }.frame(maxWidth: .infinity).frame(height: (inspectedKind?.rawValue ?? row.member.contentKind) == "audio" ? nil : 320)
                        if error != nil { Button("Retry original") { retry += 1 } }
                        Text(row.member.displayName).font(.headline).textSelection(.enabled)
                        Text("\((inspectedKind?.rawValue ?? row.member.contentKind).capitalized) · \(ByteCountFormatter.string(fromByteCount: Int64(row.member.sizeBytes), countStyle: .file))")
                            .font(.subheadline).foregroundStyle(.secondary)
                        if let pixels { Text(pixels).font(.caption).foregroundStyle(.secondary) }
                        Text("Viewing copy on \(row.owner.hostName)").font(.caption)
                        SourceMediaCard(owner: row.owner, member: inspectedKey == loadKey ? (inspectedKind.map { row.member.inspected(as: $0) } ?? row.member) : row.member, gallery: gallery, showsPreview: false)
                    } else {
                        ContentUnavailableView("Source no longer available", systemImage: "lock",
                            description: Text("The selected print or machine changed. Close this view and choose an available source."))
                    }
                    Divider()
                    Text("\(usages.count) \(usages.count == 1 ? "use" : "uses") in this view").font(.headline)
                    ForEach(usages.prefix(limit)) { row in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(row.member.sourceLabel).font(.headline)
                            if let frame = row.member.frameIndex {
                                Text("Frame \(frame)").font(.caption).foregroundStyle(.secondary)
                            }
                            Text("\(row.owner.print.displayName) · \(row.owner.hostName)")
                                .font(.subheadline).textSelection(.enabled)
                            HStack {
                                Button(row.id == current?.id ? "Selected copy" : "View this copy") { selected = row.id }
                                    .disabled(row.id == current?.id)
                                Spacer()
                                Button("Open print") { dismiss(); openOwner(row.owner.id) }
                            }.buttonStyle(.borderless)
                        }.padding(.vertical, 6)
                    }
                    if usages.count > limit { Button("Show more uses") { limit += 48 } }
                    Text("Sources are kept while library prints use them. Open an owning print to manage it. Deleting the last owning print releases the source; sources cannot be deleted independently.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding()
            }
            .navigationTitle("Source details")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
        .frame(minWidth: 640, idealWidth: 780, minHeight: 650)
        .task(id: loadKey) { await load() }
        .onDisappear(perform: clear)
        .onExitCommand { dismiss() }
    }
    private var loadKey: SourceMediaOriginalKey {
        .init(row: current, revision: current.map { library.sourceMediaCatalog.contentRevision(for: $0.owner.id) } ?? 0, retry: retry)
    }
    private func clear() { player?.pause(); player?.replaceCurrentItem(with: nil); player = nil; image = nil; file = nil; pixels = nil; inspectedKind = nil; inspectedKey = nil }
    private func load() async {
        clear(); error = nil
        guard let row = current, let route = hosts.host(row.owner.hostID) else {
            error = "This source is no longer available on its machine."; return
        }
        let instance = hosts.instanceIDs[row.owner.hostID]
        let key = loadKey
        do {
            guard let loaded = try await SourceMediaLoading.original(row: row, backend: hosts.backend(for: route), isCurrent: {
                key == loadKey && current?.authority == row.authority
                    && hosts.host(row.owner.hostID) == route && hosts.instanceIDs[row.owner.hostID] == instance
                    && library.allowsSource(row.authority, detail: !gallery)
            }) else { return }
            let original = loaded.file, inspected = loaded.inspection
            inspectedKind = inspected.kind; inspectedKey = key
            if let decoded = inspected.image {
                image = NSImage(cgImage: decoded.image, size: .zero)
                pixels = decoded.dimensions
            } else { file = original; player = AVPlayer(url: inspected.url) }
        } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
    }
}
