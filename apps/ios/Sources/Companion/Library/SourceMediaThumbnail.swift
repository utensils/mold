import UIKit
import MoldClient
import SwiftUI

/// The same cached, authority-bound preview in the contact sheet and inspector.
struct SourceMediaThumbnail: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    var gallery = true
    @State private var preview: UIImage?
    @State private var displayed: SourcePreviewKey?
    private var key: SourcePreviewKey {
        .init(authority: .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceID(of: owner.hostID)), member: member)
    }
    var body: some View {
        ZStack {
            Rectangle().fill(.quaternary)
            if let preview, displayed == key {
                Image(uiImage: preview).resizable().scaledToFit()
            } else {
                Image(systemName: member.sourceSymbol).font(.title).foregroundStyle(.secondaryText)
            }
        }
        .clipped()
        .task(id: key) {
            await load()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
                await load()
            }
        }
        .onChange(of: library.sourceAccessRevision) {
            if !library.allowsSource(key.authority, detail: !gallery) { preview = nil; displayed = nil }
        }
    }
    private func load() async {
        let captured = key
        if displayed != captured || !library.allowsSource(captured.authority, detail: !gallery) {
            preview = nil; displayed = nil
        }
        guard member.contentKind == "image", library.allowsSource(captured.authority, detail: !gallery),
              let host = hosts.host(owner.hostID) else { return }
        let image = await SourceMediaPreview.image(key: captured, backend: hosts.backend(for: host))
        guard !Task.isCancelled, key == captured, library.allowsSource(captured.authority, detail: !gallery) else { return }
        if let image { preview = image; displayed = captured }
    }
}

/// Sources are Library thumbnails; exact identities, roles and reuse actions
/// belong in Details instead of taking several text rows beneath every tile.
struct SourceMediaTile: View {
    @Environment(HostStore.self) private var hosts
    let source: SourceMediaDisplay
    let openOwner: (PrintID) -> Void
    @State private var showingDetails = false
    @State private var preferredID: String?
    private var current: SourceMediaCatalog.Row {
        source.usages.first { $0.id == preferredID && hosts.host($0.owner.hostID).map(hosts.isUp) == true }
            ?? source.preferredUsage { hosts.host($0).map(hosts.isUp) ?? false }!
    }
    var body: some View {
        Button { showingDetails = true } label: {
            SourceMediaThumbnail(owner: current.owner, member: current.member)
                .aspectRatio(1, contentMode: .fit)
                .overlay(alignment: .bottomTrailing) {
                    Label("\(source.usages.count) uses", systemImage: current.member.sourceSymbol)
                        .font(.caption2).labelStyle(.iconOnly)
                        .foregroundStyle(.white).padding(5)
                        .background(.black.opacity(0.65), in: Capsule()).padding(5)
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(current.member.sourceLabel), \(source.usages.count) uses, \(current.owner.hostName)")
        .accessibilityIdentifier("source-media-tile")
        .contextMenu { Button("Details and Reuse", systemImage: "info.circle") { showingDetails = true } }
        .onChange(of: current.id, initial: true) { _, id in preferredID = id }
        .sheet(isPresented: $showingDetails) {
            SourceMediaDetails(source: source, initial: current.id, gallery: true, openOwner: openOwner)
        }
    }
}
