import AppKit
import MoldClient
import SwiftUI

/// The same cached, authority-bound preview in the contact sheet and inspector.
struct SourceMediaThumbnail: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    var gallery = true
    var onAspect: (SourcePreviewKey, Double) -> Void = { _, _ in }
    @State private var preview: NSImage?
    @State private var displayed: SourcePreviewKey?
    private var key: SourcePreviewKey {
        .init(authority: .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceIDs[owner.hostID]), member: member)
    }
    var body: some View {
        Rectangle().fill(.quaternary)
            .overlay {
                if let image = visiblePreview {
                    Image(nsImage: image).resizable().scaledToFit()
                        .accessibilityLabel(member.displayName)
                        .accessibilityIdentifier("source-media-preview-" + member.memberId)
                } else {
                    Image(systemName: member.sourceSymbol).font(.title).foregroundStyle(.secondary)
                }
            }
            .clipped()
            .task(id: SourceThumbnailTask(key: key, revision: library.sourceMediaCatalog.contentRevision(for: owner.id))) {
                await load()
            }
        .onChange(of: library.sourceAccessRevision) {
            if !library.allowsSource(key.authority, detail: !gallery) { preview = nil; displayed = nil }
        }
    }
    private var visiblePreview: NSImage? {
        guard library.allowsSource(key.authority, detail: !gallery) else { return nil }
        if displayed == key, let preview { return preview }
        return SourceMediaPreview.peek(key: key)
    }
    private func load() async {
        let captured = key
        let revision = library.sourceMediaCatalog.contentRevision(for: owner.id)
        if displayed != captured || !library.allowsSource(captured.authority, detail: !gallery) {
            preview = nil; displayed = nil
        }
        guard member.contentKind == "image", library.allowsSource(captured.authority, detail: !gallery),
              let host = hosts.host(owner.hostID) else { return }
        let image = await SourcePreviewRetry.load(isCurrent: {
            key == captured && revision == library.sourceMediaCatalog.contentRevision(for: owner.id)
                && library.allowsSource(captured.authority, detail: !gallery)
        }) {
            await SourceMediaPreview.image(key: captured, backend: hosts.backend(for: host))
        }
        guard !Task.isCancelled, key == captured, revision == library.sourceMediaCatalog.contentRevision(for: owner.id), library.allowsSource(captured.authority, detail: !gallery) else { return }
        if let image {
            preview = image; displayed = captured
            if image.size.height > 0 { onAspect(captured, Double(image.size.width / image.size.height)) }
        }
    }
}

/// Sources are Library thumbnails; exact identities, roles and reuse actions
/// belong in Details instead of taking several text rows beneath every tile.
struct SourceMediaTile: View {
    @Environment(HostStore.self) private var hosts
    let source: SourceMediaDisplay
    let openOwner: (PrintID) -> Void
    var onAspect: (SourcePreviewKey, Double) -> Void = { _, _ in }
    @State private var showingDetails = false
    @State private var preferredID: String?
    private var current: SourceMediaCatalog.Row {
        source.usages.first { $0.id == preferredID && hosts.host($0.owner.hostID).map(hosts.isUp) == true }
            ?? source.preferredUsage { hosts.host($0).map(hosts.isUp) ?? false }!
    }
    var body: some View {
        Button { showingDetails = true } label: {
            SourceMediaThumbnail(owner: current.owner, member: current.member, onAspect: onAspect)
                .overlay(alignment: .bottomTrailing) {
                    Label("\(source.usages.count) uses", systemImage: current.member.sourceSymbol)
                        .font(.caption2).labelStyle(.iconOnly)
                        .foregroundStyle(.white).padding(5)
                        .background(.black.opacity(0.65), in: Capsule()).padding(5)
                        .accessibilityIdentifier("source-media-badge-" + current.member.memberId)
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(current.member.sourceLabel), \(source.usages.count) uses, \(current.owner.hostName)")
        .accessibilityIdentifier("source-media-tile")
        .contextMenu { Button("Details and Reuse", systemImage: "info.circle") { showingDetails = true } }
        .onChange(of: current.id, initial: true) { _, id in preferredID = id }
        .sheet(isPresented: $showingDetails) {
            SourceMediaDetails(source: source, initial: current.id, gallery: true, openOwner: openOwner)
        }
    }
}

private struct SourceThumbnailTask: Hashable {
    let key: SourcePreviewKey
    let revision: Int
}
