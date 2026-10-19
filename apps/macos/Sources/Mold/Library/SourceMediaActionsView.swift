import AppKit
import MoldClient
import SwiftUI

/// Metadata copies are immediate; file actions load the exact bounded original.
struct SourceMediaActionsView<Content: View, Extra: View>: View {
    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    let gallery: Bool
    @ViewBuilder var content: () -> Content
    @ViewBuilder var extra: () -> Extra
    @State private var requested: FileAction?
    @State private var failure: String?
    @State private var sharing = SourceMediaShare()
    enum FileAction: String { case copy, save, share }
    private var authority: SourceMediaAuthority {
        .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceIDs[owner.hostID])
    }
    private var identity: String { SourceMediaActions.identifier(filename: owner.print.filename, member: member) }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            content()
            if !gallery { Text(identity).font(.caption).textSelection(.enabled) }
            Text("On \(owner.hostName)").accessibilityIdentifier("source-media-owning-machine").font(.caption).foregroundStyle(.secondary)
            if !gallery, let hash = member.contentSha256 { Text("SHA-256: \(hash)").font(.caption2).textSelection(.enabled) }
            Menu("Source actions") { actionItems }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .disabled(requested != nil)
            if requested != nil { ProgressView("Loading original…").accessibilityIdentifier("source-media-original-loading") }
            if let failure { Text(failure).accessibilityIdentifier("source-media-original-error").font(.caption).foregroundStyle(.secondary) }
        }
        .contextMenu { actionItems; Divider(); extra() }
        .task(id: requested) {
            if let action = requested {
                await perform(action)
                if requested == action && !Task.isCancelled { requested = nil }
            }
        }
        .onChange(of: library.sourcePrivacyRevision) {
            if !library.allowsSource(authority, detail: !gallery) { sharing.clear(); requested = nil }
        }
        .onChange(of: library.sourceMediaCatalog.contentRevision(for: owner.id)) { sharing.clear(); requested = nil }
        .onChange(of: authority) { sharing.clear(); requested = nil }
        .onDisappear { sharing.clear(); requested = nil }
    }
    @ViewBuilder private var actionItems: some View {
        Button("Copy Source ID") { Clipboard.put(identity) }
        Button("Copy CLI Command") {
            if let host = hosts.host(owner.hostID) { Clipboard.put(SourceMediaActions.command(filename: owner.print.filename, member: member, host: host.baseURL)) }
        }
        if let hash = member.contentSha256 { Button("Copy SHA-256") { Clipboard.put(hash) } }
        Divider()
        Button(member.contentKind == "image" ? "Copy Image" : "Copy File") { requested = .copy }
            .disabled(requested != nil)
        Button("Save a Copy…") { requested = .save }
            .disabled(requested != nil)
        Button("Share…") { requested = .share }
            .disabled(requested != nil)
    }
    private func perform(_ action: FileAction) async {
        failure = nil
        let captured = authority
        guard let host = hosts.host(owner.hostID) else { return }
        let row = SourceMediaCatalog.Row(owner: owner, member: member, route: host, instance: captured.instance)
        let revision = library.sourceMediaCatalog.contentRevision(for: owner.id)
        func current() -> Bool {
            !Task.isCancelled && authority == captured && library.allowsSource(captured, detail: !gallery)
                && library.sourceMediaCatalog.contentRevision(for: owner.id) == revision
        }
        do {
            guard let loaded = try await SourceMediaLoading.original(row: row, backend: hosts.backend(for: host), isCurrent: current), current() else { return }
            switch action {
            case .copy:
                let data = try Data(contentsOf: loaded.file.url)
                guard current() else { return }
                NSPasteboard.general.clearContents()
                if let image = NSImage(data: data) { NSPasteboard.general.writeObjects([image]) }
                else {
                    let sourceHosts = hosts, sourceLibrary = library
                    SourceMediaClipboard.shared.copy(loaded.file, isCurrent: {
                        sourceHosts.host(owner.hostID) == host && sourceHosts.instanceIDs[owner.hostID] == captured.instance
                            && sourceLibrary.allowsSource(captured, detail: !gallery)
                            && sourceLibrary.sourceMediaCatalog.contentRevision(for: owner.id) == revision
                    })
                }
            case .save:
                let panel = NSSavePanel(); panel.nameFieldStringValue = member.displayName
                guard await panel.begin() == .OK, let url = panel.url, current() else { return }
                try Data(contentsOf: loaded.file.url).write(to: url, options: .atomic)
            case .share:
                guard let window = NSApp.keyWindow, let anchor = window.contentView, current() else { return }
                let sourceHosts = hosts, sourceLibrary = library
                sharing.present(loaded.file, relativeTo: anchor, isCurrent: {
                    sourceHosts.host(owner.hostID) == host && sourceHosts.instanceIDs[owner.hostID] == captured.instance
                        && sourceLibrary.allowsSource(captured, detail: !gallery)
                        && sourceLibrary.sourceMediaCatalog.contentRevision(for: owner.id) == revision
                })
            }
        } catch { if current() { failure = error.localizedDescription } }
    }
}
