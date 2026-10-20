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
            if requested != nil { ProgressView("Loading original…").accessibilityIdentifier("source-media-original-loading") }
            if let failure { Text(failure).accessibilityIdentifier("source-media-original-error").font(.caption).foregroundStyle(.secondary) }
        }
        .rowActionMenu(menuActions, perform: performMenuAction, extra: extra)
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
    private var menuActions: [RowAction<SourceMediaCommand>] {
        SourceMediaCommand.actions(image: member.contentKind == "image", hasHash: member.contentSha256 != nil, busy: requested != nil)
    }
    private var actionItems: some View { RowActionMenu(actions: menuActions, perform: performMenuAction) }
    private func performMenuAction(_ action: SourceMediaCommand) {
        switch action {
        case .sourceID: Clipboard.put(identity)
        case .command:
            if let host = hosts.host(owner.hostID) { Clipboard.put(SourceMediaActions.command(filename: owner.print.filename, member: member, host: host.baseURL)) }
        case .hash:
            if let hash = member.contentSha256 { Clipboard.put(hash) }
        case .copy: requested = .copy
        case .save: requested = .save
        case .share: requested = .share
        }
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
                let data = try await MediaImport.bytes(of: loaded.file.url)
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
                try await MediaImport.copy(from: loaded.file.url, to: url)
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

/// One action plan drives both the inline menu and the contextual menu.
enum SourceMediaCommand: Hashable {
    case sourceID, command, hash, copy, save, share
    static func actions(image: Bool, hasHash: Bool, busy: Bool) -> [RowAction<Self>] {
        var rows: [RowAction<Self>] = [.init(kind: .sourceID, title: "Copy Source ID"), .init(kind: .command, title: "Copy CLI Command")]
        if hasHash { rows.append(.init(kind: .hash, title: "Copy SHA-256")) }
        rows += [.separator, .init(kind: .copy, title: image ? "Copy Image" : "Copy File", isDisabled: busy),
                 .init(kind: .save, title: "Save a Copy…", isDisabled: busy), .init(kind: .share, title: "Share…", isDisabled: busy)]
        return rows
    }
}
