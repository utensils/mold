import MoldClient
import SwiftUI
import UniformTypeIdentifiers

/// All original-file operations retain exact owner authority through publication.
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
    @State private var file: SourceMediaFile?
    @State private var delivery: FileAction?
    enum FileAction: String, Identifiable { case copy, save, share; var id: Self { self } }
    private var authority: SourceMediaAuthority {
        .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceID(of: owner.hostID))
    }
    private var identity: String { SourceMediaActions.identifier(filename: owner.print.filename, member: member) }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            content()
            Text(identity).font(.caption).textSelection(.enabled)
            Text("On \(owner.hostName)").accessibilityIdentifier("source-media-owning-machine").font(.caption).foregroundStyle(.secondaryText)
            if let hash = member.contentSha256 { Text("SHA-256: \(hash)").font(.caption2).textSelection(.enabled) }
            Menu("Source actions") { actionItems }
                .disabled(requested != nil)
            if requested != nil { ProgressView("Loading original…").accessibilityIdentifier("source-media-original-loading") }
            if let failure { Text(failure).accessibilityIdentifier("source-media-original-error").font(.caption).foregroundStyle(.secondaryText) }
        }
        .contextMenu { actionItems; Divider(); extra() }
        .task(id: requested) {
            if let action = requested {
                await perform(action)
                if requested == action && !Task.isCancelled { requested = nil }
            }
        }
        .sheet(item: $delivery, onDismiss: { file = nil }) { action in
            if let file {
                if action == .save { ExportDocumentPicker(urls: [file.url]) { _ in delivery = nil } }
                else { ShareSheet(items: [file.url]) }
            }
        }
        .onChange(of: library.sourceAccessRevision) {
            if !library.allowsSource(authority, detail: !gallery) { clear() }
        }
        .onChange(of: library.sourceMediaCatalog.contentRevision(for: owner.id)) { clear() }
        .onChange(of: authority) { clear() }
        .onDisappear { clear() }
    }
    @ViewBuilder private var actionItems: some View {
        Button("Copy Source ID") { UIPasteboard.general.string = identity }
        Button("Copy CLI Command") {
            if let host = hosts.host(owner.hostID) { UIPasteboard.general.string = SourceMediaActions.command(filename: owner.print.filename, member: member, host: host.baseURL) }
        }
        if let hash = member.contentSha256 { Button("Copy SHA-256") { UIPasteboard.general.string = hash } }
        Divider()
        Button(member.contentKind == "image" ? "Copy Image" : "Copy File") { requested = .copy }
            .disabled(requested != nil)
        Button("Save a Copy…") { requested = .save }
            .disabled(requested != nil)
        Button("Share…") { requested = .share }
            .disabled(requested != nil)
    }
    private func clear() { delivery = nil; file = nil; requested = nil }
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
            if action == .copy {
                let data = try Data(contentsOf: loaded.file.url)
                guard current() else { return }
                if let image = UIImage(data: data) { UIPasteboard.general.image = image }
                else {
                    let type = UTType(filenameExtension: loaded.file.url.pathExtension) ?? .data
                    UIPasteboard.general.setData(data, forPasteboardType: type.identifier)
                }
            } else { file = loaded.file; delivery = action }
        } catch { if current() { failure = error.localizedDescription } }
    }
}
