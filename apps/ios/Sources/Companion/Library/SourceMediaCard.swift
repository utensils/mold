import MoldClient
import SwiftUI

/// One exact publication-owned member; preview and original bytes stay separate.
struct SourceMediaCard: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(GenerateController.self) private var generate
    @Environment(AppRouter.self) private var router
    @Environment(\.dismiss) private var dismiss
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    var gallery = false
    var showsPreview = true
    var openSource: (() -> Void)?
    @State private var preview: UIImage?
    @State private var displayedKey: SourcePreviewKey?
    @State private var choice: SourceMediaAttachment?
    @State private var failure: String?
    private var previewKey: SourcePreviewKey {
        .init(authority: .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceID(of: owner.hostID)), member: member)
    }
    private var recipe: GenerationRecipe? { generate.recipe }
    private var target: UUID? { generate.target?.id }
    private var choices: [SourceMediaAttachment] {
        SourceMediaAttachment.choices(member: member, recipe: recipe, media: generate.draft.media)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            let layout = gallery ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10)) : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            if showsPreview { layout {
                Group {
                    if let preview, displayedKey == previewKey { Image(uiImage: preview).resizable().scaledToFit() }
                    else { Image(systemName: member.sourceSymbol).font(.largeTitle).foregroundStyle(.secondaryText) }
                }
                .frame(maxWidth: gallery ? .infinity : 72)
                .frame(height: gallery ? 148 : 72)
                .background(.quaternary.opacity(gallery ? 0.4 : 0), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityHidden(openSource == nil)
                .accessibilityLabel("Open source")
                .accessibilityAddTraits(openSource == nil ? [] : .isButton)
                .onTapGesture { openSource?() }
                .accessibilityAction { openSource?() }
                VStack(alignment: .leading, spacing: 4) {
                    Text(member.sourceLabel).font(.headline)
                    Text(member.displayName).font(.caption).foregroundStyle(.secondaryText).textSelection(.enabled)
                    Text(ByteCountFormatter.string(fromByteCount: Int64(member.sizeBytes), countStyle: .file))
                        .font(.caption).foregroundStyle(.secondaryText)
                }
            }
            }
            if let openSource {
                Button("Open source", systemImage: "arrow.up.left.and.arrow.down.right", action: openSource)
                    .buttonStyle(.borderless)
            }
            if member.role.hasPrefix("matting_processed") {
                Text("Processed image. Reuse adds it as a new input.").font(.caption).foregroundStyle(.secondaryText)
            }
            if !choices.isEmpty {
                Menu {
                    ForEach(choices) { destination in
                        Button(destination.title) { failure = nil; choice = destination }
                    }
                } label: { Label("Use source", systemImage: "arrow.up.forward") }
                .disabled(choice != nil)
                .accessibilityIdentifier("source-media-reuse")
            } else {
                Text(member.contentKind == "unknown" ? "This source type cannot be attached in this app." : "Choose a compatible model in Generate to use this source.")
                    .font(.caption).foregroundStyle(.secondaryText)
            }
            if choice != nil { ProgressView("Preparing source…") }
            if let failure { Text(failure).font(.caption).foregroundStyle(.secondaryText) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(gallery ? 12 : 0)
        .background(.quaternary.opacity(gallery ? 0.25 : 0), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .contain)
        .onDisappear { preview = nil; displayedKey = nil }
        .task(id: previewKey) {
            guard showsPreview else { return }
            await loadPreview()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(60)) } catch { return }
                await loadPreview()
            }
        }
        .task(id: choice) {
            guard let choice else { return }
            await attach(choice)
            self.choice = nil
        }
    }

    private func loadPreview() async {
        preview = nil; displayedKey = nil
        guard library.allowsSource(previewKey.authority, detail: !gallery), member.contentKind == "image", let host = hosts.host(owner.hostID) else { return }
        let instance = hosts.instanceID(of: owner.hostID)
        let image = await SourceMediaPreview.image(key: previewKey, backend: hosts.backend(for: host))
        guard !Task.isCancelled, library.allowsSource(previewKey.authority, detail: !gallery), hosts.host(owner.hostID) == host, hosts.instanceID(of: owner.hostID) == instance else { return }
        preview = image; displayedKey = previewKey
    }

    private func attach(_ destination: SourceMediaAttachment) async {
        guard library.allowsSource(previewKey.authority, detail: !gallery), let host = hosts.host(owner.hostID), let recipe else { return }
        let instance = hosts.instanceID(of: owner.hostID)
        let fence = ReferenceImportFence(model: generate.modelName, host: target, recipe: recipe, media: generate.draft.media)
        let destinationRoute = target.flatMap(hosts.host)
        let destinationInstance = target.flatMap { hosts.instanceID(of: $0) }
        let revisions = generate.mediaRevisions
        let snapshot = generate.draft
        let sourceAuthority = previewKey.authority
        let sourceRevision = library.sourceMediaCatalog.revision
        do {
            let data = try await hosts.backend(for: host).retainedSourceMediaContent(for: owner.print.filename, member: member)
            let staged = try await destination.applying(data: data, member: member, to: snapshot, recipe: recipe)
            guard !Task.isCancelled, library.allowsSource(previewKey.authority, detail: !gallery), hosts.host(owner.hostID) == host, hosts.instanceID(of: owner.hostID) == instance,
                  library.allowsSource(sourceAuthority, detail: !gallery), library.sourceMediaCatalog.revision == sourceRevision,
                  destinationRoute == target.flatMap(hosts.host),
                  destinationInstance == target.flatMap { hosts.instanceID(of: $0) },
                  revisions == generate.mediaRevisions,
                  fence.isCurrent(model: generate.modelName, host: target, recipe: self.recipe, media: generate.draft.media) else {
                if !Task.isCancelled { failure = "The draft or machine changed. Choose the source again to retry." }
                return
            }
            var current = generate.draft
            current.media = staged.media
            if current.width == snapshot.width { current.width = staged.width }
            if current.height == snapshot.height { current.height = staged.height }
            generate.draft = current
            dismiss(); router.openedPrint = nil; router.selection = .go(.generate)
        } catch {
            if !Task.isCancelled { failure = error.localizedDescription }
        }
    }
}
