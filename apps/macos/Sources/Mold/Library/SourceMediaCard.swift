import MoldClient
import SwiftUI

/// One exact publication-owned member; preview and original bytes stay separate.
struct SourceMediaCard<Content: View>: View {
    @Environment(LibraryStore.self) private var library
    @Environment(HostStore.self) private var hosts
    @Environment(GenerateController.self) private var generate
    @Environment(ModelStore.self) private var models
    @Environment(\.sourceMediaStaged) private var onStaged
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    var gallery = false
    var showsPreview = true
    var openSource: (() -> Void)?
    @ViewBuilder var content: () -> Content
    @State private var choosingModel = false
    @State private var choice: SourceMediaAttachment?
    @State private var failure: String?
    private var previewKey: SourcePreviewKey {
        .init(authority: .init(owner: owner, route: hosts.host(owner.hostID), instance: hosts.instanceIDs[owner.hostID]), member: member)
    }
    private var recipe: GenerationRecipe? { {
        guard let id = generate.machineChoice ?? generate.hostID ?? hosts.preferredHost?.id,
              let name = generate.modelName, let model = models.model(named: name, on: id) else { return nil }
        return (generate.recipeID.flatMap { model.generationProfile?.recipe(named: $0) } ?? model.defaultRecipe)?
            .resolvingReferenceCapabilities(family: model.family, model: model.name)
    }() }
    private var target: UUID? { generate.machineChoice ?? generate.hostID ?? hosts.preferredHost?.id }
    private var choices: [SourceMediaAttachment] {
        SourceMediaAttachment.choices(member: member, recipe: recipe, media: generate.draft.media)
    }

    var body: some View {
        SourceMediaActionsView(owner: owner, member: member, gallery: gallery) {
        VStack(alignment: .leading, spacing: 8) {
            content()
            let layout = gallery ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10)) : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            if showsPreview { layout {
                SourceMediaThumbnail(owner: owner, member: member, gallery: gallery)
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
                    if !gallery { Text(member.displayName).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
                    Text(ByteCountFormatter.string(fromByteCount: Int64(member.sizeBytes), countStyle: .file))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            }
            if let openSource, !gallery {
                Button("Open source", systemImage: "arrow.up.left.and.arrow.down.right", action: openSource)
                    .buttonStyle(.borderless)
            }
            if member.role.hasPrefix("matting_processed") {
                Text("Processed image. Reuse adds it as a new input.").font(.caption).foregroundStyle(.secondary)
            }
            if member.contentKind != "unknown" {
                Menu { RowActionMenu(actions: reuseMenu, perform: perform) } label: { Label("Use source", systemImage: "arrow.up.forward") }
                    .disabled(choice != nil)
                    .help("Attach this original to Generate as a supported input.")
                    .accessibilityIdentifier("source-media-reuse")
            } else {
                Text("Open this source to check which inputs it supports.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if choice != nil { ProgressView("Preparing source…") }
            if let failure { Text(failure).font(.caption).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        } extra: {
            RowActionMenu(actions: SourceMediaReuseAction.menu(choices: choices, canOpen: openSource != nil,
                knownKind: member.contentKind != "unknown", busy: choice != nil), perform: perform)
        }
        .sheet(isPresented: $choosingModel) {
            SourceMediaModelChooser(member: member) { host, model, recipe, role in
                guard choice == nil, library.allowsSource(previewKey.authority, detail: !gallery), hosts.host(host.id) == host, hosts.isUp(host),
                      models.model(named: model.name, on: host.id) == model,
                      SourceMediaReuseOffer.offers(model: model, member: member, draft: generate.draft)
                        .contains(where: { $0.recipe == recipe && $0.choices.contains(role) }) else {
                    failure = "The model or machine changed. Choose the source again to retry."
                    return
                }
                generate.adopt(model: model, on: host.id, keepingDraft: true, selectedRecipe: recipe)
                failure = nil; choice = role
            }
        }
        .accessibilityElement(children: .contain)
        .task(id: choice) {
            guard let choice else { return }
            await attach(choice)
            self.choice = nil
        }
    }

    private var reuseMenu: [RowAction<SourceMediaReuseAction>] {
        SourceMediaReuseAction.menu(choices: choices, canOpen: false,
                                   knownKind: member.contentKind != "unknown", busy: choice != nil)
    }

    private func perform(_ action: SourceMediaReuseAction) {
        switch action {
        case .open: openSource?()
        case .chooseModel: if choice == nil { choosingModel = true }
        case let .attach(destination):
            guard choice == nil, choices.contains(destination) else { return }
            failure = nil; choice = destination
        }
    }

    private func attach(_ destination: SourceMediaAttachment) async {
        guard library.allowsSource(previewKey.authority, detail: !gallery), let host = hosts.host(owner.hostID), let recipe else {
            failure = "This source or model is no longer available. Choose the source again to retry."
            return
        }
        let instance = hosts.instanceIDs[owner.hostID]
        let fence = ReferenceImportFence(model: generate.modelName, host: target, recipe: recipe, media: generate.draft.media)
        let destinationRoute = target.flatMap(hosts.host)
        let destinationInstance = target.flatMap({ hosts.instanceIDs[$0] })
        let revisions = generate.mediaRevisions
        let snapshot = generate.draft
        let sourceAuthority = previewKey.authority
        let sourceRevision = library.sourceMediaCatalog.revision
        do {
            let data = try await hosts.backend(for: host).retainedSourceMediaContent(for: owner.print.filename, member: member)
            let staged = try await destination.applying(data: data, member: member, to: snapshot, recipe: recipe)
            guard !Task.isCancelled, library.allowsSource(previewKey.authority, detail: !gallery), hosts.host(owner.hostID) == host, hosts.instanceIDs[owner.hostID] == instance,
                  library.allowsSource(sourceAuthority, detail: !gallery), library.sourceMediaCatalog.revision == sourceRevision,
                  destinationRoute == target.flatMap(hosts.host),
                  destinationInstance == target.flatMap({ hosts.instanceIDs[$0] }),
                  revisions == generate.mediaRevisions,
                  fence.isCurrent(model: generate.modelName, host: target, recipe: self.recipe, media: generate.draft.media) else {
                if !Task.isCancelled { failure = "The draft or machine changed. Choose the source again to retry." }
                return
            }
            var current = generate.draft
            current.media = staged.media
            if current.width == snapshot.width { current.width = staged.width }
            if current.height == snapshot.height { current.height = staged.height }
            if current.frames == snapshot.frames { current.frames = staged.frames }
            generate.draft = current
            onStaged()
        } catch {
            guard !Task.isCancelled, library.allowsSource(sourceAuthority, detail: !gallery),
                  library.sourceMediaCatalog.revision == sourceRevision,
                  hosts.host(owner.hostID) == host, hosts.instanceIDs[owner.hostID] == instance,
                  destinationRoute == target.flatMap(hosts.host),
                  destinationInstance == target.flatMap({ hosts.instanceIDs[$0] }), revisions == generate.mediaRevisions,
                  fence.isCurrent(model: generate.modelName, host: target, recipe: self.recipe, media: generate.draft.media)
            else { return }
            failure = error.localizedDescription
        }
    }
}

extension EnvironmentValues {
    @Entry var sourceMediaStaged: () -> Void = {}
}

extension SourceMediaCard where Content == EmptyView {
    init(owner: LibraryEntry, member: RetainedSourceMedia.Member, gallery: Bool = false,
         showsPreview: Bool = true, openSource: (() -> Void)? = nil) {
        self.owner = owner; self.member = member; self.gallery = gallery
        self.showsPreview = showsPreview; self.openSource = openSource
        self.content = { EmptyView() }
    }
}
