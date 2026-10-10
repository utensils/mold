import MoldClient
import SwiftUI

/// One exact publication-owned member; preview and original bytes stay separate.
struct SourceMediaCard: View {
    @Environment(HostStore.self) private var hosts
    @Environment(GenerateController.self) private var generate
    @Environment(ModelStore.self) private var models
    @Environment(\.sourceMediaStaged) private var onStaged
    let owner: LibraryEntry
    let member: RetainedSourceMedia.Member
    var gallery = false
    @State private var preview: NSImage?
    @State private var choice: SourceMediaAttachment?
    @State private var failure: String?
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
        VStack(alignment: .leading, spacing: 8) {
            let layout = gallery ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10)) : AnyLayout(HStackLayout(alignment: .top, spacing: 12))
            layout {
                Group {
                    if let preview { Image(nsImage: preview).resizable().scaledToFit() }
                    else { Image(systemName: member.sourceSymbol).font(.largeTitle).foregroundStyle(.secondary) }
                }
                .frame(maxWidth: gallery ? .infinity : 72)
                .frame(height: gallery ? 148 : 72)
                .background(.quaternary.opacity(gallery ? 0.4 : 0), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(member.sourceLabel).font(.headline)
                    Text(member.displayName).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                    Text(ByteCountFormatter.string(fromByteCount: Int64(member.sizeBytes), countStyle: .file))
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if member.role.hasPrefix("matting_processed") {
                Text("Processed image. Reuse adds it as a new input.").font(.caption).foregroundStyle(.secondary)
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
                    .font(.caption).foregroundStyle(.secondary)
            }
            if choice != nil { ProgressView("Preparing source…") }
            if let failure { Text(failure).font(.caption).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(gallery ? 12 : 0)
        .background(.quaternary.opacity(gallery ? 0.25 : 0), in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .contain)
        .task(id: hosts.host(owner.hostID)) { await loadPreview() }
        .task(id: choice) {
            guard let choice else { return }
            await attach(choice)
            self.choice = nil
        }
    }

    private func loadPreview() async {
        preview = nil
        guard member.contentKind == "image", let host = hosts.host(owner.hostID) else { return }
        let instance = hosts.instanceIDs[owner.hostID]
        do {
            let data = try await hosts.backend(for: host).retainedSourceMediaThumbnail(for: owner.print.filename, member: member.memberId)
            guard !Task.isCancelled, hosts.host(owner.hostID) == host, hosts.instanceIDs[owner.hostID] == instance else { return }
            preview = NSImage(data: data)
        } catch { /* The labeled media tile remains available without a preview. */ }
    }

    private func attach(_ destination: SourceMediaAttachment) async {
        guard let host = hosts.host(owner.hostID), let recipe else { return }
        let instance = hosts.instanceIDs[owner.hostID]
        let fence = ReferenceImportFence(model: generate.modelName, host: target, recipe: recipe, media: generate.draft.media)
        let destinationRoute = target.flatMap(hosts.host)
        let destinationInstance = target.flatMap { hosts.instanceIDs[$0] }
        let revisions = generate.mediaRevisions
        let snapshot = generate.draft
        do {
            let data = try await hosts.backend(for: host).retainedSourceMediaContent(for: owner.print.filename, member: member)
            let staged = try await destination.applying(data: data, member: member, to: snapshot, recipe: recipe)
            guard !Task.isCancelled, hosts.host(owner.hostID) == host, hosts.instanceIDs[owner.hostID] == instance,
                  destinationRoute == target.flatMap(hosts.host),
                  destinationInstance == target.flatMap { hosts.instanceIDs[$0] },
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
            onStaged()
        } catch {
            if !Task.isCancelled { failure = error.localizedDescription }
        }
    }
}

extension EnvironmentValues {
    @Entry var sourceMediaStaged: () -> Void = {}
}
