import AppKit
import MoldClient
import SwiftUI
import UniformTypeIdentifiers

/// Typed references keep the host's semantic order across every media kind.
struct GenerationReferencesGroup: View {
    let recipe: GenerationRecipe
    @Binding var draft: RenderDraft
    @Environment(GenerateController.self) var controller
    @Environment(ReuseStore.self) var reuse
    @State var failure: String?
    @State var importing = false
    @State var importTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let capability = recipe.capabilities.generationReferences, capability.mode.isVisible {
                Text("\(draft.media.generationReferences.count) of \(capability.maxCount) references")
                    .font(.caption)
                Text(capability.requiresVisual ? "Add at least one image or video in semantic order." : "Add references in semantic order.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(Array(draft.media.generationReferences.enumerated()), id: \.offset) { index, reference in
                    referenceRow(reference, index: index)
                }
                if capability.kinds.contains("image"), draft.media.generationReferences.count < capability.maxCount {
                    imageAddWell
                        .disabled(importing)
                }
                if capability.kinds.contains(where: { ["video", "audio"].contains($0) }) {
                    Menu("Add video or audio", systemImage: "plus") {
                        ForEach(capability.kinds.filter { ["video", "audio"].contains($0) }, id: \.self) { kind in
                            Button(kind.capitalized) { choose(kind: kind) }
                        }
                    }
                    .disabled(importing || draft.media.generationReferences.count >= capability.maxCount)
                }
            }
            if let named = recipe.capabilities.mesh?.namedViews, named.mode.isVisible {
                ForEach(named.roles, id: \.self) { role in
                    namedWell(role)
                }
            }
            if let failure { Text(failure).font(.caption).foregroundStyle(.red) }
            if let reason = draft.media.generationReferenceError(capabilities: recipe.capabilities) {
                Text(reason).font(.caption).foregroundStyle(.secondary)
            }
        }
        .task(id: reuse.authority) { await reuse.loadPreviews(in: draft) }
        .task(id: reuse.currentFence) { await reuse.loadPreviews(in: draft) }
        .onChange(of: recipe) { importTask?.cancel(); importing = false }
        .onDisappear { importTask?.cancel() }
    }

    private func viewReference(_ reference: GenerationReference, index: Int) async throws -> SourceMediaFile {
        if let data = reference.media.data { return try await InputMediaPreview.file(base64: data) }
        guard let authority = reuse.pending(for: draft), let restored = reuse.restored,
              let route = reuse.hosts.host(authority.origin) else { throw MoldClientError.malformedResponse }
        let fence = reuse.currentFence
        let instance = reuse.hosts.instanceID(of: route.id)
        let file = try await InputMediaPreview.retainedFile(reference: reference, index: index,
            references: draft.media.generationReferences, original: restored.media.generationReferences,
            members: authority.members, filename: authority.filename, backend: reuse.hosts.backend(for: route))
        guard reuse.isCurrent(fence), reuse.hosts.host(route.id) == route,
              reuse.hosts.instanceID(of: route.id) == instance else { throw CancellationError() }
        return file
    }

    private var imageAddWell: some View {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return PictureWell(
            rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
            placeholder: "plus", actionTitle: "Add image",
            label: "Add an image reference",
            pick: { attachPicture($0, session: session) })
    }

    private func referenceRow(_ reference: GenerationReference, index: Int) -> some View {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return HStack(alignment: .top, spacing: GenerationReferenceLayout.columnGap) {
            if reference.kind == "image" {
                PictureWell(
                    rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                    picture: reference.media.data ?? reuse.preview(for: reference, in: draft),
                    originalIdentity: "\(reuse.currentFence)|\(String(describing: reuse.authority))|\(reference)",
                    originalLoader: { try await viewReference(reference, index: index) },
                    originalIsCurrent: { draft.media.generationReferences.indices.contains(index) && draft.media.generationReferences[index] == reference && (reference.media.data != nil || reuse.pending(for: draft) != nil) },
                    size: GenerationReferenceLayout.previewSize(count: draft.media.generationReferences.count),
                    label: "Replace image reference \(index + 1)",
                    pick: { attachPicture($0, replacing: index, expected: reference, session: session) })
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Reference \(index + 1) · \(reference.kind.capitalized)")
                    .font(.caption).foregroundStyle(.secondary)
                Text(reference.provenance?.name ?? reference.kind.capitalized)
                    .lineLimit(2).truncationMode(.middle)
                    .help(reference.provenance?.name ?? reference.kind.capitalized)
                if reference.media.authority == "descriptor", reference.kind == "image", reuse.preview(for: reference, in: draft) == nil {
                    Button(reuse.previewFailures.contains(reference) ? "Retry preview" : "Load preview") {
                        Task { await reuse.loadPreviews(in: draft) }
                    }.font(.caption)
                }
                if reference.kind != "image" {
                    InputMediaViewButton(title: reference.name, identity: "\(reuse.currentFence)|\(String(describing: reuse.authority))|\(index)|\(reference)",
                        load: { try await viewReference(reference, index: index) },
                        isCurrent: { draft.media.generationReferences.indices.contains(index) && draft.media.generationReferences[index] == reference && (reference.media.data != nil || reuse.pending(for: draft) != nil) })
                }
                HStack(spacing: GenerationReferenceLayout.actionGap) {
                    if reference.kind != "image" {
                        Button("Replace", systemImage: "arrow.triangle.2.circlepath") { choose(kind: reference.kind, replacing: index) }
                            .disabled(importing)
                            .help("Choose a different file for this reference")
                    }
                    Button("Move earlier", systemImage: "arrow.up") { draft.media.moveGenerationReference(from: index, to: index - 1) }
                        .disabled(index == 0)
                        .help("Move this reference one place earlier")
                    Button("Move later", systemImage: "arrow.down") { draft.media.moveGenerationReference(from: index, to: index + 1) }
                        .disabled(index == draft.media.generationReferences.count - 1)
                        .help("Move this reference one place later")
                    Button("Remove", systemImage: "xmark") { draft.media.removeGenerationReference(at: index) }
                        .help("Remove this reference from the next render")
                }
                .labelStyle(.iconOnly)
                .buttonStyle(ReferenceActionStyle())
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 4)
    }

    private func namedWell(_ role: GenerationImageReferenceRole) -> some View {
        let index = draft.media.generationReferences.firstIndex { $0.role == role }
        let expected = index.map { draft.media.generationReferences[$0] }
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return HStack {
            PictureWell(
                rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                picture: expected.flatMap { $0.media.data ?? reuse.preview(for: $0, in: draft) },
                originalIdentity: "\(reuse.currentFence)|\(String(describing: reuse.authority))|\(String(describing: expected))",
                originalLoader: expected.map { ref in { try await viewReference(ref, index: index ?? -1) } },
                originalIsCurrent: { expected == draft.media.generationReferences.first(where: { $0.role == role }) && (expected?.media.data != nil || reuse.pending(for: draft) != nil) },
                size: ReferenceStrip.thumbnailSize, caption: role.rawValue.capitalized,
                label: "\(role.rawValue.capitalized) camera view",
                pick: { attachPicture($0, replacing: index, role: role,
                                      expected: expected, session: session) })
            if let index {
                Text(draft.media.generationReferences[index].provenance?.name ?? "Image").lineLimit(1)
                Button("Remove", systemImage: "xmark.circle") {
                    draft.media.removeGenerationReference(at: index)
                }.labelStyle(.iconOnly)
                .help("Remove the \(role.rawValue) camera-view picture")
            }
        }.disabled(importing)
    }


}

extension GenerationReferencesGroup {
    static func isShown(_ capabilities: RecipeCapabilities) -> Bool {
        capabilities.generationReferences?.mode.isVisible == true
            || capabilities.mesh?.namedViews?.mode.isVisible == true
    }
}

/// These dimensions leave an 80 pt action row beside the preview even in
/// the minimum 260 pt inspector after its outer and disclosure insets.
enum GenerationReferenceLayout {
    static let columnGap: CGFloat = 12
    static let actionSize: CGFloat = 24
    static let actionGap: CGFloat = 4
    static func previewSize(count: Int) -> CGFloat { count == 1 ? 104 : 80 }
}

private struct ReferenceActionStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: GenerationReferenceLayout.actionSize, height: GenerationReferenceLayout.actionSize)
            .background(.quaternary.opacity(configuration.isPressed ? 0.8 : 0.4), in: RoundedRectangle(cornerRadius: 5))
            .contentShape(Rectangle())
    }
}
