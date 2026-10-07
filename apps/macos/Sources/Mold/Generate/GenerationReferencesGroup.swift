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
                Menu("Add video or audio", systemImage: "plus") {
                    ForEach(capability.kinds.filter { ["video", "audio"].contains($0) }, id: \.self) { kind in
                        Button(kind.capitalized) { choose(kind: kind) }
                    }
                }
                .disabled(importing || draft.media.generationReferences.count >= capability.maxCount)
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

    private var imageAddWell: some View {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return PictureWell(
                        rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                        placeholder: "plus", size: ReferenceStrip.thumbnailSize,
                        caption: "Add image", label: "Add an image reference",
                        pick: { attachPicture($0, session: session) })
    }

    private func referenceRow(_ reference: GenerationReference, index: Int) -> some View {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return HStack {
            if reference.kind == "image" {
                PictureWell(
                    rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                    picture: reference.media.data ?? reuse.preview(for: reference, in: draft), size: ReferenceStrip.thumbnailSize,
                    label: "Replace image reference \(index + 1)",
                    pick: { attachPicture($0, replacing: index, expected: reference, session: session) })
            }
            Text("\(index + 1). \(reference.provenance?.name ?? reference.kind.capitalized)")
                .lineLimit(1).truncationMode(.middle)
            if reference.media.authority == "descriptor", reference.kind == "image", reuse.preview(for: reference, in: draft) == nil {
                Button(reuse.previewFailures.contains(reference) ? "Retry preview" : "Load preview") {
                    Task { await reuse.loadPreviews(in: draft) }
                }.font(.caption)
            }
            Spacer(minLength: 0)
            if reference.kind != "image" {
                Button("Replace", systemImage: "arrow.triangle.2.circlepath") { choose(kind: reference.kind, replacing: index) }
                    .labelStyle(.iconOnly).disabled(importing)
                    .help("Choose a different file for this reference")
            }
            Button("Move earlier", systemImage: "arrow.up") { draft.media.moveGenerationReference(from: index, to: index - 1) }
                .labelStyle(.iconOnly).disabled(index == 0)
                .help("Move this reference one place earlier")
            Button("Move later", systemImage: "arrow.down") { draft.media.moveGenerationReference(from: index, to: index + 1) }
                .labelStyle(.iconOnly).disabled(index == draft.media.generationReferences.count - 1)
                .help("Move this reference one place later")
            Button("Remove", systemImage: "xmark.circle") { draft.media.removeGenerationReference(at: index) }
                .labelStyle(.iconOnly)
                .help("Remove this reference from the next render")
        }.buttonStyle(.plain)
    }

    private func namedWell(_ role: GenerationImageReferenceRole) -> some View {
        let index = draft.media.generationReferences.firstIndex { $0.role == role }
        let expected = index.map { draft.media.generationReferences[$0] }
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return HStack {
            PictureWell(
                rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                picture: expected.flatMap { $0.media.data ?? reuse.preview(for: $0, in: draft) },
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
