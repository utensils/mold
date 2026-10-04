import MoldClient
import SwiftUI

/// Opening and closing anchors, rather than interpolation points.
struct BoundaryFrameWells: View {
    let recipe: GenerationRecipe
    @Environment(GenerateController.self) var controller
    @Binding var draft: RenderDraft

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            well(first: true)
            well(first: false)
        }
        .onChange(of: draft.frames) {
            BoundaryFramePolicy.apply(to: &draft, capabilities: recipe.capabilities)
        }
    }

    private func well(first: Bool) -> some View {
        let session = ReferenceImportSession(controller: controller, recipe: recipe, media: draft.media)
        return VStack(alignment: .leading, spacing: 4) {
            PictureWell(
                rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
                picture: BoundaryFramePolicy.image(first: first, draft: draft, capabilities: recipe.capabilities),
                label: first ? "First frame" : "Last frame",
                pick: { picked in
                    guard session.isCurrent(controller: controller, media: draft.media) else { return }
                    BoundaryFramePolicy.set(first: first, picture: picked, draft: &draft,
                                            capabilities: recipe.capabilities)
                })
            HStack {
                Text(first ? "First frame" : "Last frame").font(.caption)
                if BoundaryFramePolicy.image(first: first, draft: draft, capabilities: recipe.capabilities) != nil {
                    Button("Remove", systemImage: "xmark.circle") {
                        BoundaryFramePolicy.set(first: first, picture: nil, draft: &draft,
                                                capabilities: recipe.capabilities)
                    }.labelStyle(.iconOnly).buttonStyle(.plain)
                }
            }
        }
    }
}
