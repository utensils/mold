import MoldClient
import SwiftUI

/// The picture wells beside the prompt: the source still, the reference
/// strip, or both.
///
/// Which of the two draws is the RELATION's answer, never "a strip if the
/// recipe advertises one". `combines` -- every SD1.5 and SDXL recipe, the two
/// families most people use for img2img -- rides WITH a source image, its
/// strength and its mask, so an `else if` that hid the well took img2img and
/// inpainting off those models entirely (finding 02#1). `exclusive` keeps both
/// wells too and parks whichever is not in use, media intact.
struct ImageConditioningWells: View {
    let recipe: GenerationRecipe
    let model: Model?
    @Binding var draft: RenderDraft

    @Environment(GenerateController.self) private var controller

    var body: some View {
        let layout = Self.layout(recipe: recipe, model: model, media: draft.media)
        if layout.showsSourceWell || layout.references != nil {
            VStack(alignment: .trailing, spacing: 4) {
                HStack(alignment: .top, spacing: 8) {
                    if layout.showsSourceWell {
                        SourceImageWell(
                            draft: $draft,
                            recipe: recipe,
                            canEditMask: RefineGroup.maskCapable(recipe.capabilities),
                            openMaskEditor: { controller.showsMaskEditor = true },
                            caption: WellCaption.source(parked: layout.parked == .source))
                            .opacity(layout.parked == .source ? Self.parkedOpacity : 1)
                    }
                    if let references = layout.references {
                        ReferenceStrip(
                            capability: references, draft: $draft,
                            caption: WellCaption.references(
                                max: references.maxCount, parked: layout.parked == .references))
                            .opacity(layout.parked == .references ? Self.parkedOpacity : 1)
                    }
                }
                if let note = layout.note {
                    Text(note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: 220, alignment: .trailing)
                        .multilineTextAlignment(.trailing)
                }
            }
        }
    }

    /// A parked well is dimmed, not hidden: its picture is kept and comes
    /// straight back when the active one is removed. The dimming is the
    /// SECOND signal -- its caption says "(not used)" in words, because
    /// opacity tells you something is different and never what.
    private static let parkedOpacity = 0.45
}

extension ImageConditioningWells {
    /// What the pane draws, resolved purely so a test needs no view host.
    struct Layout: Equatable {
        let showsSourceWell: Bool
        /// The strip's own contract, or nil where no strip is drawn.
        let references: ReferenceImagesCapability?
        /// The well the EXCLUSIVE relation has parked, or nil.
        let parked: ExclusiveWell?
        /// The sentence a parked well renders.
        let note: String?
    }

    static func layout(
        recipe: GenerationRecipe, model: Model?, media: DraftMedia
    ) -> Layout {
        // The resolved contract, which falls back to the legacy family rule
        // ONLY where the host advertises no block at all (finding 01#4).
        let references = recipe.capabilities.referenceImages(
            family: model?.family, model: model?.name)
        let mode = SourceImageMode(references: references)
        let wells = mode == .singleOrReferences
            ? ExclusiveWells.resolve(
                hasSource: media.sourceImage != nil,
                referenceCount: media.editImages.count,
                lastWrite: media.lastExclusiveWrite)
            : nil
        return Layout(
            showsSourceWell: mode.showsSourceWell && PromptPanel.showsSourceWell(for: recipe),
            references: mode.showsReferenceStrip ? references : nil,
            parked: wells?.parked,
            note: wells?.parked == nil ? nil : ExclusiveWells.note)
    }
}
