import AppKit
import MoldClient
import SwiftUI

/// The still a render is conditioned on.
///
/// Only shown when the recipe says it reads one. A well on a text-only model
/// would collect bytes the host is going to refuse. Everything about CHOOSING
/// a picture -- the menu, the Library sheet, the panel, Paste, the drop, the
/// import -- is `PictureWell`'s; what is left here is the only thing that is
/// this well's own: where the bytes go, and the canvas a newly attached source
/// moves to.
struct SourceImageWell: View {
    @Binding var draft: RenderDraft
    /// The recipe this picture is being attached FOR -- read only to decide
    /// the canvas a newly attached source moves to (`attachSourceShape`).
    var recipe: GenerationRecipe?
    /// Whether this recipe has a mask path at all -- `RefineGroup.maskCapable`'s
    /// answer, passed in rather than re-derived.
    var canEditMask: Bool = false
    /// Opens the mask editor. `nil` where there is none to open.
    var openMaskEditor: (() -> Void)?
    /// What this well is, under it. Passed in because only
    /// `ImageConditioningWells` knows whether this recipe has PARKED it.
    var caption: String?

    /// Not the strip's `plus`: the owner's complaint was two anonymous squares
    /// side by side, and a caption alone cannot be read at a glance.
    static let placeholderGlyph = "photo.badge.plus"

    var body: some View {
        PictureWell(
            rows: items,
            picture: draft.media.sourceImage,
            placeholder: Self.placeholderGlyph,
            caption: caption,
            label: "Source picture",
            help: "Drop a picture or a Library print, or click to choose one",
            seedsLibraryPicker: true,
            pick: apply,
            perform: perform)
            .refittingSource(draft: $draft)
    }

    /// The click menu and the contextual menu are the SAME list
    /// (`GenerateMenus.sourceWell`) -- one declaration, two surfaces.
    var items: [GenerateMenus.Row] {
        GenerateMenus.sourceWell(
            hasPicture: draft.media.sourceImage != nil,
            canEditMask: canEditMask, canPaste: PicturePaste.hasPicture)
    }
}

extension SourceImageWell {
    /// Only the rows the chooser does not own reach here.
    func perform(_ action: GenerateAction) {
        switch action {
        case .editMask: openMaskEditor?()
        case .removeSource: clear()
        default: break
        }
    }
}
