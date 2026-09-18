import AppKit
import MoldClient
import SwiftUI

/// The still that steers a ControlNet render, bound to `draft.media.control`.
///
/// It WAS "a second instance of `SourceImageWell`'s idiom rather than a
/// genericized one", on the argument that the binding shapes differ -- this
/// one writes `control?.image`/`control?.name` behind an optional whose other
/// half, the chosen adapter, can already be set. That asymmetry is real, and
/// it is now the only thing in this file: the binding is what a caller keeps.
/// Everything else -- the menu, both doors, Paste, the drop, the import -- is
/// `PictureWell`'s, and folding it in is what gave this well a Library door
/// and a contextual menu it never had.
struct ControlPictureWell: View {
    @Binding var draft: RenderDraft

    var body: some View {
        PictureWell(
            rows: GenerateMenus.controlWell(
                hasPicture: draft.media.control?.image != nil,
                canPaste: PicturePaste.hasPicture),
            picture: draft.media.control?.image,
            placeholder: "wand.and.rays",
            caption: WellCaption.control,
            label: "Control picture",
            help: draft.media.control?.name
                ?? "Drop a control picture or a Library print, or click to choose one",
            pick: apply,
            perform: perform)
    }

    private func perform(_ action: GenerateAction) {
        guard action == .removeControl else { return }
        draft.media.control?.image = nil
        draft.media.control?.name = nil
        // An empty `ControlConditioning` is not a meaningful state to leave
        // sitting in the draft (`RefineGroup.controlModelBinding`'s rule).
        if draft.media.control?.model == nil { draft.media.control = nil }
    }

    private func apply(_ picked: ImportedPicture) {
        var control = draft.media.control ?? ControlConditioning()
        control.image = picked.encoded
        control.name = picked.name
        draft.media.control = control
    }
}
