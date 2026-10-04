import AppKit
import MoldClient
import SwiftUI

// Where a picked reference goes. The choosing is `PictureWell`'s; the ORDER is
// this strip's, and it is the whole reason the wells hand pictures over one at
// a time in pick order: on a `primaryIsTarget` recipe index 0 is the picture
// being EDITED, so a drop that landed in completion order edited the wrong
// picture.
extension ReferenceStrip {
    func append(_ picked: ImportedPicture, session: ReferenceImportSession) {
        guard session.isCurrent(controller: controller, media: draft.media) else { return }
        DraftPictureAttachment.addReference(
            picked, to: &draft, capability: capability, recipe: recipe)
        session.advance(controller: controller, media: draft.media)
    }

    /// Replace swaps ONE slot in place, so the strip's order -- and Qwen's
    /// Target at index 0 -- is untouched. A slot that went away while the
    /// panel was open is left alone rather than appended to the end.
    func replace(_ picked: ImportedPicture, at index: Int, session: ReferenceImportSession) {
        guard session.isCurrent(controller: controller, media: draft.media) else { return }
        guard draft.media.editImages.indices.contains(index) else { return }
        DraftPictureAttachment.replaceReference(picked, at: index, in: &draft, recipe: recipe)
    }
}
