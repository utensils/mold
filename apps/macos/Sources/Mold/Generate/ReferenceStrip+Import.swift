import AppKit
import MoldClient
import SwiftUI

// Where a picked reference goes. The choosing is `PictureWell`'s; the ORDER is
// this strip's, and it is the whole reason the wells hand pictures over one at
// a time in pick order: on a `primaryIsTarget` recipe index 0 is the picture
// being EDITED, so a drop that landed in completion order edited the wrong
// picture.
extension ReferenceStrip {
    func append(_ picked: ImportedPicture) {
        DraftPictureAttachment.addReference(picked, to: &draft, capability: capability)
    }

    /// Replace swaps ONE slot in place, so the strip's order -- and Qwen's
    /// Target at index 0 -- is untouched. A slot that went away while the
    /// panel was open is left alone rather than appended to the end.
    func replace(_ picked: ImportedPicture, at index: Int) {
        guard draft.media.editImages.indices.contains(index) else { return }
        draft.media.editImages[index] = picked.encoded
        draft.media.lastExclusiveWrite = .references
    }
}
