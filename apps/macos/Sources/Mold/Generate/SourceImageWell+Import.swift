import AppKit
import MoldClient
import SwiftUI

// What this well does with a picture once `PictureWell` has one: the draft
// writes, and the canvas an attached source moves to. Split from the well's
// own shape purely for size.
extension SourceImageWell {
    func clear() {
        draft.media.sourceImage = nil
        draft.media.sourceImageName = nil
        draft.media.sourceImageOriginal = nil
        draft.media.sourceImageOriginalName = nil
        draft.media.sourceImagePixels = nil
    }

    /// `PictureImport` has already read, conformed and base64'd it off the
    /// main actor, so the draft holds exactly what will be sent.
    func apply(_ picked: ImportedPicture) {
        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: recipe)
    }
}
