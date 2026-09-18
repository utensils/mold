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
        // Studio's own predicate: the bytes are not the bytes that were
        // there, which a FIRST picture satisfies too (`CreatePage.vue:1151`).
        let replaced = draft.media.sourceImageOriginal != picked.encoded
        // The UNFITTED copy is what every later re-fit starts from: fitting an
        // already-fitted picture crops a crop. `sourceImage` below is the
        // first, unfitted showing of it; `refittingSource` replaces it the
        // moment the canvas is known.
        draft.media.sourceImageOriginal = picked.encoded
        draft.media.sourceImageOriginalName = picked.name
        draft.media.sourceImage = picked.encoded
        draft.media.sourceImageName = picked.name
        if let size = PictureImport.pixelSize(of: picked.data) {
            draft.media.sourceImagePixels = SourcePixels(width: size.width, height: size.height)
            draft.attachSourceShape(size, recipe: recipe, replaced: replaced)
        }
        // Last write wins on an EXCLUSIVE recipe: attaching here parks the
        // reference strip rather than refusing the drop (`ExclusiveWells`).
        draft.media.lastExclusiveWrite = .source
    }
}
