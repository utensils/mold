import Foundation
import Testing
@testable import MoldClient

struct DraftPictureAttachmentTests {
    @Test func replacingSourceInvalidatesOnlyItsOldMask() {
        var draft = RenderDraft()
        draft.media.sourceImageOriginal = "old"
        draft.media.maskImage = "old-source-coordinates"
        let picked = ImportedPicture(encoded: "new", name: "new.png", data: Data())
        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: nil)
        #expect(draft.media.maskImage == nil)
        draft.media.maskImage = "new-source-coordinates"
        DraftPictureAttachment.useAsSource(picked, in: &draft, recipe: nil)
        #expect(draft.media.maskImage == "new-source-coordinates")
    }
}
