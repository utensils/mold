import Foundation
import MoldClient

// The print's own picture goes in the well, on every route.
extension ReuseStore {
    /// The draft with the print's picture placed in its source well, or nil
    /// when there is nothing to place: no authority, no `source_image`
    /// member, or a well already holding a picture of the person's own.
    ///
    /// Runs AFTER the probe and re-arms on the placed draft, so the authority
    /// survives the one edit this store made itself. A fetch that fails is
    /// said once, the way the long-clip route already says it, and the well
    /// stays empty rather than pretending.
    func placePicture(in draft: RenderDraft, outgoing: GenerateRequest?) async -> RenderDraft? {
        guard let authority = pending(for: draft),
              let member = RetainedSourcePicture.member(of: authority, forHydrating: outgoing)
        else { return nil }
        let fence = currentFence
        switch await RetainedSourcePicture.fetch(member, of: authority, hosts: hosts) {
        case let .refused(sentence):
            if isCurrent(fence), notice == nil { notice = sentence }
            return nil
        case let .picture(picture):
            guard isCurrent(fence), pending(for: draft) != nil else { return nil }
            var placed = draft
            RetainedSourcePicture.place(picture, named: authority.filename, in: &placed)
            arm(placed)
            return placed
        }
    }
}
