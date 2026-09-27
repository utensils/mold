import Foundation

// A batch's fan-out, and the redacted request a placement preview is priced
// from. Split from `RenderRequest.swift` for size.
public extension RenderRequest {
    /// The requests one press of Generate submits.
    ///
    /// A batch of N is N INDEPENDENT one-output requests, because the server
    /// refuses any child whose `batchSize` is not 1 -- the batch's size is
    /// the LENGTH of this array, capped by
    /// `queue.heterogeneous_batch_max_outputs`. They share one prompt, one
    /// filing and one logical `batchId`, and differ only by seed, so four
    /// presses give four variations rather than four copies, mirroring the
    /// fleet's other GUI (`web/src/pages/CreatePage.vue:3457-3464`).
    ///
    /// `randomBase` is injected so the fan-out is a pure function a test can
    /// pin; production passes `UInt64.random(in: 0 ... UInt64(UInt32.max))`,
    /// the 32-bit range web draws from.
    static func batch(
        _ draft: RenderDraft, model: String, copies: Int, randomBase: UInt64,
        maxIdentityPhotos: Int = 0
    ) -> [GenerateRequest] {
        guard copies > 1 else {
            return [one(draft, model: model, maxIdentityPhotos: maxIdentityPhotos)]
        }
        let batchId = UUID().uuidString
        // An unlocked seed with copies > 1 mints `randomBase` client-side --
        // the alternative is N absent seeds, and a host that picks its own
        // seed per child would make the four siblings unrelated in a way
        // nobody can reproduce.
        let base = draft.locksSeed ? (draft.seed ?? randomBase) : randomBase
        return (0 ..< copies).map { index in
            var sibling = one(draft, model: model, maxIdentityPhotos: maxIdentityPhotos)
            // Wrapping: a seed near `UInt64.max` is legal and a trap is not
            // an answer.
            sibling.seed = base &+ UInt64(index)
            sibling.batchId = batchId
            sibling.batchIndex = index + 1
            sibling.batchCount = copies
            return sibling
        }
    }

    /// The request to PREVIEW a fan-out with.
    ///
    /// The count goes in `PlacementRequest.copies` and the request itself
    /// stays a single output, or the host multiplies the two and previews
    /// sixteen pictures for a batch of four
    /// (`studio/api/generationPlacement.ts:339-347`). REDACTED, because a
    /// preview prices a render rather than making one --
    /// `GenerateRequest.redactedForPlacement`.
    static func placement(
        _ draft: RenderDraft, model: String, maxIdentityPhotos: Int = 0
    ) -> GenerateRequest {
        one(draft, model: model, maxIdentityPhotos: maxIdentityPhotos).redactedForPlacement()
    }
}
