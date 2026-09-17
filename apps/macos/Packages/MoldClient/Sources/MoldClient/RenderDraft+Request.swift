import Foundation

// Turning a draft into what one press of Generate submits. Split from
// `RenderDraft+Recipe.swift` (which keeps `adopting` and `refusal`) purely
// for size.
public extension RenderDraft {
    /// One request, with `batchSize` always 1.
    ///
    /// `/api/generation-batches` refuses any child whose `batch_size` is not
    /// 1 (`crates/mold-server/src/queue_media_admission.rs:380-386`), so this
    /// never reads the draft's own `batchSize` -- see `requests(model:copies:
    /// randomBase:)` for how a batch of several is actually built.
    func request(model: String) -> GenerateRequest {
        var request = GenerateRequest(
            prompt: prompt, model: model, width: width, height: height,
            steps: steps, guidance: guidance, batchSize: 1,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: locksSeed ? seed : nil
        )
        request.frames = frames
        request.fps = fps
        request.sourceImage = sourceImage
        request.sourceImageName = sourceImageName
        request.editImages = editImages.isEmpty ? nil : editImages
        request.referenceWeight = editImages.isEmpty ? nil : referenceWeight
        // Strength only means something with something to apply it to.
        request.strength = sourceImage == nil ? nil : strength

        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        request.title = trimmedTitle.isEmpty ? nil : trimmedTitle
        // Folds the title into the tag list when the switch asks for it --
        // see `ClientTags.compose`.
        let composedTags = ClientTags.compose(
            explicit: tags, title: request.title, autoTagTitle: autoTagTitle
        ).tags
        request.tags = composedTags.isEmpty ? nil : composedTags
        request.collection = collectionName.map(CollectionRef.named)
        request.outputFormat = outputFormat
        request.upscaleModel = upscaleModel
        // Absent means the server's own default (save). `false` is the only
        // instruction worth sending over the wire.
        request.saveToGallery = savesToGallery ? nil : false
        request.originalPrompt = originalPrompt
        request.promptTransform = promptTransform
        return request
    }

    /// The requests one press of Generate submits.
    ///
    /// A batch of N is N INDEPENDENT one-output requests, because the server
    /// refuses any child whose `batchSize` is not 1 -- the batch's size is
    /// the LENGTH of this array, capped by
    /// `queue.heterogeneous_batch_max_outputs`. They share one prompt, one
    /// filing and one logical `batchId`, and differ only by seed, so four
    /// presses of one button give four variations rather than four copies.
    /// This mirrors the fleet's other GUI exactly
    /// (`web/src/pages/CreatePage.vue:3457-3464`).
    ///
    /// `randomBase` is injected so the fan-out is a pure function a test can
    /// pin; production passes `UInt64.random(in: 0 ... UInt64(UInt32.max))`,
    /// the 32-bit range web draws from.
    func requests(model: String, copies: Int, randomBase: UInt64) -> [GenerateRequest] {
        guard copies > 1 else { return [request(model: model)] }
        let batchId = UUID().uuidString
        // An unlocked seed with copies > 1 mints `randomBase` client-side --
        // the alternative is N absent seeds, and a host that picks its own
        // seed per child would make the four siblings unrelated in a way
        // nobody can reproduce.
        let base = locksSeed ? (seed ?? randomBase) : randomBase
        return (0 ..< copies).map { index in
            var sibling = request(model: model)
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
    /// (`studio/api/generationPlacement.ts:339-347`).
    func placementRequest(model: String) -> GenerateRequest {
        request(model: model)
    }
}
